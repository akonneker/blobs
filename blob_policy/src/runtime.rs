//! Versioned frozen policy weights and deterministic scalar CPU execution.
//!
//! Linear weights retain Burn's [input, output] layout. Inference uses explicit
//! f32 multiply/add order and portable libm transcendental functions on both
//! native and WASM. Value and learned-gate diagnostic heads are not deployed.
use crate::{action::*, greedy::*, memory::*, observation::*};
use blob_interface::reference_mind::{ReferenceMind, ReferenceMindDecision, ReferenceMindInput};

pub const POLICY_WEIGHT_FORMAT_VERSION: u32 = 1;
pub const POLICY_EXPORT_SCHEMA_VERSION: u32 = 1;
pub const POLICY_BUILD_SCHEMA_VERSION: u32 = 1;
pub const POLICY_VERIFICATION_SCHEMA_VERSION: u32 = 1;
pub const EXECUTION_CONTRACT: &str = "blob.policy.scalar-f32-libm.v1";
const MAGIC: &[u8; 8] = b"BLPOL001";
pub const MAX_WEIGHT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct Linear {
    inputs: usize,
    outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    pub fn new(
        inputs: usize,
        outputs: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, String> {
        if inputs.checked_mul(outputs) != Some(weights.len())
            || bias.len() != outputs
            || !weights.iter().chain(&bias).all(|x| x.is_finite())
        {
            return Err("invalid linear shape or non-finite weights".into());
        }
        Ok(Self {
            inputs,
            outputs,
            weights,
            bias,
        })
    }
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.inputs);
        // Add bias after the dot product, matching the training linear layer.
        let mut out = vec![0.0; self.outputs];
        for (value, row) in x.iter().zip(self.weights.chunks_exact(self.outputs)) {
            for (dst, weight) in out.iter_mut().zip(row) {
                *dst += value * weight;
            }
        }
        for (dst, bias) in out.iter_mut().zip(&self.bias) {
            *dst += bias;
        }
        out
    }
}

/// Layer order is part of format v1. Shapes are derived, never trusted from bytes.
pub fn layer_shapes(
    h1: usize,
    h2: usize,
    memory: usize,
) -> Result<Vec<(&'static str, usize, usize)>, String> {
    if [h1, h2, memory].iter().any(|n| !(1..=512).contains(n)) {
        return Err("deployment dimensions must be in 1..=512".into());
    }
    Ok(vec![
        ("slot_encoder", SLOT_FEATURES, h2),
        ("shared_fc1", HEADER_FEATURES + h2, h1),
        ("recurrent", h1 + memory, memory),
        ("shared_fc2", memory, h2),
        ("foraging_action_kind_head", h2, 10),
        ("foraging_adapter_fc", h2 + 8, 16),
        ("foraging_adapter_head", 16, 10),
        ("interaction_action_kind_head", h2, 10),
        (
            "interaction_adapter_fc",
            h2 + OBS_RANDOMNESS_FEATURE_START,
            16,
        ),
        ("interaction_adapter_head", 16, 10),
        (
            "interaction_slot_adapter_fc",
            OBS_RANDOMNESS_FEATURE_START + SLOT_FEATURES,
            16,
        ),
        ("interaction_slot_adapter_head", 16, 10),
        ("exploration_action_kind_head", h2, 10),
        (
            "exploration_adapter_fc",
            h2 + OBS_RANDOMNESS_FEATURE_START,
            16,
        ),
        ("exploration_adapter_head", 16, 10),
        (
            "exploration_slot_adapter_fc",
            OBS_RANDOMNESS_FEATURE_START + SLOT_FEATURES,
            16,
        ),
        ("exploration_slot_adapter_head", 16, 10),
        ("exploration_guard_readiness_head", 2, 1),
        ("target_query_head", h2, NUM_POLICY_ACTION_KINDS * h2),
        ("effort_head", h2, NUM_POLICY_EFFORT_LOGITS),
        ("amount_head", h2, NUM_POLICY_AMOUNT_LOGITS),
        ("signal_head", h2, NUM_SIGNAL_CHOICES),
        ("signal_strength_head", h2, NUM_SIGNAL_STRENGTH_CHOICES),
        ("target_residual.fc", 32 + SLOT_FEATURES, 32),
        ("target_residual.head", 32, 10),
    ])
}

#[derive(Clone, Debug)]
pub struct FrozenPolicy {
    hidden1: usize,
    hidden2: usize,
    memory: usize,
    layers: Vec<Linear>,
}

pub struct PolicyOutput {
    pub kind: Vec<f32>,
    pub target: Vec<f32>,
    pub effort: Vec<f32>,
    pub amount: Vec<f32>,
    pub signal: Vec<f32>,
    pub signal_strength: Vec<f32>,
    pub next_memory: Vec<f32>,
}

/// Read-only diagnostics; never serialized into weights or a Mind decision.
pub struct PolicyTrace {
    pub context: ObservationExpertContext,
    pub hidden: Vec<f32>,
    pub base_kind: Vec<f32>,
    pub context_delta: Vec<f32>,
    pub slot_delta: Vec<f32>,
    pub raw_kind: Vec<f32>,
}

impl FrozenPolicy {
    pub fn new(
        hidden1: usize,
        hidden2: usize,
        memory: usize,
        layers: Vec<Linear>,
    ) -> Result<Self, String> {
        let shapes = layer_shapes(hidden1, hidden2, memory)?;
        if layers.len() != shapes.len()
            || layers
                .iter()
                .zip(&shapes)
                .any(|(l, (_, i, o))| l.inputs != *i || l.outputs != *o)
        {
            return Err("policy layers do not match deployment architecture".into());
        }
        let size = 20
            + shapes
                .iter()
                .map(|(_, i, o)| (i * o + o) * 4)
                .sum::<usize>();
        if size > MAX_WEIGHT_BYTES {
            return Err("deployment weights exceed byte budget".into());
        }
        Ok(Self {
            hidden1,
            hidden2,
            memory,
            layers,
        })
    }
    pub fn recurrent_size(&self) -> usize {
        self.memory
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        for n in [self.hidden1, self.hidden2, self.memory] {
            bytes.extend_from_slice(&(n as u32).to_le_bytes());
        }
        for layer in &self.layers {
            for value in layer.weights.iter().chain(&layer.bias) {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 20 || bytes.len() > MAX_WEIGHT_BYTES || &bytes[..8] != MAGIC {
            return Err("invalid deployment weight header/size".into());
        }
        let dim =
            |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let (h1, h2, memory) = (dim(8), dim(12), dim(16));
        let shapes = layer_shapes(h1, h2, memory)?;
        let expected = 20
            + shapes
                .iter()
                .map(|(_, i, o)| (i * o + o) * 4)
                .sum::<usize>();
        if bytes.len() != expected {
            return Err("deployment weight length mismatch".into());
        }
        let mut values = bytes[20..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        let layers = shapes
            .into_iter()
            .map(|(_, i, o)| {
                Linear::new(
                    i,
                    o,
                    values.by_ref().take(i * o).collect(),
                    values.by_ref().take(o).collect(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(h1, h2, memory, layers)
    }
    pub fn forward(&self, observation: &[f32], memory: &[f32]) -> Result<PolicyOutput, String> {
        self.forward_impl(observation, memory, None, None)
    }

    pub fn forward_with_trace(
        &self,
        observation: &[f32],
        memory: &[f32],
    ) -> Result<(PolicyOutput, PolicyTrace), String> {
        let mut trace = None;
        let output = self.forward_impl(observation, memory, Some(&mut trace), None)?;
        Ok((output, trace.expect("requested trace is populated")))
    }

    /// Add a finite action-kind residual before the legacy probability transform.
    /// An exactly zero residual preserves every original output bit.
    pub fn forward_with_kind_delta(
        &self,
        observation: &[f32],
        memory: &[f32],
        delta: &[f32],
    ) -> Result<PolicyOutput, String> {
        if delta.len() != NUM_POLICY_ACTION_KINDS || !delta.iter().all(|x| x.is_finite()) {
            return Err("invalid kind residual".into());
        }
        self.forward_impl(observation, memory, None, Some(delta))
    }
    fn forward_impl(
        &self,
        observation: &[f32],
        memory: &[f32],
        trace: Option<&mut Option<PolicyTrace>>,
        delta: Option<&[f32]>,
    ) -> Result<PolicyOutput, String> {
        if observation.len() != OBS_DIM
            || memory.len() != self.memory
            || !observation.iter().chain(memory).all(|x| x.is_finite())
        {
            return Err("invalid policy input shape or values".into());
        }
        let header = &observation[..HEADER_FEATURES];
        let slots = observation[HEADER_FEATURES..]
            .chunks_exact(SLOT_FEATURES)
            .collect::<Vec<_>>();
        let encoded = slots
            .iter()
            .map(|s| relu(self.layers[0].forward(s)))
            .collect::<Vec<_>>();
        let pooled = max_pool(&encoded);
        let trunk = relu(self.layers[1].forward(&join(header, &pooled)));
        let next_memory = self.layers[2]
            .forward(&join(&trunk, memory))
            .into_iter()
            .map(libm::tanhf)
            .collect::<Vec<_>>();
        let x = relu(self.layers[3].forward(&next_memory));
        let context = observation_expert_context(observation);
        let context_features = join(&x, &header[..OBS_RANDOMNESS_FEATURE_START]);
        let raw_summary = (0..SLOT_FEATURES)
            .map(|i| slots.iter().map(|s| s[i]).fold(f32::NEG_INFINITY, f32::max))
            .collect::<Vec<_>>();
        let local = join(&header[..OBS_RANDOMNESS_FEATURE_START], &raw_summary);
        let mut kind = match context {
            ObservationExpertContext::Foraging => {
                let features = [1, 2, 10, 11, 12, 15, 19, 20].map(|i| header[i]);
                add(
                    self.layers[4].forward(&x),
                    self.layers[6].forward(&relu(self.layers[5].forward(&join(&x, &features)))),
                )
            }
            ObservationExpertContext::Interaction => add(
                add(
                    self.layers[7].forward(&x),
                    self.layers[9].forward(&relu(self.layers[8].forward(&context_features))),
                ),
                self.layers[11].forward(&relu(self.layers[10].forward(&local))),
            ),
            ObservationExpertContext::Exploration => {
                let mut y = add(
                    add(
                        self.layers[12].forward(&x),
                        self.layers[14].forward(&relu(self.layers[13].forward(&context_features))),
                    ),
                    self.layers[16].forward(&relu(self.layers[15].forward(&local))),
                );
                y[PolicyActionKind::Guard.index()] += self.layers[17].forward(&header[1..3])[0];
                y
            }
        };
        if let Some(trace) = trace {
            let (base_kind, context_delta, slot_delta) = match context {
                ObservationExpertContext::Foraging => {
                    let features = [1, 2, 10, 11, 12, 15, 19, 20].map(|i| header[i]);
                    (
                        self.layers[4].forward(&x),
                        self.layers[6].forward(&relu(self.layers[5].forward(&join(&x, &features)))),
                        vec![0.; NUM_POLICY_ACTION_KINDS],
                    )
                }
                ObservationExpertContext::Interaction => (
                    self.layers[7].forward(&x),
                    self.layers[9].forward(&relu(self.layers[8].forward(&context_features))),
                    self.layers[11].forward(&relu(self.layers[10].forward(&local))),
                ),
                ObservationExpertContext::Exploration => {
                    let mut slot = self.layers[16].forward(&relu(self.layers[15].forward(&local)));
                    slot[PolicyActionKind::Guard.index()] +=
                        self.layers[17].forward(&header[1..3])[0];
                    (
                        self.layers[12].forward(&x),
                        self.layers[14].forward(&relu(self.layers[13].forward(&context_features))),
                        slot,
                    )
                }
            };
            *trace = Some(PolicyTrace {
                context,
                hidden: x.clone(),
                base_kind,
                context_delta,
                slot_delta,
                raw_kind: kind.clone(),
            });
        }
        if let Some(delta) = delta {
            for (value, adjustment) in kind.iter_mut().zip(delta) {
                if *adjustment != 0. {
                    *value += adjustment;
                }
            }
        }
        // Preserve the training softmax/clamp/log contract, including floor ties.
        let max = kind.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for k in &mut kind {
            *k = libm::expf(*k - max);
        }
        let sum: f32 = kind.iter().sum();
        for k in &mut kind {
            *k = libm::logf((*k / sum).max(1e-20));
        }
        let queries = self.layers[18].forward(&x);
        let residuals = slots
            .iter()
            .map(|s| {
                self.layers[24].forward(&relu(self.layers[23].forward(&join(
                    &header[OBS_RANDOMNESS_FEATURE_START..OBS_RANDOMNESS_FEATURE_END],
                    s,
                ))))
            })
            .collect::<Vec<_>>();
        let mut target = vec![0.0; NUM_POLICY_TARGET_LOGITS];
        let scale = (self.hidden2 as f64).sqrt() as f32;
        for k in 0..NUM_POLICY_ACTION_KINDS {
            for s in 0..NUM_POLICY_TARGETS {
                let mut dot = 0.0;
                for i in 0..self.hidden2 {
                    dot += queries[k * self.hidden2 + i] * encoded[s][i];
                }
                target[k * NUM_POLICY_TARGETS + s] = dot / scale + residuals[s][k];
            }
        }
        let out = PolicyOutput {
            kind,
            target,
            effort: self.layers[19].forward(&x),
            amount: self.layers[20].forward(&x),
            signal: self.layers[21].forward(&x),
            signal_strength: self.layers[22].forward(&x),
            next_memory,
        };
        if !out
            .kind
            .iter()
            .chain(&out.target)
            .chain(&out.effort)
            .chain(&out.amount)
            .chain(&out.signal)
            .chain(&out.signal_strength)
            .chain(&out.next_memory)
            .all(|x| x.is_finite())
        {
            return Err("non-finite deployment inference output".into());
        }
        Ok(out)
    }
    pub fn try_decide(&self, input: &ReferenceMindInput) -> Result<ReferenceMindDecision, String> {
        let observation = Observation::from_reference(input);
        let out = self.forward(
            &observation.data,
            &decode_policy_memory(&input.private_memory, self.memory),
        )?;
        Ok(Self::decode_output(input, &observation, &out))
    }
    pub(crate) fn decode_output(
        input: &ReferenceMindInput,
        observation: &Observation,
        out: &PolicyOutput,
    ) -> ReferenceMindDecision {
        let action = greedy_policy_action(&out.kind, &out.target, &out.effort, observation);
        let kind = decompose_policy_action(action).unwrap().kind;
        let amount = greedy_amount(
            &out.amount[kind * NUM_AMOUNT_CHOICES..(kind + 1) * NUM_AMOUNT_CHOICES],
            observation,
            action,
        );
        let signal = greedy_signal(&out.signal, observation, action, amount);
        let signal_strength =
            greedy_signal_strength(&out.signal_strength, observation, action, amount, signal);
        let mut decision = decode_policy_choice(
            PolicyChoice {
                action,
                amount,
                signal,
                signal_strength,
            },
            input,
        );
        attach_policy_memory(&mut decision, encode_policy_memory(&out.next_memory));
        decision
    }
}
impl ReferenceMind for FrozenPolicy {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        self.try_decide(input)
            .expect("validated frozen policy inference")
    }
    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}
fn relu(mut x: Vec<f32>) -> Vec<f32> {
    for v in &mut x {
        *v = v.max(0.0);
    }
    x
}
fn join(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().chain(b).copied().collect()
}
fn add(mut a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    for (a, b) in a.iter_mut().zip(b) {
        *a += b;
    }
    a
}
fn max_pool(rows: &[Vec<f32>]) -> Vec<f32> {
    (0..rows[0].len())
        .map(|i| rows.iter().map(|r| r[i]).fold(f32::NEG_INFINITY, f32::max))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> FrozenPolicy {
        FrozenPolicy::new(
            4,
            3,
            2,
            layer_shapes(4, 3, 2)
                .unwrap()
                .into_iter()
                .map(|(_, i, o)| Linear::new(i, o, vec![0.01; i * o], vec![0.0; o]).unwrap())
                .collect(),
        )
        .unwrap()
    }
    #[test]
    fn weights_roundtrip_and_reject_bad_shapes_truncation_trailing_and_nonfinite() {
        let bytes = fixture().to_bytes();
        assert_eq!(FrozenPolicy::from_bytes(&bytes).unwrap().to_bytes(), bytes);
        for cut in [0, 7, 19, 20, bytes.len() - 1] {
            assert!(FrozenPolicy::from_bytes(&bytes[..cut]).is_err());
        }
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(FrozenPolicy::from_bytes(&bad).is_err());
        for bits in [f32::NAN.to_bits(), f32::INFINITY.to_bits()] {
            let mut bad = bytes.clone();
            bad[20..24].copy_from_slice(&bits.to_le_bytes());
            assert!(FrozenPolicy::from_bytes(&bad).is_err());
        }
        let mut bad = bytes;
        bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(FrozenPolicy::from_bytes(&bad).is_err());
    }
    #[test]
    fn inference_is_stateless_and_memory_is_explicit() {
        let model = fixture();
        let obs = vec![0.1; OBS_DIM];
        let a = model.forward(&obs, &[0.0; 2]).unwrap();
        let b = model.forward(&obs, &[0.9; 2]).unwrap();
        assert_ne!(a.next_memory, b.next_memory);
        assert_eq!(
            a.next_memory,
            model.forward(&obs, &[0.0; 2]).unwrap().next_memory
        );
        assert!(model.forward(&obs, &[]).is_err());
    }

    #[test]
    fn tracing_preserves_every_output_bit_and_selected_context() {
        let model = fixture();
        for context in [
            ObservationExpertContext::Foraging,
            ObservationExpertContext::Interaction,
            ObservationExpertContext::Exploration,
        ] {
            let mut observation = vec![0.0; OBS_DIM];
            if context == ObservationExpertContext::Foraging {
                observation[CURRENT_TILE_PLANT_CAPACITY_FEATURE] = 0.5;
            } else if context == ObservationExpertContext::Interaction {
                observation[HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.;
            }
            let plain = model.forward(&observation, &[0.1, -0.2]).unwrap();
            let (traced, trace) = model
                .forward_with_trace(&observation, &[0.1, -0.2])
                .unwrap();
            assert_eq!(trace.context, context);
            for (a, b) in [
                (&plain.kind, &traced.kind),
                (&plain.target, &traced.target),
                (&plain.effort, &traced.effort),
                (&plain.amount, &traced.amount),
                (&plain.signal, &traced.signal),
                (&plain.signal_strength, &traced.signal_strength),
                (&plain.next_memory, &traced.next_memory),
            ] {
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            for k in 0..NUM_POLICY_ACTION_KINDS {
                assert!(
                    (trace.raw_kind[k]
                        - (trace.base_kind[k] + trace.context_delta[k] + trace.slot_delta[k]))
                        .abs()
                        < 1e-6
                );
            }
        }
    }
}
