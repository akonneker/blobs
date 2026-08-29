//! Actor-critic neural network model using Burn.

use burn::nn;
use burn::prelude::*;

use crate::action::{
    NUM_POLICY_ACTION_KINDS, NUM_POLICY_AMOUNT_LOGITS, NUM_POLICY_EFFORT_LOGITS,
    NUM_POLICY_TARGETS, NUM_POLICY_TARGET_LOGITS, NUM_SIGNAL_CHOICES, NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::observation::{HEADER_FEATURES, OBS_DIM, SLOT_FEATURES};

const POLICY_MEMORY_MAGIC: [u8; 4] = *b"BRM1";
const POLICY_MEMORY_HEADER_BYTES: usize = 8;
pub const NUM_ACTION_KIND_EXPERTS: usize = 2;

/// Actor-critic model with a cell-private recurrent state.
///
/// The recurrent state is row-separable: no operation aggregates across the
/// batch dimension. Deployment stores the state in canonical Mind private
/// memory, so it follows the same isolation and replay rules as handwritten
/// stateful Minds.
#[derive(Module, Debug)]
pub struct PolicyValueNet<B: Backend> {
    slot_encoder: nn::Linear<B>,
    phase_slot_encoder: nn::Linear<B>,
    phase_gate_fc: nn::Linear<B>,
    shared_fc1: nn::Linear<B>,
    recurrent: nn::Linear<B>,
    shared_fc2: nn::Linear<B>,
    feeding_action_kind_head: nn::Linear<B>,
    combat_action_kind_head: nn::Linear<B>,
    phase_gate_head: nn::Linear<B>,
    target_query_head: nn::Linear<B>,
    effort_head: nn::Linear<B>,
    amount_head: nn::Linear<B>,
    signal_head: nn::Linear<B>,
    signal_strength_head: nn::Linear<B>,
    value_head: nn::Linear<B>,
}

/// Configuration for creating a PolicyValueNet.
#[derive(Config, Debug)]
pub struct PolicyValueNetConfig {
    #[config(default = 128)]
    pub hidden1: usize,
    #[config(default = 64)]
    pub hidden2: usize,
    #[config(default = 64)]
    pub recurrent_size: usize,
}

impl PolicyValueNetConfig {
    /// Exact trainable scalar count for capacity and deployment planning.
    pub fn parameter_count(&self) -> usize {
        let linear = |inputs: usize, outputs: usize| inputs * outputs + outputs;
        linear(SLOT_FEATURES, self.hidden2)
            + linear(SLOT_FEATURES, self.hidden2)
            + linear(HEADER_FEATURES + self.hidden2, self.hidden2)
            + linear(HEADER_FEATURES + self.hidden2, self.hidden1)
            + linear(self.hidden1 + self.recurrent_size, self.recurrent_size)
            + linear(self.recurrent_size, self.hidden2)
            + 2 * linear(self.hidden2, NUM_POLICY_ACTION_KINDS)
            + linear(self.hidden2, NUM_ACTION_KIND_EXPERTS)
            + linear(self.hidden2, NUM_POLICY_ACTION_KINDS * self.hidden2)
            + linear(self.hidden2, NUM_POLICY_EFFORT_LOGITS)
            + linear(self.hidden2, NUM_POLICY_AMOUNT_LOGITS)
            + linear(self.hidden2, NUM_SIGNAL_CHOICES)
            + linear(self.hidden2, NUM_SIGNAL_STRENGTH_CHOICES)
            + linear(self.hidden2, 1)
    }

    /// Initialize a new PolicyValueNet on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PolicyValueNet<B> {
        PolicyValueNet {
            slot_encoder: nn::LinearConfig::new(SLOT_FEATURES, self.hidden2).init(device),
            phase_slot_encoder: nn::LinearConfig::new(SLOT_FEATURES, self.hidden2).init(device),
            phase_gate_fc: nn::LinearConfig::new(HEADER_FEATURES + self.hidden2, self.hidden2)
                .init(device),
            shared_fc1: nn::LinearConfig::new(HEADER_FEATURES + self.hidden2, self.hidden1)
                .init(device),
            recurrent: nn::LinearConfig::new(
                self.hidden1 + self.recurrent_size,
                self.recurrent_size,
            )
            .init(device),
            shared_fc2: nn::LinearConfig::new(self.recurrent_size, self.hidden2).init(device),
            feeding_action_kind_head: nn::LinearConfig::new(self.hidden2, NUM_POLICY_ACTION_KINDS)
                .init(device),
            combat_action_kind_head: nn::LinearConfig::new(self.hidden2, NUM_POLICY_ACTION_KINDS)
                .init(device),
            phase_gate_head: nn::LinearConfig::new(self.hidden2, NUM_ACTION_KIND_EXPERTS)
                .init(device),
            target_query_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS * self.hidden2,
            )
            .init(device),
            effort_head: nn::LinearConfig::new(self.hidden2, NUM_POLICY_EFFORT_LOGITS).init(device),
            amount_head: nn::LinearConfig::new(self.hidden2, NUM_POLICY_AMOUNT_LOGITS).init(device),
            signal_head: nn::LinearConfig::new(self.hidden2, NUM_SIGNAL_CHOICES).init(device),
            signal_strength_head: nn::LinearConfig::new(self.hidden2, NUM_SIGNAL_STRENGTH_CHOICES)
                .init(device),
            value_head: nn::LinearConfig::new(self.hidden2, 1).init(device),
        }
    }
}

/// Output of a forward pass through the model.
pub struct ModelOutput<B: Backend> {
    /// Locally gated inference logits consumed by PPO and deployed Minds.
    pub action_kind_logits: Tensor<B, 2>,
    /// Feeding then combat expert logits, flattened as [expert, action kind].
    /// Supervised training routes each local action-family label to one expert;
    /// inference receives neither that label nor host scenario metadata.
    pub action_kind_expert_logits: Tensor<B, 2>,
    /// Observation-driven feeding/combat gate logits.
    pub phase_gate_logits: Tensor<B, 2>,
    /// Action-kind-conditioned target logits, flattened as [kind, target].
    pub target_logits: Tensor<B, 2>,
    /// Action-kind-conditioned effort logits, flattened as [kind, effort].
    pub effort_logits: Tensor<B, 2>,
    /// Action-kind-conditioned payload/amount logits, flattened as
    /// [kind, amount].
    pub amount_logits: Tensor<B, 2>,
    /// Optional signal-selection logits [batch, NUM_SIGNAL_CHOICES].
    pub signal_logits: Tensor<B, 2>,
    /// Conditional signal-strength logits [batch, NUM_SIGNAL_STRENGTH_CHOICES].
    pub signal_strength_logits: Tensor<B, 2>,
    /// State value estimates [batch, 1]
    pub values: Tensor<B, 2>,
    /// Updated cell-private recurrent state [batch, recurrent_size].
    pub next_memory: Tensor<B, 2>,
}

impl<B: Backend> PolicyValueNet<B> {
    pub fn recurrent_size(&self) -> usize {
        self.recurrent.weight.val().dims()[1]
    }

    /// Stateless convenience pass using zero private memory.
    pub fn forward(&self, obs: Tensor<B, 2>) -> ModelOutput<B> {
        let [batch, _] = obs.dims();
        let recurrent_size = self.recurrent_size();
        let device = obs.device();
        let memory = Tensor::zeros([batch, recurrent_size], &device);
        self.forward_with_memory(obs, memory)
    }

    /// One recurrent decision step. Every output row depends only on the
    /// corresponding observation and private-memory row.
    pub fn forward_with_memory(&self, obs: Tensor<B, 2>, memory: Tensor<B, 2>) -> ModelOutput<B> {
        let [batch, observation_width] = obs.dims();
        debug_assert_eq!(observation_width, OBS_DIM);
        let header = obs.clone().slice([0..batch, 0..HEADER_FEATURES]);
        let raw_slots = obs
            .slice([0..batch, HEADER_FEATURES..OBS_DIM])
            .reshape([batch * NUM_POLICY_TARGETS, SLOT_FEATURES]);
        let phase_slots =
            burn::tensor::activation::relu(self.phase_slot_encoder.forward(raw_slots.clone()))
                .reshape([batch, NUM_POLICY_TARGETS, self.hidden2()]);
        let phase_pooled_slots = phase_slots.max_dim(1).reshape([batch, self.hidden2()]);
        let phase_gate_features = burn::tensor::activation::relu(
            self.phase_gate_fc
                .forward(Tensor::cat(vec![header.clone(), phase_pooled_slots], 1)),
        );
        let slots = burn::tensor::activation::relu(self.slot_encoder.forward(raw_slots)).reshape([
            batch,
            NUM_POLICY_TARGETS,
            self.hidden2(),
        ]);
        let pooled_slots = slots.clone().max_dim(1).reshape([batch, self.hidden2()]);
        let x = self
            .shared_fc1
            .forward(Tensor::cat(vec![header, pooled_slots], 1));
        let x = burn::tensor::activation::relu(x);
        let next_memory =
            burn::tensor::activation::tanh(self.recurrent.forward(Tensor::cat(vec![x, memory], 1)));
        let x = self.shared_fc2.forward(next_memory.clone());
        let x = burn::tensor::activation::relu(x);

        let feeding_action_kind_logits = self.feeding_action_kind_head.forward(x.clone());
        let combat_action_kind_logits = self.combat_action_kind_head.forward(x.clone());
        let phase_gate_logits = self.phase_gate_head.forward(phase_gate_features);
        let phase_gate_probs = burn::tensor::activation::softmax(phase_gate_logits.clone(), 1);
        let feeding_weight = phase_gate_probs
            .clone()
            .slice([0..batch, 0..1])
            .repeat_dim(1, NUM_POLICY_ACTION_KINDS);
        let combat_weight = phase_gate_probs
            .slice([0..batch, 1..2])
            .repeat_dim(1, NUM_POLICY_ACTION_KINDS);
        let action_kind_probs =
            burn::tensor::activation::softmax(feeding_action_kind_logits.clone(), 1)
                * feeding_weight
                + burn::tensor::activation::softmax(combat_action_kind_logits.clone(), 1)
                    * combat_weight;
        let action_kind_logits = action_kind_probs.clamp_min(1.0e-20).log();
        let action_kind_expert_logits = Tensor::cat(
            vec![feeding_action_kind_logits, combat_action_kind_logits],
            1,
        );
        let target_queries = self.target_query_head.forward(x.clone()).reshape([
            batch,
            NUM_POLICY_ACTION_KINDS,
            self.hidden2(),
        ]);
        let target_logits = target_queries
            .matmul(slots.swap_dims(1, 2))
            .div_scalar((self.hidden2() as f64).sqrt())
            .reshape([batch, NUM_POLICY_TARGET_LOGITS]);
        let effort_logits = self.effort_head.forward(x.clone());
        let amount_logits = self.amount_head.forward(x.clone());
        let signal_logits = self.signal_head.forward(x.clone());
        let signal_strength_logits = self.signal_strength_head.forward(x.clone());
        let values = self.value_head.forward(x);

        ModelOutput {
            action_kind_logits,
            action_kind_expert_logits,
            phase_gate_logits,
            target_logits,
            effort_logits,
            amount_logits,
            signal_logits,
            signal_strength_logits,
            values,
            next_memory,
        }
    }

    fn hidden2(&self) -> usize {
        self.shared_fc2.weight.val().dims()[1]
    }

    /// Get action probabilities via softmax.
    pub fn action_kind_probs(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.forward(obs);
        burn::tensor::activation::softmax(output.action_kind_logits, 1)
    }
}

pub fn policy_memory_bytes(recurrent_size: usize) -> Option<usize> {
    recurrent_size
        .checked_mul(std::mem::size_of::<i16>())
        .and_then(|bytes| bytes.checked_add(POLICY_MEMORY_HEADER_BYTES))
}

/// Decode only this policy's exact versioned memory format. Empty or malformed
/// memory deterministically initializes a fresh zero state.
pub fn decode_policy_memory(bytes: &[u8], recurrent_size: usize) -> Vec<f32> {
    let Some(expected) = policy_memory_bytes(recurrent_size) else {
        return vec![0.0; recurrent_size];
    };
    if bytes.len() != expected || bytes[..4] != POLICY_MEMORY_MAGIC {
        return vec![0.0; recurrent_size];
    }
    let declared = u32::from_le_bytes(bytes[4..8].try_into().expect("fixed memory header"));
    if usize::try_from(declared).ok() != Some(recurrent_size) {
        return vec![0.0; recurrent_size];
    }
    bytes[POLICY_MEMORY_HEADER_BYTES..]
        .chunks_exact(2)
        .map(|chunk| {
            f32::from(i16::from_le_bytes(
                chunk.try_into().expect("two-byte memory element"),
            )) / f32::from(i16::MAX)
        })
        .collect()
}

pub fn encode_policy_memory(memory: &[f32]) -> Vec<u8> {
    let declared = u32::try_from(memory.len()).expect("validated recurrent state fits u32");
    let mut bytes = Vec::with_capacity(
        policy_memory_bytes(memory.len()).expect("validated recurrent state byte size"),
    );
    bytes.extend_from_slice(&POLICY_MEMORY_MAGIC);
    bytes.extend_from_slice(&declared.to_le_bytes());
    for value in memory {
        let value = if value.is_finite() { *value } else { 0.0 };
        let quantized = (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray as NdArrayBackend;

    type TestBackend = NdArrayBackend;

    #[test]
    fn test_model_forward_shapes() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig::new().init(&device);

        let batch_size = 8;
        let obs = Tensor::<TestBackend, 2>::zeros([batch_size, OBS_DIM], &device);
        let output = model.forward(obs);

        assert_eq!(
            output.action_kind_logits.dims(),
            [batch_size, NUM_POLICY_ACTION_KINDS]
        );
        assert_eq!(
            output.action_kind_expert_logits.dims(),
            [
                batch_size,
                NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS
            ]
        );
        assert_eq!(
            output.phase_gate_logits.dims(),
            [batch_size, NUM_ACTION_KIND_EXPERTS]
        );
        assert_eq!(
            output.target_logits.dims(),
            [batch_size, NUM_POLICY_TARGET_LOGITS]
        );
        assert_eq!(
            output.effort_logits.dims(),
            [batch_size, NUM_POLICY_EFFORT_LOGITS]
        );
        assert_eq!(
            output.amount_logits.dims(),
            [batch_size, NUM_POLICY_AMOUNT_LOGITS]
        );
        assert_eq!(
            output.signal_logits.dims(),
            [batch_size, NUM_SIGNAL_CHOICES]
        );
        assert_eq!(
            output.signal_strength_logits.dims(),
            [batch_size, NUM_SIGNAL_STRENGTH_CHOICES]
        );
        assert_eq!(output.values.dims(), [batch_size, 1]);
        assert_eq!(output.next_memory.dims(), [batch_size, 64]);
        assert_eq!(PolicyValueNetConfig::new().parameter_count(), 96_444);
    }

    #[test]
    fn large_capacity_profile_has_an_explicit_cost() {
        let large = PolicyValueNetConfig {
            hidden1: 256,
            hidden2: 128,
            recurrent_size: 128,
        };
        assert_eq!(large.parameter_count(), 332_028);
        assert_eq!(policy_memory_bytes(large.recurrent_size), Some(264));
    }

    #[test]
    #[ignore = "manual release-mode small/large policy inference comparison"]
    fn benchmark_policy_capacity_inference() {
        fn run(config: PolicyValueNetConfig, batch: usize, iterations: usize) -> f64 {
            let device = Default::default();
            let model: PolicyValueNet<TestBackend> = config.init(&device);
            let observations = Tensor::<TestBackend, 2>::zeros([batch, OBS_DIM], &device);
            let memory = Tensor::<TestBackend, 2>::zeros([batch, model.recurrent_size()], &device);
            for _ in 0..3 {
                std::hint::black_box(
                    model
                        .forward_with_memory(observations.clone(), memory.clone())
                        .action_kind_logits
                        .into_data(),
                );
            }
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                std::hint::black_box(
                    model
                        .forward_with_memory(observations.clone(), memory.clone())
                        .action_kind_logits
                        .into_data(),
                );
            }
            started.elapsed().as_secs_f64()
        }

        let batch = 512;
        let iterations = 20;
        let small = run(PolicyValueNetConfig::new(), batch, iterations);
        let large = run(
            PolicyValueNetConfig {
                hidden1: 256,
                hidden2: 128,
                recurrent_size: 128,
            },
            batch,
            iterations,
        );
        eprintln!(
            "policy capacity inference: batch={batch} iterations={iterations} small={small:.3}s large={large:.3}s slowdown={:.2}x",
            large / small
        );
    }

    #[test]
    fn test_action_probs_sum_to_one() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig::new().init(&device);

        let obs = Tensor::<TestBackend, 2>::zeros([4, OBS_DIM], &device);
        let probs = model.action_kind_probs(obs);

        // Sum across actions dimension should be ~1.0 for each batch item
        let sums = probs.sum_dim(1);
        let sums_data: Vec<f32> = sums.into_data().to_vec().unwrap();
        for sum in sums_data {
            assert!((sum - 1.0).abs() < 1e-5, "Sum = {}, expected ~1.0", sum);
        }
    }

    #[test]
    fn slot_permutation_preserves_global_outputs_and_permutes_every_target_head() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        TestBackend::seed(&device, 991);
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig {
            hidden1: 16,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init(&device);
        let mut original = vec![0.0; OBS_DIM];
        for (index, value) in original.iter_mut().enumerate() {
            *value = index as f32 / OBS_DIM as f32;
        }
        let mut permuted = original.clone();
        for feature in 0..SLOT_FEATURES {
            permuted.swap(
                HEADER_FEATURES + feature,
                HEADER_FEATURES + 7 * SLOT_FEATURES + feature,
            );
        }
        let output = model.forward(Tensor::from_data(
            TensorData::new([original, permuted].concat(), [2, OBS_DIM]),
            &device,
        ));
        let assert_rows_equal = |values: Vec<f32>, width: usize| {
            for column in 0..width {
                assert!((values[column] - values[width + column]).abs() < 1.0e-5);
            }
        };
        assert_rows_equal(
            output
                .action_kind_logits
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            NUM_POLICY_ACTION_KINDS,
        );
        assert_rows_equal(
            output
                .action_kind_expert_logits
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS,
        );
        assert_rows_equal(
            output
                .phase_gate_logits
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            NUM_ACTION_KIND_EXPERTS,
        );
        assert_rows_equal(
            output.effort_logits.into_data().to_vec::<f32>().unwrap(),
            NUM_POLICY_EFFORT_LOGITS,
        );
        assert_rows_equal(
            output.amount_logits.into_data().to_vec::<f32>().unwrap(),
            NUM_POLICY_AMOUNT_LOGITS,
        );
        assert_rows_equal(
            output.signal_logits.into_data().to_vec::<f32>().unwrap(),
            NUM_SIGNAL_CHOICES,
        );
        assert_rows_equal(
            output
                .signal_strength_logits
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            NUM_SIGNAL_STRENGTH_CHOICES,
        );
        assert_rows_equal(output.values.into_data().to_vec::<f32>().unwrap(), 1);
        assert_rows_equal(
            output.next_memory.into_data().to_vec::<f32>().unwrap(),
            model.recurrent_size(),
        );
        let targets = output.target_logits.into_data().to_vec::<f32>().unwrap();
        for kind in 0..NUM_POLICY_ACTION_KINDS {
            for target in 0..NUM_POLICY_TARGETS {
                let expected_target = match target {
                    0 => 7,
                    7 => 0,
                    target => target,
                };
                let left = targets[kind * NUM_POLICY_TARGETS + target];
                let right =
                    targets[NUM_POLICY_TARGET_LOGITS + kind * NUM_POLICY_TARGETS + expected_target];
                assert!((left - right).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    fn policy_memory_is_versioned_bounded_and_fail_closed() {
        let state = vec![0.25, -0.5, 1.0];
        let encoded = encode_policy_memory(&state);
        assert_eq!(encoded.len(), 8 + state.len() * 2);
        let decoded = decode_policy_memory(&encoded, state.len());
        for (expected, actual) in state.iter().zip(decoded) {
            assert!((expected - actual).abs() <= 1.0 / f32::from(i16::MAX));
        }
        assert_eq!(
            decode_policy_memory(&encoded, state.len() + 1),
            vec![0.0; 4]
        );
        let mut malformed = encoded;
        malformed[0] ^= 1;
        assert_eq!(decode_policy_memory(&malformed, state.len()), vec![0.0; 3]);
    }

    #[test]
    fn recurrent_rows_are_isolated_and_memory_changes_the_next_state() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        TestBackend::seed(&device, 123);
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 4,
        }
        .init(&device);
        let observation = vec![0.125; OBS_DIM];
        let output = model.forward_with_memory(
            Tensor::from_data(
                TensorData::new(
                    observation
                        .iter()
                        .chain(&observation)
                        .copied()
                        .collect::<Vec<_>>(),
                    [2, OBS_DIM],
                ),
                &device,
            ),
            Tensor::from_data(
                TensorData::new(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [2, 4]),
                &device,
            ),
        );
        let memory = output.next_memory.into_data().to_vec::<f32>().unwrap();
        assert_ne!(&memory[..4], &memory[4..]);

        let scalar = model
            .forward_with_memory(
                Tensor::from_data(TensorData::new(observation, [1, OBS_DIM]), &device),
                Tensor::zeros([1, 4], &device),
            )
            .next_memory
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (batched, scalar) in memory[..4].iter().zip(scalar) {
            assert!((batched - scalar).abs() < 1.0e-6);
        }
    }
}
