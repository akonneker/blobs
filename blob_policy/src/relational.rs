//! Explicit frozen per-slot interaction residual envelope. No host/global state.
use crate::{
    composite::CompositePolicy,
    memory::decode_policy_memory,
    observation::*,
    runtime::{FrozenPolicy, Linear, PolicyOutput, MAX_WEIGHT_BYTES},
};
use blob_interface::{
    reference_mind::*,
    reference_mind_converter::{validate_reference_mind_input, ReferenceMindLimits},
};
pub const RELATIONAL_WEIGHT_FORMAT_VERSION: u32 = 1;
pub const EXECUTION_CONTRACT: &str = "blob.policy.interaction-slot-encoder-f32-libm-move-q48.v1";
pub(crate) const MAGIC: &[u8; 8] = b"BLCMR001";
pub const PARAMETERS: usize = 3418;
pub const SHAPES: [(usize, usize); 4] = [(33, 16), (16, 16), (70, 32), (32, 10)];
#[derive(Clone, Debug)]
pub struct FrozenResidual {
    layers: Vec<Linear>,
    values: Vec<f32>,
}
impl FrozenResidual {
    pub fn new(values: Vec<f32>) -> Result<Self, String> {
        if values.len() != PARAMETERS || !values.iter().all(|x| x.is_finite()) {
            return Err("invalid relational residual parameters".into());
        }
        let mut cursor = values.iter().copied();
        let layers = SHAPES
            .into_iter()
            .map(|(i, o)| {
                Linear::new(
                    i,
                    o,
                    cursor.by_ref().take(i * o).collect(),
                    cursor.by_ref().take(o).collect(),
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { layers, values })
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    /// 33 raw slot features (dx/dy in tile units), two tanh layers, masked
    /// sorted sum/32 plus maximum, then nonrandom header + summaries -> ReLU ->10.
    pub fn forward(&self, observation: &[f32]) -> Result<Vec<f32>, String> {
        if observation.len() != OBS_DIM || !observation.iter().all(|x| x.is_finite()) {
            return Err("invalid relational observation".into());
        }
        let encoded: Vec<Vec<f32>> = observation[HEADER_FEATURES..]
            .chunks_exact(SLOT_FEATURES)
            .map(|slot| {
                let mut local = slot.to_vec();
                local[2] *= 127.;
                local[3] *= 127.;
                let first: Vec<_> = self.layers[0]
                    .forward(&local)
                    .into_iter()
                    .map(libm::tanhf)
                    .collect();
                self.layers[1]
                    .forward(&first)
                    .into_iter()
                    .map(|x| libm::tanhf(x) * slot[0])
                    .collect()
            })
            .collect();
        let mut features = observation[..OBS_RANDOMNESS_FEATURE_START].to_vec();
        for i in 0..16 {
            let mut values: Vec<_> = encoded.iter().map(|s| s[i]).collect();
            values.sort_by(f32::total_cmp);
            features.push(values.into_iter().sum::<f32>() / 32.);
        }
        features.extend((0..16).map(|i| {
            encoded
                .iter()
                .map(|s| s[i])
                .fold(f32::NEG_INFINITY, f32::max)
        }));
        let hidden: Vec<_> = self.layers[2]
            .forward(&features)
            .into_iter()
            .map(|x| x.max(0.))
            .collect();
        let result = self.layers[3].forward(&hidden);
        if !result.iter().all(|x| x.is_finite()) {
            return Err("nonfinite relational inference".into());
        }
        Ok(result)
    }
}
#[derive(Clone, Debug)]
pub struct RelationalPolicy {
    pub base: CompositePolicy,
    pub residual: FrozenResidual,
}
impl RelationalPolicy {
    pub fn to_bytes(&self) -> Vec<u8> {
        let base = self.base.to_bytes();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(base.len() as u32).to_le_bytes());
        bytes.extend(base);
        for v in self.residual.values() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || bytes.len() > MAX_WEIGHT_BYTES || &bytes[..8] != MAGIC {
            return Err("invalid relational envelope".into());
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        if n.checked_add(12 + PARAMETERS * 4) != Some(bytes.len()) {
            return Err("relational envelope length mismatch".into());
        }
        Ok(Self {
            base: CompositePolicy::from_bytes(&bytes[12..12 + n])?,
            residual: FrozenResidual::new(
                bytes[12 + n..]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            )?,
        })
    }
    pub fn forward(&self, observation: &[f32], memory: &[f32]) -> Result<PolicyOutput, String> {
        if observation.len() != OBS_DIM {
            return Err("invalid relational observation size".into());
        }
        if observation_expert_context(observation) == ObservationExpertContext::Interaction {
            self.base.parent.forward_with_kind_delta(
                observation,
                memory,
                &self.residual.forward(observation)?,
            )
        } else {
            self.base.parent.forward(observation, memory)
        }
    }
    pub fn try_decide(&self, input: &ReferenceMindInput) -> Result<ReferenceMindDecision, String> {
        validate_reference_mind_input(input, ReferenceMindLimits::default())
            .map_err(|e| e.to_string())?;
        let observation = Observation::from_reference(input);
        let out = self.forward(
            &observation.data,
            &decode_policy_memory(&input.private_memory, self.base.parent.recurrent_size()),
        )?;
        let decision = FrozenPolicy::decode_output(input, &observation, &out);
        self.base.apply_move_utility(input, decision)
    }
}
