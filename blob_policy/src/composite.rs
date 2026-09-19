//! Frozen parent plus a separately trained, sampled Move utility.
use crate::{
    action::action_is_commit_legal, observation::*, runtime::*, sampling::CategoricalDistribution,
};
use blob_interface::{
    reference_mind::*,
    reference_mind_converter::{validate_reference_mind_input, ReferenceMindLimits},
};

pub const COMPOSITE_EXECUTION_CONTRACT: &str =
    "blob.policy.move-utility-scalar-f32-libm-q48-signal-reserved.v1";
const MAGIC: &[u8; 8] = b"BLCMP001";
pub const UTILITY_PARAMETERS: usize = 5105;
const SHAPES: [(usize, usize); 4] = [(33, 16), (16, 16), (65, 64), (64, 1)];

#[derive(Clone, Debug)]
pub struct FrozenUtility {
    layers: Vec<Linear>,
    values: Vec<f32>,
}
impl FrozenUtility {
    /// Linear input-major weights followed by biases, encoder/encoder2/fc/head.
    pub fn new(values: Vec<f32>) -> Result<Self, String> {
        if values.len() != UTILITY_PARAMETERS || !values.iter().all(|v| v.is_finite()) {
            return Err("invalid utility parameter count or values".into());
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
    /// Raw slot projection with dx/dy already restored to tile scale.
    pub fn forward(&self, local: &[f32], legal: &[bool; 32]) -> Result<[f32; 32], String> {
        if local.len() != 32 * SLOT_FEATURES || !local.iter().all(|x| x.is_finite()) {
            return Err("invalid utility observation".into());
        }
        let encoded: Vec<Vec<f32>> = local
            .chunks_exact(SLOT_FEATURES)
            .map(|slot| {
                let first: Vec<_> = self.layers[0]
                    .forward(slot)
                    .into_iter()
                    .map(libm::tanhf)
                    .collect();
                self.layers[1]
                    .forward(&first)
                    .into_iter()
                    .map(libm::tanhf)
                    .collect()
            })
            .collect();
        let summary: Vec<f32> = (0..16)
            .map(|feature| {
                let mut values: Vec<_> = (0..32)
                    .map(|slot| encoded[slot][feature] * f32::from(legal[slot]))
                    .collect();
                values.sort_by(f32::total_cmp);
                values.into_iter().sum::<f32>() / 32.
            })
            .collect();
        let scores = std::array::from_fn(|slot| {
            let features: Vec<_> = local[slot * SLOT_FEATURES..(slot + 1) * SLOT_FEATURES]
                .iter()
                .chain(&encoded[slot])
                .chain(&summary)
                .copied()
                .collect();
            let hidden: Vec<_> = self.layers[2]
                .forward(&features)
                .into_iter()
                .map(|x| x.max(0.))
                .collect();
            self.layers[3].forward(&hidden)[0] + (f32::from(legal[slot]) - 1.) * 1e9
        });
        if !scores.iter().all(|x| x.is_finite()) {
            return Err("nonfinite utility inference".into());
        }
        Ok(scores)
    }
}

#[derive(Clone, Debug)]
pub struct CompositePolicy {
    pub parent: FrozenPolicy,
    pub utility: FrozenUtility,
}
impl CompositePolicy {
    pub fn to_bytes(&self) -> Vec<u8> {
        let parent = self.parent.to_bytes();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(parent.len() as u32).to_le_bytes());
        bytes.extend(parent);
        for value in self.utility.values() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || bytes.len() > MAX_WEIGHT_BYTES || &bytes[..8] != MAGIC {
            return Err("invalid composite weight header/size".into());
        }
        let parent_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        if parent_len.checked_add(12 + UTILITY_PARAMETERS * 4) != Some(bytes.len()) {
            return Err("composite weight length mismatch".into());
        }
        Ok(Self {
            parent: FrozenPolicy::from_bytes(&bytes[12..12 + parent_len])?,
            utility: FrozenUtility::new(
                bytes[12 + parent_len..]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            )?,
        })
    }
    pub fn try_decide(&self, input: &ReferenceMindInput) -> Result<ReferenceMindDecision, String> {
        validate_reference_mind_input(input, ReferenceMindLimits::default())
            .map_err(|e| e.to_string())?;
        let decision = self.parent.try_decide(input)?;
        self.apply_move_utility(input, decision)
    }
    pub(crate) fn apply_move_utility(
        &self,
        input: &ReferenceMindInput,
        mut decision: ReferenceMindDecision,
    ) -> Result<ReferenceMindDecision, String> {
        if let ReferenceMindAction::Move {
            target_slot,
            effort,
        } = &mut decision.action
        {
            let mut slots: Vec<_> = input.slots.iter().collect();
            slots.sort_by_key(|s| (s.dy, s.dx));
            if slots.iter().any(|s| s.dx == 0 && s.dy == 0)
                || slots
                    .windows(2)
                    .any(|p| (p[0].dy, p[0].dx) == (p[1].dy, p[1].dx))
            {
                return Err("invalid composite target geometry".into());
            }
            let legal = std::array::from_fn(|slot| {
                action_is_commit_legal(
                    input,
                    &ReferenceMindAction::Move {
                        target_slot: slot as u8,
                        effort: *effort,
                    },
                    decision.signal.is_some(),
                )
            });
            let mut local = Observation::from_reference(input).data[HEADER_FEATURES..].to_vec();
            for slot in local.chunks_exact_mut(SLOT_FEATURES) {
                slot[2] *= 127.;
                slot[3] *= 127.;
            }
            let scores = self.utility.forward(&local, &legal)?;
            let ordered: Vec<_> = slots.iter().map(|s| scores[usize::from(s.slot)]).collect();
            let mask: Vec<_> = slots.iter().map(|s| legal[usize::from(s.slot)]).collect();
            let selected = CategoricalDistribution::from_logits(&ordered, &mask)?
                .sample(input.randomness.sample_u64(0))
                .ok_or("parent Move has no legal signal-preserving target")?;
            *target_slot = slots[selected].slot;
        }
        Ok(decision)
    }
}

/// The envelope explicitly selects greedy or composite execution; old weights
/// never acquire sampled semantics merely by using a newer runtime.
#[derive(Clone, Debug)]
pub enum DeployedPolicy {
    Greedy(FrozenPolicy),
    Composite(CompositePolicy),
    Relational(crate::relational::RelationalPolicy),
}
impl DeployedPolicy {
    pub fn forward(&self, observation: &[f32], memory: &[f32]) -> Result<PolicyOutput, String> {
        match self {
            Self::Greedy(p) => p.forward(observation, memory),
            Self::Composite(p) => p.parent.forward(observation, memory),
            Self::Relational(p) => p.forward(observation, memory),
        }
    }
    pub fn parent(&self) -> &FrozenPolicy {
        match self {
            Self::Greedy(p) => p,
            Self::Composite(p) => &p.parent,
            Self::Relational(p) => &p.base.parent,
        }
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.starts_with(crate::relational::MAGIC) {
            crate::relational::RelationalPolicy::from_bytes(bytes).map(Self::Relational)
        } else if bytes.starts_with(MAGIC) {
            CompositePolicy::from_bytes(bytes).map(Self::Composite)
        } else {
            FrozenPolicy::from_bytes(bytes).map(Self::Greedy)
        }
    }
    pub fn execution_contract(&self) -> &'static str {
        match self {
            Self::Greedy(_) => EXECUTION_CONTRACT,
            Self::Composite(_) => COMPOSITE_EXECUTION_CONTRACT,
            Self::Relational(_) => crate::relational::EXECUTION_CONTRACT,
        }
    }
    pub fn try_decide(&self, input: &ReferenceMindInput) -> Result<ReferenceMindDecision, String> {
        match self {
            Self::Greedy(p) => p.try_decide(input),
            Self::Composite(p) => p.try_decide(input),
            Self::Relational(p) => p.try_decide(input),
        }
    }
}
impl ReferenceMind for DeployedPolicy {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        self.try_decide(input)
            .expect("validated deployed inference")
    }
    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}
