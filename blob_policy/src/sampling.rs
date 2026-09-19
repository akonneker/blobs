//! Portable categorical target selection, independent of the greedy weight format.
//!
//! f32 logits widen to f64 before max subtraction; libm exp and round produce
//! integer weights at scale 2^48. Cumulative sampling uses exact u128 arithmetic.
use crate::action::{action_mask, policy_target_mask, PolicyActionKind, NUM_POLICY_TARGETS};
use blob_interface::{
    reference_mind::ReferenceMindInput,
    reference_mind_converter::{validate_reference_mind_input, ReferenceMindLimits},
};

pub const TARGET_SAMPLING_CONTRACT: &str =
    "blob.policy.target-categorical-f64-libm-q48-dy-dx-word0.v1";
const SCALE: u64 = 1 << 48;

#[derive(Debug, Clone)]
pub struct CategoricalDistribution {
    weights: [u64; NUM_POLICY_TARGETS],
    len: usize,
    total: u64,
}

impl CategoricalDistribution {
    /// Empty support is valid and samples to None. Nonfinite masked logits are
    /// ignored; nonfinite legal logits and malformed shapes are rejected.
    pub fn from_logits(logits: &[f32], legal: &[bool]) -> Result<Self, &'static str> {
        if logits.len() != legal.len() || logits.len() > NUM_POLICY_TARGETS {
            return Err("categorical logits/mask shape mismatch or capacity exceeded");
        }
        let mut maximum = f32::NEG_INFINITY;
        for (&logit, &allowed) in logits.iter().zip(legal) {
            if allowed {
                if !logit.is_finite() {
                    return Err("nonfinite legal categorical logit");
                }
                maximum = maximum.max(logit);
            }
        }
        let mut weights = [0; NUM_POLICY_TARGETS];
        if maximum.is_finite() {
            for (index, (&logit, &allowed)) in logits.iter().zip(legal).enumerate() {
                if allowed {
                    weights[index] = libm::round(
                        libm::exp(f64::from(logit) - f64::from(maximum)) * SCALE as f64,
                    ) as u64;
                }
            }
        }
        let total = weights.iter().sum();
        Ok(Self {
            weights,
            len: logits.len(),
            total,
        })
    }

    pub fn weights(&self) -> &[u64] {
        &self.weights[..self.len]
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    /// At most 32 * 2^48 weight units. The product fits u128 for every u64 word.
    pub fn sample(&self, word: u64) -> Option<usize> {
        let point = ((u128::from(word) * u128::from(self.total)) >> 64) as u64;
        let mut cumulative = 0;
        self.weights().iter().position(|weight| {
            cumulative += weight;
            point < cumulative
        })
    }
}

/// Distribution indexed by geometry order, mapped back to ordinary ABI slot IDs.
#[derive(Debug, Clone)]
pub struct TargetDistribution {
    slots: Vec<u8>,
    distribution: CategoricalDistribution,
}

impl TargetDistribution {
    /// Logits are indexed by ABI slot ID. Affordability/reachability masks come
    /// from the same policy catalog as greedy inference; food utility is learned
    /// by the caller, never inferred by this decoder. Invalid input is an error.
    pub fn from_input(
        input: &ReferenceMindInput,
        kind: PolicyActionKind,
        logits: &[f32; NUM_POLICY_TARGETS],
    ) -> Result<Self, String> {
        validate_reference_mind_input(input, ReferenceMindLimits::default())
            .map_err(|error| error.to_string())?;
        if !kind.uses_target() {
            return Err("categorical target decoding requires a targeted action kind".into());
        }
        let mut slots: Vec<_> = input.slots.iter().collect();
        slots.sort_by_key(|slot| (slot.dy, slot.dx));
        if slots.iter().any(|slot| slot.dx == 0 && slot.dy == 0)
            || slots
                .windows(2)
                .any(|pair| (pair[0].dy, pair[0].dx) == (pair[1].dy, pair[1].dx))
        {
            return Err("target geometry must be unique and exclude the current tile".into());
        }
        let mask = policy_target_mask(&action_mask(input), kind.index());
        let mut ordered_logits = Vec::with_capacity(slots.len());
        let mut ordered_mask = Vec::with_capacity(slots.len());
        for slot in &slots {
            ordered_logits.push(logits[usize::from(slot.slot)]);
            ordered_mask.push(mask[usize::from(slot.slot)]);
        }
        let distribution = CategoricalDistribution::from_logits(&ordered_logits, &ordered_mask)?;
        Ok(Self {
            slots: slots.iter().map(|slot| slot.slot).collect(),
            distribution,
        })
    }

    pub fn canonical_slots(&self) -> &[u8] {
        &self.slots
    }
    pub fn categorical(&self) -> &CategoricalDistribution {
        &self.distribution
    }
    pub fn sample(&self, word: u64) -> Option<u8> {
        self.distribution
            .sample(word)
            .map(|index| self.slots[index])
    }
}

/// Target draw consumes only the first little-endian private random word.
pub fn sample_target(
    input: &ReferenceMindInput,
    kind: PolicyActionKind,
    logits: &[f32; NUM_POLICY_TARGETS],
) -> Result<Option<u8>, String> {
    Ok(TargetDistribution::from_input(input, kind, logits)?.sample(input.randomness.sample_u64(0)))
}
