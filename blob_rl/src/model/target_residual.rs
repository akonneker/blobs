//! Shared slot scorer over one cell's private randomness and one raw local slot.
use burn::nn;
use burn::prelude::*;

use crate::action::{NUM_POLICY_ACTION_KINDS, NUM_POLICY_TARGETS, NUM_POLICY_TARGET_LOGITS};
use crate::observation::{OBS_RANDOMNESS_FEATURE_END, OBS_RANDOMNESS_FEATURE_START, SLOT_FEATURES};

const RANDOM_FEATURES: usize = OBS_RANDOMNESS_FEATURE_END - OBS_RANDOMNESS_FEATURE_START;
const HIDDEN: usize = 32;

#[derive(Module, Debug)]
pub(super) struct TargetResidual<B: Backend> {
    pub(super) fc: nn::Linear<B>,
    pub(super) head: nn::Linear<B>,
}

impl<B: Backend> TargetResidual<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc: nn::LinearConfig::new(RANDOM_FEATURES + SLOT_FEATURES, HIDDEN).init(device),
            head: nn::LinearConfig::new(HIDDEN, NUM_POLICY_ACTION_KINDS)
                .with_initializer(nn::Initializer::Zeros)
                .init(device),
        }
    }

    #[cfg(test)]
    pub(super) fn activate_for_test(&mut self, device: &B::Device) {
        self.head = nn::LinearConfig::new(HIDDEN, NUM_POLICY_ACTION_KINDS).init(device);
    }

    pub fn forward(&self, random: Tensor<B, 2>, slots: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = random.dims()[0];
        let random = random
            .reshape([batch, 1, RANDOM_FEATURES])
            .repeat_dim(1, NUM_POLICY_TARGETS)
            .reshape([batch * NUM_POLICY_TARGETS, RANDOM_FEATURES]);
        self.head
            .forward(burn::tensor::activation::relu(
                self.fc.forward(Tensor::cat(vec![random, slots], 1)),
            ))
            .reshape([batch, NUM_POLICY_TARGETS, NUM_POLICY_ACTION_KINDS])
            .swap_dims(1, 2)
            .reshape([batch, NUM_POLICY_TARGET_LOGITS])
    }
}

pub(super) fn parameter_count() -> usize {
    (RANDOM_FEATURES + SLOT_FEATURES + 1) * HIDDEN + (HIDDEN + 1) * NUM_POLICY_ACTION_KINDS
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn active_scorer_uses_randomness_for_relative_targets_and_keeps_slots_independent() {
        let _guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        NdArray::<f32>::seed(&device, 719);
        let mut scorer = TargetResidual::<NdArray<f32>>::new(&device);
        scorer.activate_for_test(&device);
        let slots = (0..NUM_POLICY_TARGETS * SLOT_FEATURES)
            .map(|i| (i % 37) as f32 / 37.0)
            .collect::<Vec<_>>();
        let evaluate = |random: f32, slots: Vec<f32>| {
            scorer
                .forward(
                    Tensor::from_data(
                        TensorData::new(vec![random; RANDOM_FEATURES], [1, RANDOM_FEATURES]),
                        &device,
                    ),
                    Tensor::from_data(
                        TensorData::new(slots, [NUM_POLICY_TARGETS, SLOT_FEATURES]),
                        &device,
                    ),
                )
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let before = evaluate(0.0, slots.clone());
        let random_changed = evaluate(1.0, slots.clone());
        assert!(
            (0..NUM_POLICY_ACTION_KINDS).any(|kind| {
                let base = kind * NUM_POLICY_TARGETS;
                (before[base + 1] - before[base] - random_changed[base + 1] + random_changed[base])
                    .abs()
                    > 1e-5
            }),
            "randomness must change relative scores, not only an irrelevant shared offset"
        );
        let mut modified_slots = slots;
        modified_slots[SLOT_FEATURES..2 * SLOT_FEATURES].fill(0.9);
        let slot_changed = evaluate(0.0, modified_slots);
        assert_ne!(before, slot_changed);
        for kind in 0..NUM_POLICY_ACTION_KINDS {
            for slot in 0..NUM_POLICY_TARGETS {
                if slot != 1 {
                    assert_eq!(
                        before[kind * NUM_POLICY_TARGETS + slot].to_bits(),
                        slot_changed[kind * NUM_POLICY_TARGETS + slot].to_bits()
                    );
                }
            }
        }
    }
}
