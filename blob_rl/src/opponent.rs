//! Stateless policy snapshots exposed through the canonical native Mind ABI.

use blob_interface::reference_mind::{ReferenceMind, ReferenceMindDecision, ReferenceMindInput};
use burn::prelude::*;

use crate::action::{
    attach_policy_memory, decode_policy_choice, decompose_policy_action, PolicyChoice,
    NUM_AMOUNT_CHOICES, NUM_POLICY_ACTION_KINDS, NUM_POLICY_AMOUNT_LOGITS,
    NUM_POLICY_EFFORT_LOGITS, NUM_POLICY_TARGET_LOGITS, NUM_SIGNAL_CHOICES,
    NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::evaluation::{
    greedy_amount, greedy_policy_action, greedy_signal, greedy_signal_strength,
};
use crate::model::{decode_policy_memory, encode_policy_memory, PolicyValueNet};
use crate::observation::{Observation, OBS_DIM};

/// Trusted host-side batching boundary. Implementations receive independently
/// owned anonymous inputs, and must apply a row-separable policy without
/// aggregation or cross-row state.
pub trait SnapshotBatchPolicy: Send {
    fn decide_batch(
        &mut self,
        inputs: &[ReferenceMindInput],
    ) -> Result<Vec<ReferenceMindDecision>, String>;

    fn forward_calls(&self) -> u64;

    fn inferred_rows(&self) -> u64;
}

/// Batched form of the same stateless feed-forward snapshot policy used by
/// `SnapshotPolicyMind`. The model contains only row-separable linear and ReLU
/// operations; there is no batch normalization, attention, or shared memory.
pub struct BatchedSnapshotPolicy<B: Backend> {
    model: PolicyValueNet<B>,
    device: B::Device,
    forward_calls: u64,
    inferred_rows: u64,
}

impl<B: Backend> BatchedSnapshotPolicy<B> {
    pub fn new(model: PolicyValueNet<B>, device: B::Device) -> Self {
        Self {
            model,
            device,
            forward_calls: 0,
            inferred_rows: 0,
        }
    }
}

impl<B: Backend> SnapshotBatchPolicy for BatchedSnapshotPolicy<B>
where
    f32: From<B::FloatElem>,
{
    fn decide_batch(
        &mut self,
        inputs: &[ReferenceMindInput],
    ) -> Result<Vec<ReferenceMindDecision>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let observations = inputs
            .iter()
            .map(Observation::from_reference)
            .collect::<Vec<_>>();
        let data = observations
            .iter()
            .flat_map(|observation| observation.data)
            .collect::<Vec<_>>();
        let recurrent_size = self.model.recurrent_size();
        let memory = inputs
            .iter()
            .flat_map(|input| decode_policy_memory(&input.private_memory, recurrent_size))
            .collect::<Vec<_>>();
        let output = self.model.forward_with_memory(
            Tensor::<B, 2>::from_data(TensorData::new(data, [inputs.len(), OBS_DIM]), &self.device),
            Tensor::<B, 2>::from_data(
                TensorData::new(memory, [inputs.len(), recurrent_size]),
                &self.device,
            ),
        );
        let width = NUM_POLICY_ACTION_KINDS
            + NUM_POLICY_TARGET_LOGITS
            + NUM_POLICY_EFFORT_LOGITS
            + NUM_POLICY_AMOUNT_LOGITS
            + NUM_SIGNAL_CHOICES
            + NUM_SIGNAL_STRENGTH_CHOICES
            + recurrent_size;
        let output = Tensor::cat(
            vec![
                output.action_kind_logits,
                output.target_logits,
                output.effort_logits,
                output.amount_logits,
                output.signal_logits,
                output.signal_strength_logits,
                output.next_memory,
            ],
            1,
        )
        .into_data()
        .to_vec()
        .map_err(|error| format!("snapshot opponent logits use an unsupported type: {error}"))?
        .into_iter()
        .map(f32::from)
        .collect::<Vec<_>>();
        self.forward_calls = self.forward_calls.saturating_add(1);
        self.inferred_rows = self
            .inferred_rows
            .saturating_add(u64::try_from(inputs.len()).unwrap_or(u64::MAX));
        Ok(inputs
            .iter()
            .zip(&observations)
            .enumerate()
            .map(|(index, (input, observation))| {
                let start = index * width;
                let target_start = start + NUM_POLICY_ACTION_KINDS;
                let effort_start = target_start + NUM_POLICY_TARGET_LOGITS;
                let amount_start = effort_start + NUM_POLICY_EFFORT_LOGITS;
                let action = greedy_policy_action(
                    &output[start..target_start],
                    &output[target_start..effort_start],
                    &output[effort_start..amount_start],
                    observation,
                );
                let kind = decompose_policy_action(action)
                    .expect("greedy action is in the policy catalog")
                    .kind;
                let amount = greedy_amount(
                    &output[amount_start + kind * NUM_AMOUNT_CHOICES
                        ..amount_start + (kind + 1) * NUM_AMOUNT_CHOICES],
                    observation,
                    action,
                );
                let signal_start = amount_start + NUM_POLICY_AMOUNT_LOGITS;
                let signal = greedy_signal(
                    &output[signal_start..signal_start + NUM_SIGNAL_CHOICES],
                    observation,
                    action,
                    amount,
                );
                let signal_strength_start = signal_start + NUM_SIGNAL_CHOICES;
                let signal_strength = greedy_signal_strength(
                    &output[signal_strength_start
                        ..signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES],
                    observation,
                    action,
                    amount,
                    signal,
                );
                let mut decision = decode_policy_choice(
                    PolicyChoice {
                        action,
                        amount,
                        signal,
                        signal_strength,
                    },
                    input,
                );
                attach_policy_memory(
                    &mut decision,
                    encode_policy_memory(
                        &output[signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES..start + width],
                    ),
                );
                decision
            })
            .collect())
    }

    fn forward_calls(&self) -> u64 {
        self.forward_calls
    }

    fn inferred_rows(&self) -> u64 {
        self.inferred_rows
    }
}

/// Immutable Burn policy adapter. It receives one anonymous observation and
/// returns the next recurrent vector only through canonical private memory.
/// The adapter itself retains no cross-cell invocation state.
pub struct SnapshotPolicyMind<B: Backend> {
    model: PolicyValueNet<B>,
    device: B::Device,
}

impl<B: Backend> SnapshotPolicyMind<B> {
    pub fn new(model: PolicyValueNet<B>, device: B::Device) -> Self {
        Self { model, device }
    }
}

impl<B: Backend> ReferenceMind for SnapshotPolicyMind<B>
where
    f32: From<B::FloatElem>,
{
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let observation = Observation::from_reference(input);
        let tensor = Tensor::<B, 2>::from_data(
            TensorData::new(observation.data.to_vec(), [1, OBS_DIM]),
            &self.device,
        );
        let recurrent_size = self.model.recurrent_size();
        let memory = decode_policy_memory(&input.private_memory, recurrent_size);
        let output = self.model.forward_with_memory(
            tensor,
            Tensor::<B, 2>::from_data(TensorData::new(memory, [1, recurrent_size]), &self.device),
        );
        let width = NUM_POLICY_ACTION_KINDS
            + NUM_POLICY_TARGET_LOGITS
            + NUM_POLICY_EFFORT_LOGITS
            + NUM_POLICY_AMOUNT_LOGITS
            + NUM_SIGNAL_CHOICES
            + NUM_SIGNAL_STRENGTH_CHOICES
            + recurrent_size;
        let output = Tensor::cat(
            vec![
                output.action_kind_logits,
                output.target_logits,
                output.effort_logits,
                output.amount_logits,
                output.signal_logits,
                output.signal_strength_logits,
                output.next_memory,
            ],
            1,
        )
        .into_data()
        .to_vec()
        .expect("snapshot opponent logits should use a supported float type")
        .into_iter()
        .map(f32::from)
        .collect::<Vec<_>>();
        let target_start = NUM_POLICY_ACTION_KINDS;
        let effort_start = target_start + NUM_POLICY_TARGET_LOGITS;
        let amount_start = effort_start + NUM_POLICY_EFFORT_LOGITS;
        let action = greedy_policy_action(
            &output[..target_start],
            &output[target_start..effort_start],
            &output[effort_start..amount_start],
            &observation,
        );
        let kind = decompose_policy_action(action)
            .expect("greedy action is in the policy catalog")
            .kind;
        let amount = greedy_amount(
            &output[amount_start + kind * NUM_AMOUNT_CHOICES
                ..amount_start + (kind + 1) * NUM_AMOUNT_CHOICES],
            &observation,
            action,
        );
        let signal_start = amount_start + NUM_POLICY_AMOUNT_LOGITS;
        let signal = greedy_signal(
            &output[signal_start..signal_start + NUM_SIGNAL_CHOICES],
            &observation,
            action,
            amount,
        );
        let signal_strength_start = signal_start + NUM_SIGNAL_CHOICES;
        let signal_strength = greedy_signal_strength(
            &output[signal_strength_start..signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES],
            &observation,
            action,
            amount,
            signal,
        );
        let mut decision = decode_policy_choice(
            PolicyChoice {
                action,
                amount,
                signal,
                signal_strength,
            },
            input,
        );
        attach_policy_memory(
            &mut decision,
            encode_policy_memory(
                &output[signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES..width],
            ),
        );
        decision
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}
