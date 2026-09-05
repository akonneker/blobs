//! Actor-critic neural network model using Burn.

use burn::nn;
use burn::prelude::*;

use crate::action::{
    PolicyActionKind, NUM_POLICY_ACTION_KINDS, NUM_POLICY_AMOUNT_LOGITS, NUM_POLICY_EFFORT_LOGITS,
    NUM_POLICY_TARGETS, NUM_POLICY_TARGET_LOGITS, NUM_SIGNAL_CHOICES, NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::observation::{
    ObservationExpertContext, CURRENT_TILE_LOOSE_ENERGY_FEATURE,
    CURRENT_TILE_PLANT_CAPACITY_FEATURE, HEADER_FEATURES, OBS_DIM, OBS_RANDOMNESS_FEATURE_START,
    SLOT_FEATURES, SLOT_NEIGHBOR_ACTIVITY_FEATURE, SLOT_NEIGHBOR_PRESENT_FEATURE,
};

const POLICY_MEMORY_MAGIC: [u8; 4] = *b"BRM1";
const POLICY_MEMORY_HEADER_BYTES: usize = 8;
pub const NUM_ACTION_KIND_EXPERTS: usize = ObservationExpertContext::COUNT;
const FORAGING_ADAPTER_FEATURES: usize = 8;
const FORAGING_ADAPTER_HIDDEN: usize = 16;
const CONTEXT_ADAPTER_HIDDEN: usize = 16;

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
    foraging_action_kind_head: nn::Linear<B>,
    foraging_adapter_fc: nn::Linear<B>,
    foraging_adapter_head: nn::Linear<B>,
    interaction_action_kind_head: nn::Linear<B>,
    interaction_adapter_fc: nn::Linear<B>,
    interaction_adapter_head: nn::Linear<B>,
    interaction_slot_adapter_fc: nn::Linear<B>,
    interaction_slot_adapter_head: nn::Linear<B>,
    exploration_action_kind_head: nn::Linear<B>,
    exploration_adapter_fc: nn::Linear<B>,
    exploration_adapter_head: nn::Linear<B>,
    exploration_slot_adapter_fc: nn::Linear<B>,
    exploration_slot_adapter_head: nn::Linear<B>,
    exploration_guard_readiness_head: nn::Linear<B>,
    phase_gate_head: nn::Linear<B>,
    target_query_head: nn::Linear<B>,
    effort_head: nn::Linear<B>,
    amount_head: nn::Linear<B>,
    signal_head: nn::Linear<B>,
    signal_strength_head: nn::Linear<B>,
    value_head: nn::Linear<B>,
}

/// Recorder-compatible policy layout used by behavior-cloning artifacts
/// published before the local foraging adapter was introduced. It is kept
/// only for a one-way, behavior-preserving warm-start migration.
#[derive(Module, Debug)]
pub struct LegacyPolicyValueNet<B: Backend> {
    slot_encoder: nn::Linear<B>,
    phase_slot_encoder: nn::Linear<B>,
    phase_gate_fc: nn::Linear<B>,
    shared_fc1: nn::Linear<B>,
    recurrent: nn::Linear<B>,
    shared_fc2: nn::Linear<B>,
    foraging_action_kind_head: nn::Linear<B>,
    interaction_action_kind_head: nn::Linear<B>,
    exploration_action_kind_head: nn::Linear<B>,
    phase_gate_head: nn::Linear<B>,
    target_query_head: nn::Linear<B>,
    effort_head: nn::Linear<B>,
    amount_head: nn::Linear<B>,
    signal_head: nn::Linear<B>,
    signal_strength_head: nn::Linear<B>,
    value_head: nn::Linear<B>,
}

/// Recorder-compatible layout used after the foraging adapter and before the
/// interaction/exploration context adapters.
#[derive(Module, Debug)]
pub struct ForagingAdapterPolicyValueNet<B: Backend> {
    slot_encoder: nn::Linear<B>,
    phase_slot_encoder: nn::Linear<B>,
    phase_gate_fc: nn::Linear<B>,
    shared_fc1: nn::Linear<B>,
    recurrent: nn::Linear<B>,
    shared_fc2: nn::Linear<B>,
    foraging_action_kind_head: nn::Linear<B>,
    foraging_adapter_fc: nn::Linear<B>,
    foraging_adapter_head: nn::Linear<B>,
    interaction_action_kind_head: nn::Linear<B>,
    exploration_action_kind_head: nn::Linear<B>,
    phase_gate_head: nn::Linear<B>,
    target_query_head: nn::Linear<B>,
    effort_head: nn::Linear<B>,
    amount_head: nn::Linear<B>,
    signal_head: nn::Linear<B>,
    signal_strength_head: nn::Linear<B>,
    value_head: nn::Linear<B>,
}

/// Recorder-compatible layout used after header context adapters and before
/// raw local-slot context residuals. Migration initializes the new residual
/// heads to zero, preserving every policy output exactly.
#[derive(Module, Debug)]
pub struct HeaderContextAdapterPolicyValueNet<B: Backend> {
    slot_encoder: nn::Linear<B>,
    phase_slot_encoder: nn::Linear<B>,
    phase_gate_fc: nn::Linear<B>,
    shared_fc1: nn::Linear<B>,
    recurrent: nn::Linear<B>,
    shared_fc2: nn::Linear<B>,
    foraging_action_kind_head: nn::Linear<B>,
    foraging_adapter_fc: nn::Linear<B>,
    foraging_adapter_head: nn::Linear<B>,
    interaction_action_kind_head: nn::Linear<B>,
    interaction_adapter_fc: nn::Linear<B>,
    interaction_adapter_head: nn::Linear<B>,
    exploration_action_kind_head: nn::Linear<B>,
    exploration_adapter_fc: nn::Linear<B>,
    exploration_adapter_head: nn::Linear<B>,
    phase_gate_head: nn::Linear<B>,
    target_query_head: nn::Linear<B>,
    effort_head: nn::Linear<B>,
    amount_head: nn::Linear<B>,
    signal_head: nn::Linear<B>,
    signal_strength_head: nn::Linear<B>,
    value_head: nn::Linear<B>,
}

/// Recorder-compatible layout used after local slot adapters and before the
/// scalar exploration Guard readiness residual. Migration initializes the new
/// residual to zero and therefore preserves every output exactly.
#[derive(Module, Debug)]
pub struct SlotContextAdapterPolicyValueNet<B: Backend> {
    slot_encoder: nn::Linear<B>,
    phase_slot_encoder: nn::Linear<B>,
    phase_gate_fc: nn::Linear<B>,
    shared_fc1: nn::Linear<B>,
    recurrent: nn::Linear<B>,
    shared_fc2: nn::Linear<B>,
    foraging_action_kind_head: nn::Linear<B>,
    foraging_adapter_fc: nn::Linear<B>,
    foraging_adapter_head: nn::Linear<B>,
    interaction_action_kind_head: nn::Linear<B>,
    interaction_adapter_fc: nn::Linear<B>,
    interaction_adapter_head: nn::Linear<B>,
    interaction_slot_adapter_fc: nn::Linear<B>,
    interaction_slot_adapter_head: nn::Linear<B>,
    exploration_action_kind_head: nn::Linear<B>,
    exploration_adapter_fc: nn::Linear<B>,
    exploration_adapter_head: nn::Linear<B>,
    exploration_slot_adapter_fc: nn::Linear<B>,
    exploration_slot_adapter_head: nn::Linear<B>,
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
            + NUM_ACTION_KIND_EXPERTS * linear(self.hidden2, NUM_POLICY_ACTION_KINDS)
            + linear(
                self.hidden2 + FORAGING_ADAPTER_FEATURES,
                FORAGING_ADAPTER_HIDDEN,
            )
            + linear(FORAGING_ADAPTER_HIDDEN, NUM_POLICY_ACTION_KINDS)
            + 2 * linear(
                self.hidden2 + OBS_RANDOMNESS_FEATURE_START,
                CONTEXT_ADAPTER_HIDDEN,
            )
            + 2 * linear(CONTEXT_ADAPTER_HIDDEN, NUM_POLICY_ACTION_KINDS)
            + linear(2, 1)
            + 2 * linear(
                OBS_RANDOMNESS_FEATURE_START + SLOT_FEATURES,
                CONTEXT_ADAPTER_HIDDEN,
            )
            + 2 * linear(CONTEXT_ADAPTER_HIDDEN, NUM_POLICY_ACTION_KINDS)
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
            foraging_action_kind_head: nn::LinearConfig::new(self.hidden2, NUM_POLICY_ACTION_KINDS)
                .init(device),
            foraging_adapter_fc: nn::LinearConfig::new(
                self.hidden2 + FORAGING_ADAPTER_FEATURES,
                FORAGING_ADAPTER_HIDDEN,
            )
            .init(device),
            // A zero residual makes model migration behavior-preserving while
            // still allowing the output head, then its input projection, to
            // learn from the first supervised updates.
            foraging_adapter_head: nn::LinearConfig::new(
                FORAGING_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
            .with_initializer(nn::Initializer::Zeros)
            .init(device),
            interaction_action_kind_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            interaction_adapter_fc: nn::LinearConfig::new(
                self.hidden2 + OBS_RANDOMNESS_FEATURE_START,
                CONTEXT_ADAPTER_HIDDEN,
            )
            .init(device),
            interaction_adapter_head: nn::LinearConfig::new(
                CONTEXT_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
            .with_initializer(nn::Initializer::Zeros)
            .init(device),
            exploration_action_kind_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            exploration_adapter_fc: nn::LinearConfig::new(
                self.hidden2 + OBS_RANDOMNESS_FEATURE_START,
                CONTEXT_ADAPTER_HIDDEN,
            )
            .init(device),
            exploration_adapter_head: nn::LinearConfig::new(
                CONTEXT_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
            .with_initializer(nn::Initializer::Zeros)
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
            interaction_slot_adapter_fc: nn::LinearConfig::new(
                OBS_RANDOMNESS_FEATURE_START + SLOT_FEATURES,
                CONTEXT_ADAPTER_HIDDEN,
            )
            .init(device),
            interaction_slot_adapter_head: nn::LinearConfig::new(
                CONTEXT_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
            .with_initializer(nn::Initializer::Zeros)
            .init(device),
            exploration_slot_adapter_fc: nn::LinearConfig::new(
                OBS_RANDOMNESS_FEATURE_START + SLOT_FEATURES,
                CONTEXT_ADAPTER_HIDDEN,
            )
            .init(device),
            exploration_slot_adapter_head: nn::LinearConfig::new(
                CONTEXT_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
            .with_initializer(nn::Initializer::Zeros)
            .init(device),
            exploration_guard_readiness_head: nn::LinearConfig::new(2, 1)
                .with_initializer(nn::Initializer::Zeros)
                .init(device),
        }
    }

    pub fn init_slot_context_adapter_legacy<B: Backend>(
        &self,
        device: &B::Device,
    ) -> SlotContextAdapterPolicyValueNet<B> {
        let current = self.init::<B>(device);
        SlotContextAdapterPolicyValueNet {
            slot_encoder: current.slot_encoder,
            phase_slot_encoder: current.phase_slot_encoder,
            phase_gate_fc: current.phase_gate_fc,
            shared_fc1: current.shared_fc1,
            recurrent: current.recurrent,
            shared_fc2: current.shared_fc2,
            foraging_action_kind_head: current.foraging_action_kind_head,
            foraging_adapter_fc: current.foraging_adapter_fc,
            foraging_adapter_head: current.foraging_adapter_head,
            interaction_action_kind_head: current.interaction_action_kind_head,
            interaction_adapter_fc: current.interaction_adapter_fc,
            interaction_adapter_head: current.interaction_adapter_head,
            interaction_slot_adapter_fc: current.interaction_slot_adapter_fc,
            interaction_slot_adapter_head: current.interaction_slot_adapter_head,
            exploration_action_kind_head: current.exploration_action_kind_head,
            exploration_adapter_fc: current.exploration_adapter_fc,
            exploration_adapter_head: current.exploration_adapter_head,
            exploration_slot_adapter_fc: current.exploration_slot_adapter_fc,
            exploration_slot_adapter_head: current.exploration_slot_adapter_head,
            phase_gate_head: current.phase_gate_head,
            target_query_head: current.target_query_head,
            effort_head: current.effort_head,
            amount_head: current.amount_head,
            signal_head: current.signal_head,
            signal_strength_head: current.signal_strength_head,
            value_head: current.value_head,
        }
    }

    pub fn init_legacy<B: Backend>(&self, device: &B::Device) -> LegacyPolicyValueNet<B> {
        LegacyPolicyValueNet {
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
            foraging_action_kind_head: nn::LinearConfig::new(self.hidden2, NUM_POLICY_ACTION_KINDS)
                .init(device),
            interaction_action_kind_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            exploration_action_kind_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS,
            )
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

    pub fn init_foraging_adapter_legacy<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ForagingAdapterPolicyValueNet<B> {
        ForagingAdapterPolicyValueNet {
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
            foraging_action_kind_head: nn::LinearConfig::new(self.hidden2, NUM_POLICY_ACTION_KINDS)
                .init(device),
            foraging_adapter_fc: nn::LinearConfig::new(
                self.hidden2 + FORAGING_ADAPTER_FEATURES,
                FORAGING_ADAPTER_HIDDEN,
            )
            .init(device),
            foraging_adapter_head: nn::LinearConfig::new(
                FORAGING_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            interaction_action_kind_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            exploration_action_kind_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS,
            )
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

    pub fn init_header_context_adapter_legacy<B: Backend>(
        &self,
        device: &B::Device,
    ) -> HeaderContextAdapterPolicyValueNet<B> {
        HeaderContextAdapterPolicyValueNet {
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
            foraging_action_kind_head: nn::LinearConfig::new(self.hidden2, NUM_POLICY_ACTION_KINDS)
                .init(device),
            foraging_adapter_fc: nn::LinearConfig::new(
                self.hidden2 + FORAGING_ADAPTER_FEATURES,
                FORAGING_ADAPTER_HIDDEN,
            )
            .init(device),
            foraging_adapter_head: nn::LinearConfig::new(
                FORAGING_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            interaction_action_kind_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            interaction_adapter_fc: nn::LinearConfig::new(
                self.hidden2 + OBS_RANDOMNESS_FEATURE_START,
                CONTEXT_ADAPTER_HIDDEN,
            )
            .init(device),
            interaction_adapter_head: nn::LinearConfig::new(
                CONTEXT_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            exploration_action_kind_head: nn::LinearConfig::new(
                self.hidden2,
                NUM_POLICY_ACTION_KINDS,
            )
            .init(device),
            exploration_adapter_fc: nn::LinearConfig::new(
                self.hidden2 + OBS_RANDOMNESS_FEATURE_START,
                CONTEXT_ADAPTER_HIDDEN,
            )
            .init(device),
            exploration_adapter_head: nn::LinearConfig::new(
                CONTEXT_ADAPTER_HIDDEN,
                NUM_POLICY_ACTION_KINDS,
            )
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

    pub fn migrate_legacy<B: Backend>(
        &self,
        legacy: LegacyPolicyValueNet<B>,
        device: &B::Device,
    ) -> PolicyValueNet<B> {
        let mut model = self.init(device);
        model.slot_encoder = legacy.slot_encoder;
        model.phase_slot_encoder = legacy.phase_slot_encoder;
        model.phase_gate_fc = legacy.phase_gate_fc;
        model.shared_fc1 = legacy.shared_fc1;
        model.recurrent = legacy.recurrent;
        model.shared_fc2 = legacy.shared_fc2;
        model.foraging_action_kind_head = legacy.foraging_action_kind_head;
        model.interaction_action_kind_head = legacy.interaction_action_kind_head;
        model.exploration_action_kind_head = legacy.exploration_action_kind_head;
        model.phase_gate_head = legacy.phase_gate_head;
        model.target_query_head = legacy.target_query_head;
        model.effort_head = legacy.effort_head;
        model.amount_head = legacy.amount_head;
        model.signal_head = legacy.signal_head;
        model.signal_strength_head = legacy.signal_strength_head;
        model.value_head = legacy.value_head;
        model
    }

    pub fn migrate_foraging_adapter<B: Backend>(
        &self,
        legacy: ForagingAdapterPolicyValueNet<B>,
        device: &B::Device,
    ) -> PolicyValueNet<B> {
        let mut model = self.init(device);
        model.slot_encoder = legacy.slot_encoder;
        model.phase_slot_encoder = legacy.phase_slot_encoder;
        model.phase_gate_fc = legacy.phase_gate_fc;
        model.shared_fc1 = legacy.shared_fc1;
        model.recurrent = legacy.recurrent;
        model.shared_fc2 = legacy.shared_fc2;
        model.foraging_action_kind_head = legacy.foraging_action_kind_head;
        model.foraging_adapter_fc = legacy.foraging_adapter_fc;
        model.foraging_adapter_head = legacy.foraging_adapter_head;
        model.interaction_action_kind_head = legacy.interaction_action_kind_head;
        model.exploration_action_kind_head = legacy.exploration_action_kind_head;
        model.phase_gate_head = legacy.phase_gate_head;
        model.target_query_head = legacy.target_query_head;
        model.effort_head = legacy.effort_head;
        model.amount_head = legacy.amount_head;
        model.signal_head = legacy.signal_head;
        model.signal_strength_head = legacy.signal_strength_head;
        model.value_head = legacy.value_head;
        model
    }

    pub fn migrate_header_context_adapter<B: Backend>(
        &self,
        legacy: HeaderContextAdapterPolicyValueNet<B>,
        device: &B::Device,
    ) -> PolicyValueNet<B> {
        let mut model = self.init(device);
        model.slot_encoder = legacy.slot_encoder;
        model.phase_slot_encoder = legacy.phase_slot_encoder;
        model.phase_gate_fc = legacy.phase_gate_fc;
        model.shared_fc1 = legacy.shared_fc1;
        model.recurrent = legacy.recurrent;
        model.shared_fc2 = legacy.shared_fc2;
        model.foraging_action_kind_head = legacy.foraging_action_kind_head;
        model.foraging_adapter_fc = legacy.foraging_adapter_fc;
        model.foraging_adapter_head = legacy.foraging_adapter_head;
        model.interaction_action_kind_head = legacy.interaction_action_kind_head;
        model.interaction_adapter_fc = legacy.interaction_adapter_fc;
        model.interaction_adapter_head = legacy.interaction_adapter_head;
        model.exploration_action_kind_head = legacy.exploration_action_kind_head;
        model.exploration_adapter_fc = legacy.exploration_adapter_fc;
        model.exploration_adapter_head = legacy.exploration_adapter_head;
        model.phase_gate_head = legacy.phase_gate_head;
        model.target_query_head = legacy.target_query_head;
        model.effort_head = legacy.effort_head;
        model.amount_head = legacy.amount_head;
        model.signal_head = legacy.signal_head;
        model.signal_strength_head = legacy.signal_strength_head;
        model.value_head = legacy.value_head;
        model
    }

    pub fn migrate_slot_context_adapter<B: Backend>(
        &self,
        legacy: SlotContextAdapterPolicyValueNet<B>,
        device: &B::Device,
    ) -> PolicyValueNet<B> {
        let mut model = self.init(device);
        model.slot_encoder = legacy.slot_encoder;
        model.phase_slot_encoder = legacy.phase_slot_encoder;
        model.phase_gate_fc = legacy.phase_gate_fc;
        model.shared_fc1 = legacy.shared_fc1;
        model.recurrent = legacy.recurrent;
        model.shared_fc2 = legacy.shared_fc2;
        model.foraging_action_kind_head = legacy.foraging_action_kind_head;
        model.foraging_adapter_fc = legacy.foraging_adapter_fc;
        model.foraging_adapter_head = legacy.foraging_adapter_head;
        model.interaction_action_kind_head = legacy.interaction_action_kind_head;
        model.interaction_adapter_fc = legacy.interaction_adapter_fc;
        model.interaction_adapter_head = legacy.interaction_adapter_head;
        model.interaction_slot_adapter_fc = legacy.interaction_slot_adapter_fc;
        model.interaction_slot_adapter_head = legacy.interaction_slot_adapter_head;
        model.exploration_action_kind_head = legacy.exploration_action_kind_head;
        model.exploration_adapter_fc = legacy.exploration_adapter_fc;
        model.exploration_adapter_head = legacy.exploration_adapter_head;
        model.exploration_slot_adapter_fc = legacy.exploration_slot_adapter_fc;
        model.exploration_slot_adapter_head = legacy.exploration_slot_adapter_head;
        model.phase_gate_head = legacy.phase_gate_head;
        model.target_query_head = legacy.target_query_head;
        model.effort_head = legacy.effort_head;
        model.amount_head = legacy.amount_head;
        model.signal_head = legacy.signal_head;
        model.signal_strength_head = legacy.signal_strength_head;
        model.value_head = legacy.value_head;
        model
    }
}

/// Output of a forward pass through the model.
pub struct ModelOutput<B: Backend> {
    /// Authoritatively context-routed inference logits consumed by PPO and
    /// deployed Minds.
    pub action_kind_logits: Tensor<B, 2>,
    /// Foraging, interaction, then exploration expert logits, flattened as
    /// [expert, action kind]. Supervised training routes each anonymous local
    /// observation to one expert; inference receives neither the teacher
    /// action nor host scenario metadata.
    pub action_kind_expert_logits: Tensor<B, 2>,
    /// Learned foraging/interaction/exploration gate logits retained only as
    /// diagnostic telemetry; they cannot override authoritative routing.
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
    /// Keep this model's complete parameter set except for one action-kind
    /// expert copied from `trained`. This supports stage-local supervised
    /// adaptation without allowing gradients for the shared recurrent trunk
    /// or unrelated experts to cause competency regressions.
    pub fn with_action_kind_expert_from(
        mut self,
        trained: Self,
        expert: ObservationExpertContext,
    ) -> Self {
        match expert {
            ObservationExpertContext::Foraging => {
                self.foraging_action_kind_head = trained.foraging_action_kind_head;
            }
            ObservationExpertContext::Interaction => {
                self.interaction_action_kind_head = trained.interaction_action_kind_head;
            }
            ObservationExpertContext::Exploration => {
                self.exploration_action_kind_head = trained.exploration_action_kind_head;
            }
        }
        self
    }

    /// Keep this model's complete parameter set except for the local
    /// foraging adapter copied from `trained`.
    pub fn with_foraging_adapter_from(mut self, trained: Self) -> Self {
        self.foraging_adapter_fc = trained.foraging_adapter_fc;
        self.foraging_adapter_head = trained.foraging_adapter_head;
        self
    }

    /// Keep this model's complete parameter set except for one non-foraging
    /// observation-local action-kind residual copied from `trained`.
    pub fn with_context_adapter_from(
        mut self,
        context: ObservationExpertContext,
        trained: Self,
    ) -> Self {
        match context {
            ObservationExpertContext::Foraging => {
                self.foraging_adapter_fc = trained.foraging_adapter_fc;
                self.foraging_adapter_head = trained.foraging_adapter_head;
            }
            ObservationExpertContext::Interaction => {
                self.interaction_adapter_fc = trained.interaction_adapter_fc;
                self.interaction_adapter_head = trained.interaction_adapter_head;
                self.interaction_slot_adapter_fc = trained.interaction_slot_adapter_fc;
                self.interaction_slot_adapter_head = trained.interaction_slot_adapter_head;
            }
            ObservationExpertContext::Exploration => {
                self.exploration_adapter_fc = trained.exploration_adapter_fc;
                self.exploration_adapter_head = trained.exploration_adapter_head;
                self.exploration_slot_adapter_fc = trained.exploration_slot_adapter_fc;
                self.exploration_slot_adapter_head = trained.exploration_slot_adapter_head;
            }
        }
        self
    }

    /// Keep this model's complete parameter set except for one non-random
    /// local-header plus featurewise pooled raw-slot residual from `trained`.
    pub fn with_context_slot_adapter_from(
        mut self,
        context: ObservationExpertContext,
        trained: Self,
    ) -> Self {
        match context {
            ObservationExpertContext::Foraging => {}
            ObservationExpertContext::Interaction => {
                self.interaction_slot_adapter_fc = trained.interaction_slot_adapter_fc;
                self.interaction_slot_adapter_head = trained.interaction_slot_adapter_head;
            }
            ObservationExpertContext::Exploration => {
                self.exploration_slot_adapter_fc = trained.exploration_slot_adapter_fc;
                self.exploration_slot_adapter_head = trained.exploration_slot_adapter_head;
            }
        }
        self
    }

    /// Keep every parameter except the scalar exploration Guard readiness
    /// residual. This is the narrowest adaptation path for the low-energy
    /// Guard/Move boundary.
    pub fn with_exploration_guard_readiness_from(mut self, trained: Self) -> Self {
        self.exploration_guard_readiness_head = trained.exploration_guard_readiness_head;
        self
    }

    /// Keep this model's complete parameter set except for the shared
    /// action-kind-conditioned target-query head copied from `trained`.
    pub fn with_target_query_head_from(mut self, trained: Self) -> Self {
        self.target_query_head = trained.target_query_head;
        self
    }

    /// Keep this model's complete parameter set except for the shared
    /// action-kind-conditioned effort head copied from `trained`.
    pub fn with_effort_head_from(mut self, trained: Self) -> Self {
        self.effort_head = trained.effort_head;
        self
    }

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
        let context_adapter_header = header
            .clone()
            .slice([0..batch, 0..OBS_RANDOMNESS_FEATURE_START]);
        let guard_readiness_features = header.clone().slice([0..batch, 1..3]);
        let foraging_adapter_features = Tensor::cat(
            vec![
                header.clone().slice([0..batch, 1..3]),
                header.clone().slice([0..batch, 10..13]),
                header.clone().slice([0..batch, 15..16]),
                header.clone().slice([0..batch, 19..21]),
            ],
            1,
        );
        let raw_slots = obs
            .slice([0..batch, HEADER_FEATURES..OBS_DIM])
            .reshape([batch * NUM_POLICY_TARGETS, SLOT_FEATURES]);
        let current_loose_energy = header
            .clone()
            .slice([
                0..batch,
                CURRENT_TILE_LOOSE_ENERGY_FEATURE..CURRENT_TILE_LOOSE_ENERGY_FEATURE + 1,
            ])
            .greater_elem(0.0);
        let current_plant = header
            .clone()
            .slice([
                0..batch,
                CURRENT_TILE_PLANT_CAPACITY_FEATURE..CURRENT_TILE_PLANT_CAPACITY_FEATURE + 1,
            ])
            .greater_elem(0.0);
        let current_food = current_plant.bool_or(current_loose_energy);
        let neighbor_present = raw_slots
            .clone()
            .slice([
                0..batch * NUM_POLICY_TARGETS,
                SLOT_NEIGHBOR_PRESENT_FEATURE..SLOT_NEIGHBOR_PRESENT_FEATURE + 1,
            ])
            .reshape([batch, NUM_POLICY_TARGETS])
            .max_dim(1)
            .greater_elem(0.5);
        let neighbor_activity = raw_slots.clone().slice([
            0..batch * NUM_POLICY_TARGETS,
            SLOT_NEIGHBOR_ACTIVITY_FEATURE..SLOT_NEIGHBOR_ACTIVITY_FEATURE + 1,
        ]);
        let visible_threat = neighbor_activity
            .clone()
            .equal_elem(3.0 / 8.0)
            .bool_or(neighbor_activity.equal_elem(4.0 / 8.0))
            .float()
            .reshape([batch, NUM_POLICY_TARGETS])
            .max_dim(1)
            .greater_elem(0.5);
        let foraging_context = visible_threat
            .clone()
            .bool_not()
            .bool_and(current_food.clone());
        let interaction_context = visible_threat.clone().bool_or(
            current_food
                .clone()
                .bool_not()
                .bool_and(neighbor_present.clone()),
        );
        let exploration_context = visible_threat
            .bool_not()
            .bool_and(current_food.bool_not())
            .bool_and(neighbor_present.bool_not());
        let phase_slots =
            burn::tensor::activation::relu(self.phase_slot_encoder.forward(raw_slots.clone()))
                .reshape([batch, NUM_POLICY_TARGETS, self.hidden2()]);
        let phase_pooled_slots = phase_slots.max_dim(1).reshape([batch, self.hidden2()]);
        let phase_gate_features = burn::tensor::activation::relu(
            self.phase_gate_fc
                .forward(Tensor::cat(vec![header.clone(), phase_pooled_slots], 1)),
        );
        let slots = burn::tensor::activation::relu(self.slot_encoder.forward(raw_slots.clone()))
            .reshape([batch, NUM_POLICY_TARGETS, self.hidden2()]);
        let pooled_slots = slots.clone().max_dim(1).reshape([batch, self.hidden2()]);
        let raw_slot_summary = raw_slots
            .reshape([batch, NUM_POLICY_TARGETS, SLOT_FEATURES])
            .max_dim(1)
            .reshape([batch, SLOT_FEATURES]);
        let x = self
            .shared_fc1
            .forward(Tensor::cat(vec![header, pooled_slots], 1));
        let x = burn::tensor::activation::relu(x);
        let next_memory =
            burn::tensor::activation::tanh(self.recurrent.forward(Tensor::cat(vec![x, memory], 1)));
        let x = self.shared_fc2.forward(next_memory.clone());
        let x = burn::tensor::activation::relu(x);

        let foraging_adapter = burn::tensor::activation::relu(self.foraging_adapter_fc.forward(
            Tensor::cat(vec![x.clone(), foraging_adapter_features.clone()], 1),
        ));
        let foraging_action_kind_logits = self.foraging_action_kind_head.forward(x.clone())
            + self.foraging_adapter_head.forward(foraging_adapter);
        let context_adapter_features =
            Tensor::cat(vec![x.clone(), context_adapter_header.clone()], 1);
        let local_context_features = Tensor::cat(vec![context_adapter_header, raw_slot_summary], 1);
        let interaction_adapter = burn::tensor::activation::relu(
            self.interaction_adapter_fc
                .forward(context_adapter_features.clone()),
        );
        let interaction_action_kind_logits = self.interaction_action_kind_head.forward(x.clone())
            + self.interaction_adapter_head.forward(interaction_adapter)
            + self
                .interaction_slot_adapter_head
                .forward(burn::tensor::activation::relu(
                    self.interaction_slot_adapter_fc
                        .forward(local_context_features.clone()),
                ));
        let exploration_adapter = burn::tensor::activation::relu(
            self.exploration_adapter_fc
                .forward(context_adapter_features),
        );
        let guard_readiness = self
            .exploration_guard_readiness_head
            .forward(guard_readiness_features);
        let zero_readiness = guard_readiness.clone() * 0.0;
        let guard_readiness_logits = Tensor::cat(
            vec![
                zero_readiness.clone(),
                guard_readiness,
                zero_readiness.repeat_dim(
                    1,
                    NUM_POLICY_ACTION_KINDS - PolicyActionKind::Guard.index() - 1,
                ),
            ],
            1,
        );
        let exploration_action_kind_logits = self.exploration_action_kind_head.forward(x.clone())
            + self.exploration_adapter_head.forward(exploration_adapter)
            + self
                .exploration_slot_adapter_head
                .forward(burn::tensor::activation::relu(
                    self.exploration_slot_adapter_fc
                        .forward(local_context_features),
                ))
            + guard_readiness_logits;
        let phase_gate_logits = self.phase_gate_head.forward(phase_gate_features);
        let foraging_weight = foraging_context
            .float()
            .repeat_dim(1, NUM_POLICY_ACTION_KINDS);
        let interaction_weight = interaction_context
            .float()
            .repeat_dim(1, NUM_POLICY_ACTION_KINDS);
        let exploration_weight = exploration_context
            .float()
            .repeat_dim(1, NUM_POLICY_ACTION_KINDS);
        let action_kind_probs =
            burn::tensor::activation::softmax(foraging_action_kind_logits.clone(), 1)
                * foraging_weight
                + burn::tensor::activation::softmax(interaction_action_kind_logits.clone(), 1)
                    * interaction_weight
                + burn::tensor::activation::softmax(exploration_action_kind_logits.clone(), 1)
                    * exploration_weight;
        let action_kind_logits = action_kind_probs.clamp_min(1.0e-20).log();
        let action_kind_expert_logits = Tensor::cat(
            vec![
                foraging_action_kind_logits,
                interaction_action_kind_logits,
                exploration_action_kind_logits,
            ],
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
    use crate::observation::{
        CURRENT_TILE_DIFFUSE_ENERGY_FEATURE, CURRENT_TILE_PLANT_CAPACITY_FEATURE,
    };
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
        assert_eq!(PolicyValueNetConfig::new().parameter_count(), 104_780);
    }

    #[test]
    fn large_capacity_profile_has_an_explicit_cost() {
        let large = PolicyValueNetConfig {
            hidden1: 256,
            hidden2: 128,
            recurrent_size: 128,
        };
        assert_eq!(large.parameter_count(), 344_140);
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
    fn authoritative_context_selects_the_matching_action_expert() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        TestBackend::seed(&device, 771);
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig::new().init(&device);
        let mut observations = vec![vec![0.0; OBS_DIM]; 6];
        observations[0][CURRENT_TILE_PLANT_CAPACITY_FEATURE] = 0.2;
        observations[1][HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        observations[3][CURRENT_TILE_PLANT_CAPACITY_FEATURE] = 0.2;
        observations[3][HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        observations[3][HEADER_FEATURES + SLOT_NEIGHBOR_ACTIVITY_FEATURE] = 3.0 / 8.0;
        observations[4][CURRENT_TILE_DIFFUSE_ENERGY_FEATURE] = 0.2;
        observations[5][CURRENT_TILE_PLANT_CAPACITY_FEATURE] = 0.2;
        let expected_contexts = [
            ObservationExpertContext::Foraging,
            ObservationExpertContext::Interaction,
            ObservationExpertContext::Exploration,
            ObservationExpertContext::Interaction,
            ObservationExpertContext::Exploration,
            ObservationExpertContext::Foraging,
        ];
        assert_eq!(
            observations
                .iter()
                .map(|observation| crate::observation::observation_expert_context(observation))
                .collect::<Vec<_>>(),
            expected_contexts
        );

        let output = model.forward(Tensor::from_data(
            TensorData::new(observations.concat(), [6, OBS_DIM]),
            &device,
        ));
        let routed = output
            .action_kind_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let experts = output
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (row, context) in expected_contexts.into_iter().enumerate() {
            let start = row * NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS
                + context.index() * NUM_POLICY_ACTION_KINDS;
            let selected = &experts[start..start + NUM_POLICY_ACTION_KINDS];
            let maximum = selected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_normalizer = maximum
                + selected
                    .iter()
                    .map(|value| (*value - maximum).exp())
                    .sum::<f32>()
                    .ln();
            for action in 0..NUM_POLICY_ACTION_KINDS {
                let actual = routed[row * NUM_POLICY_ACTION_KINDS + action];
                let expected = selected[action] - log_normalizer;
                assert!((actual - expected).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    fn expert_replacement_preserves_every_unselected_parameter_path() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        TestBackend::seed(&device, 772);
        let parent: PolicyValueNet<TestBackend> = PolicyValueNetConfig::new().init(&device);
        TestBackend::seed(&device, 773);
        let donor: PolicyValueNet<TestBackend> = PolicyValueNetConfig::new().init(&device);
        let mut trained = parent.clone();
        trained.exploration_action_kind_head = donor.exploration_action_kind_head;
        let merged = parent
            .clone()
            .with_action_kind_expert_from(trained.clone(), ObservationExpertContext::Exploration);
        let observations = Tensor::<TestBackend, 2>::zeros([2, OBS_DIM], &device);
        let parent_output = parent.forward(observations.clone());
        let trained_output = trained.forward(observations.clone());
        let merged_output = merged.forward(observations);
        let parent_experts = parent_output
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let trained_experts = trained_output
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let merged_experts = merged_output
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for row in 0..2 {
            for expert in 0..NUM_ACTION_KIND_EXPERTS {
                let start = (row * NUM_ACTION_KIND_EXPERTS + expert) * NUM_POLICY_ACTION_KINDS;
                let end = start + NUM_POLICY_ACTION_KINDS;
                let expected = if expert == ObservationExpertContext::Exploration.index() {
                    &trained_experts[start..end]
                } else {
                    &parent_experts[start..end]
                };
                assert_eq!(&merged_experts[start..end], expected);
            }
        }
        assert_eq!(
            merged_output
                .phase_gate_logits
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            parent_output
                .phase_gate_logits
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            merged_output
                .next_memory
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            parent_output
                .next_memory
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        );
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
