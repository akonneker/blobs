use blob_interface::reference_mind::*;
use blob_policy::{
    action::*, composite::DeployedPolicy, memory::decode_policy_memory, observation::Observation,
};
use serde::Serialize;
#[derive(Serialize)]
pub struct Row {
    episode: usize,
    context: String,
    selected_kind: usize,
    teacher_kind: usize,
    legal_occupied_attack: bool,
    legal_kinds: Vec<bool>,
    raw_kind: Vec<f32>,
    base_kind: Vec<f32>,
    context_delta: Vec<f32>,
    slot_delta: Vec<f32>,
    observation: Vec<f32>,
    memory: Vec<f32>,
    hidden: Vec<f32>,
}
pub struct Probe {
    pub policy: DeployedPolicy,
    pub rows: Vec<Row>,
    pub episode: usize,
    pub interaction_only: bool,
    pub per_episode_limit: usize,
    pub recorded: usize,
}
fn kind(action: &ReferenceMindAction) -> usize {
    match action {
        ReferenceMindAction::Wait => PolicyActionKind::Wait,
        ReferenceMindAction::Guard { .. } => PolicyActionKind::Guard,
        ReferenceMindAction::Consume { .. } => PolicyActionKind::Consume,
        ReferenceMindAction::Move { .. } => PolicyActionKind::Move,
        ReferenceMindAction::Attack { .. } => PolicyActionKind::Attack,
        ReferenceMindAction::Split { .. } => PolicyActionKind::Split,
        ReferenceMindAction::Regurgitate { .. } => PolicyActionKind::Regurgitate,
        ReferenceMindAction::Signal { .. } => PolicyActionKind::Signal,
        ReferenceMindAction::Excavate => PolicyActionKind::Excavate,
        ReferenceMindAction::DepositTerrain => PolicyActionKind::DepositTerrain,
    }
    .index()
}
impl ReferenceMind for Probe {
    fn reset(&mut self) -> Result<(), String> {
        self.episode += 1;
        self.recorded = 0;
        Ok(())
    }
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let decision = self.policy.try_decide(input).unwrap();
        let parent = self.policy.parent();
        let observation = Observation::from_reference(input);
        if self.recorded >= self.per_episode_limit
            || (self.interaction_only
                && blob_policy::observation::observation_expert_context(&observation.data)
                    != blob_policy::observation::ObservationExpertContext::Interaction)
        {
            return decision;
        }
        self.recorded += 1;
        let memory = decode_policy_memory(&input.private_memory, parent.recurrent_size());
        let (_, trace) = parent
            .forward_with_trace(&observation.data, &memory)
            .unwrap();
        let teacher = aggressive_mind::decide(input);
        let occupied = input.slots.iter().any(|slot| {
            slot.neighbor.is_some()
                && [
                    ReferenceEffort::Gentle,
                    ReferenceEffort::Standard,
                    ReferenceEffort::Burst,
                ]
                .into_iter()
                .any(|effort| {
                    action_is_commit_legal(
                        input,
                        &ReferenceMindAction::Attack {
                            target_slot: slot.slot,
                            effort,
                            payload: 1,
                        },
                        decision.signal.is_some(),
                    )
                })
        });
        self.rows.push(Row {
            episode: self.episode - 1,
            context: format!("{:?}", trace.context),
            selected_kind: kind(&decision.action),
            teacher_kind: kind(&teacher.action),
            legal_occupied_attack: occupied,
            legal_kinds: policy_action_kind_mask(&observation.action_mask).to_vec(),
            raw_kind: trace.raw_kind,
            base_kind: trace.base_kind,
            context_delta: trace.context_delta,
            slot_delta: trace.slot_delta,
            observation: observation.data.to_vec(),
            memory,
            hidden: trace.hidden,
        });
        decision
    }
}
