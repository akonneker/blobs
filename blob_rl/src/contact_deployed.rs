//! Canonical contact/skirmish evaluation through ordinary anonymous Mind calls.
use crate::{
    config::{CombatCurriculumConfig, FeedingCurriculumStage, OpponentProfile, TrainingConfig},
    contact_evaluation::{
        finalize_variant, report_from_variants, ContactEvaluationReport, ContactVariantMetrics,
    },
    env::{BlobEnv, EpisodeOutcome},
    telemetry::{ActionTelemetry, TelemetryConfig},
};
use blob_interface::reference_mind::*;
use blob_policy::action::action_is_commit_legal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
pub const CONTACT_DEPLOYED_SCHEMA_VERSION: u32 = 1;
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContactDecisions {
    pub total: u64,
    pub illegal: u64,
    pub action_kinds: BTreeMap<String, u64>,
}
struct ObservedMind<'a> {
    mind: &'a mut dyn ReferenceMind,
    counts: ContactDecisions,
}
impl ReferenceMind for ObservedMind<'_> {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let decision = self.mind.decide(input);
        self.counts.total += 1;
        self.counts.illegal += u64::from(!action_is_commit_legal(
            input,
            &decision.action,
            decision.signal.is_some(),
        ));
        let kind = match &decision.action {
            ReferenceMindAction::Wait => "Wait",
            ReferenceMindAction::Move { .. } => "Move",
            ReferenceMindAction::Attack { .. } => "Attack",
            ReferenceMindAction::Guard { .. } => "Guard",
            ReferenceMindAction::Consume { .. } => "Consume",
            ReferenceMindAction::Split { .. } => "Split",
            ReferenceMindAction::Regurgitate { .. } => "Regurgitate",
            ReferenceMindAction::Signal { .. } => "Signal",
            ReferenceMindAction::Excavate => "Excavate",
            ReferenceMindAction::DepositTerrain => "DepositTerrain",
        };
        *self.counts.action_kinds.entry(kind.into()).or_default() += 1;
        decision
    }
    fn reset(&mut self) -> Result<(), String> {
        self.mind.reset()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContactMindEpisode {
    pub stage: FeedingCurriculumStage,
    pub seed: u64,
    pub initial_energy: u32,
    pub opponent: OpponentProfile,
    pub cells_per_team: usize,
    pub effective_env_sha256: String,
    pub initial_state_hash: String,
    pub first_frontier_sha256: String,
    pub final_state_hash: String,
    pub ruleset_hash: String,
    pub horizon_quanta: u64,
    pub max_episode_steps: u64,
    pub elapsed_quanta: u64,
    pub steps: u64,
    pub training_cells: usize,
    pub opponent_cells: usize,
    pub outcome: String,
    pub end_reason: String,
    pub attack: ActionTelemetry,
    pub movement: ActionTelemetry,
    pub consume: ActionTelemetry,
    pub damage_dealt: u128,
    pub kills: u64,
    pub decisions: ContactDecisions,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContactMindEvaluation {
    pub schema_version: u32,
    pub report: ContactEvaluationReport,
    pub episodes: Vec<ContactMindEpisode>,
    pub combat_thresholds_passed: bool,
}
pub fn evaluate_contact_mind(
    mind: &mut dyn ReferenceMind,
    config: &TrainingConfig,
    seeds: &[u64],
) -> Result<ContactMindEvaluation, String> {
    config.validate()?;
    if seeds.is_empty()
        || !config.combat_curriculum.enabled
        || seeds
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != seeds.len()
    {
        return Err("contact evaluation requires enabled combat and unique nonempty seeds".into());
    }
    let base_env = &config.env;
    let reward = &config.reward;
    let feeding = &config.feeding_curriculum;
    let combat = &config.combat_curriculum;
    let mut episodes = Vec::new();
    let contact_start = combat
        .on_food_sim_time_quanta_per_cycle
        .saturating_add(combat.adjacent_food_sim_time_quanta_per_cycle);
    let contact_template = combat.evaluation_environment_for_stage(
        feeding,
        base_env,
        FeedingCurriculumStage::Contact,
        contact_start,
    );
    let ruleset_hash =
        BlobEnv::new(contact_template, reward.clone(), seeds[0]).compiled_ruleset_hash();
    let mut variants = Vec::new();

    for stage in CombatCurriculumConfig::combat_stages() {
        let stage_start = match stage {
            FeedingCurriculumStage::Contact => contact_start,
            FeedingCurriculumStage::Skirmish => {
                contact_start.saturating_add(combat.contact_sim_time_quanta_per_cycle)
            }
            _ => unreachable!(),
        };
        let template =
            combat.evaluation_environment_for_stage(feeding, base_env, stage, stage_start);
        for &initial_energy in &combat.contact_initial_energies {
            for &opponent in &combat.contact_opponents {
                let mut metrics = ContactVariantMetrics {
                    stage,
                    cells_per_team: template.cells_per_team,
                    initial_energy,
                    opponent,
                    episodes: seeds.len(),
                    wins: 0,
                    losses: 0,
                    timeouts: 0,
                    safety_aborts: 0,
                    attacking_episodes: 0,
                    damaging_episodes: 0,
                    attacks_committed: 0,
                    attacks_succeeded: 0,
                    attacks_frustrated: 0,
                    attacks_interrupted: 0,
                    damage_dealt: 0,
                    kills: 0,
                    attacking_episode_rate: 0.0,
                    damaging_episode_rate: 0.0,
                    attack_success_rate: 0.0,
                };
                for &seed in seeds {
                    let mut env_config = template.clone();
                    env_config.initial_energy = initial_energy;
                    env_config.opponent = opponent;
                    let effective_env_sha256 = crate::sweep::sha256(
                        &serde_json::to_vec(&env_config).map_err(|e| e.to_string())?,
                    );
                    let max_episode_steps = env_config.max_episode_len;
                    let horizon_quanta = env_config.victory.sim_time_limit_quanta;
                    let mut env = BlobEnv::new(env_config, reward.clone(), seed);
                    let initial_state_hash = env
                        .reference_simulation_for_diagnostics()?
                        .state_hash()
                        .to_hex();
                    let mut frontier = Sha256::new();
                    for (_, input) in env.prepare_training_reference_inputs()? {
                        let bytes = blob_interface::reference_mind_converter::reference_mind_input_to_capnp(&input, Default::default()).map_err(|e|e.to_string())?;
                        frontier.update((bytes.len() as u64).to_le_bytes());
                        frontier.update(bytes);
                    }
                    let first_frontier_sha256 = format!("{:x}", frontier.finalize());
                    mind.reset()?;
                    let mut observed = ObservedMind {
                        mind,
                        counts: Default::default(),
                    };
                    let mut attack_total = ActionTelemetry::default();
                    let mut movement = ActionTelemetry::default();
                    let mut consume = ActionTelemetry::default();
                    let mut damage_dealt = 0;
                    let mut kills = 0;
                    env.enable_telemetry(TelemetryConfig {
                        enabled: true,
                        state_sample_interval_steps: u64::MAX,
                        max_state_samples_per_episode: 2,
                        episode_log_stride: 0,
                    });
                    let mut episode_attacks = 0u64;
                    let mut episode_damage = 0u128;
                    loop {
                        let result = env.step_with_reference_mind(&mut observed);
                        let telemetry = result
                            .telemetry
                            .as_ref()
                            .expect("contact evaluation telemetry is enabled");
                        let attack = &telemetry.training.actions.attack;
                        attack_total.merge(attack);
                        movement.merge(&telemetry.training.actions.movement);
                        consume.merge(&telemetry.training.actions.consume);
                        damage_dealt += telemetry.training.damage.applied_dealt;
                        kills += telemetry.training.kills;
                        episode_attacks = episode_attacks.saturating_add(attack.committed);
                        episode_damage =
                            episode_damage.saturating_add(telemetry.training.damage.applied_dealt);
                        metrics.attacks_committed =
                            metrics.attacks_committed.saturating_add(attack.committed);
                        metrics.attacks_succeeded =
                            metrics.attacks_succeeded.saturating_add(attack.succeeded);
                        metrics.attacks_frustrated =
                            metrics.attacks_frustrated.saturating_add(attack.frustrated);
                        metrics.attacks_interrupted = metrics
                            .attacks_interrupted
                            .saturating_add(attack.interrupted);
                        metrics.damage_dealt = metrics
                            .damage_dealt
                            .saturating_add(telemetry.training.damage.applied_dealt);
                        metrics.kills = metrics.kills.saturating_add(telemetry.training.kills);
                        if result.done {
                            match result
                                .outcome
                                .expect("completed contact episode has an outcome")
                            {
                                EpisodeOutcome::Win => metrics.wins += 1,
                                EpisodeOutcome::Loss => metrics.losses += 1,
                                EpisodeOutcome::Timeout => metrics.timeouts += 1,
                                EpisodeOutcome::SafetyAbort => metrics.safety_aborts += 1,
                            }
                            episodes.push(ContactMindEpisode {
                                stage,
                                initial_energy,
                                opponent,
                                seed,
                                cells_per_team: template.cells_per_team,
                                effective_env_sha256,
                                initial_state_hash,
                                first_frontier_sha256,
                                final_state_hash: env
                                    .reference_simulation_for_diagnostics()?
                                    .state_hash()
                                    .to_hex(),
                                ruleset_hash: env.compiled_ruleset_hash(),
                                horizon_quanta,
                                max_episode_steps,
                                elapsed_quanta: env.sim_time_quanta(),
                                steps: result.episode_step,
                                training_cells: result.training_cells,
                                opponent_cells: result.opponent_cells,
                                outcome: format!("{:?}", result.outcome),
                                end_reason: format!("{:?}", result.end_reason),
                                attack: attack_total,
                                movement,
                                consume,
                                damage_dealt,
                                kills,
                                decisions: observed.counts,
                            });
                            break;
                        }
                    }
                    metrics.attacking_episodes += usize::from(episode_attacks > 0);
                    metrics.damaging_episodes += usize::from(episode_damage > 0);
                }
                variants.push(finalize_variant(metrics));
            }
        }
    }

    let report = report_from_variants(
        ruleset_hash,
        seeds.to_vec(),
        combat.contact_evaluation_sim_time_limit_quanta,
        combat.skirmish_evaluation_sim_time_limit_quanta,
        variants,
    );
    report.validate_against(&report.ruleset_hash, seeds, combat)?;
    let combat_thresholds_passed = report.meets_promotion_thresholds(combat);
    Ok(ContactMindEvaluation {
        schema_version: CONTACT_DEPLOYED_SCHEMA_VERSION,
        report,
        episodes,
        combat_thresholds_passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::ActionBufferMind;
    fn config() -> TrainingConfig {
        let mut c =
            TrainingConfig::from_toml_str(include_str!("../config/combat_warm_start_256.toml"))
                .unwrap();
        c.env.world_size = 32;
        c.env.cells_per_team = 4;
        c.env.num_scattered_energy = 16;
        c.env.num_plants = 4;
        c.combat_curriculum.contact_initial_energies = vec![100];
        c.combat_curriculum.contact_opponents = vec![OpponentProfile::Defensive];
        c.combat_curriculum.contact_evaluation_sim_time_limit_quanta = 8192;
        c.combat_curriculum
            .skirmish_evaluation_sim_time_limit_quanta = 8192;
        c
    }
    #[test]
    fn mind_contact_matches_canonical_baseline_state_and_resolver_metrics() {
        let c = config();
        let actual = evaluate_contact_mind(
            &mut ActionBufferMind::new_opponent(OpponentProfile::Aggressive),
            &c,
            &[701],
        )
        .unwrap();
        for episode in &actual.episodes {
            let combat = &c.combat_curriculum;
            let start = combat.on_food_sim_time_quanta_per_cycle
                + combat.adjacent_food_sim_time_quanta_per_cycle
                + if episode.stage == FeedingCurriculumStage::Skirmish {
                    combat.contact_sim_time_quanta_per_cycle
                } else {
                    0
                };
            let mut ec = combat.evaluation_environment_for_stage(
                &c.feeding_curriculum,
                &c.env,
                episode.stage,
                start,
            );
            ec.initial_energy = episode.initial_energy;
            ec.opponent = episode.opponent;
            let mut env = BlobEnv::new(ec, c.reward.clone(), episode.seed);
            assert_eq!(
                episode.initial_state_hash,
                env.reference_simulation_for_diagnostics()
                    .unwrap()
                    .state_hash()
                    .to_hex()
            );
            env.enable_telemetry(TelemetryConfig {
                enabled: true,
                state_sample_interval_steps: u64::MAX,
                max_state_samples_per_episode: 2,
                episode_log_stride: 0,
            });
            let mut attacks = ActionTelemetry::default();
            let mut damage = 0;
            let mut kills = 0;
            let terminal = loop {
                let result = env.step_with_baseline(OpponentProfile::Aggressive);
                let t = result.telemetry.as_ref().unwrap();
                attacks.merge(&t.training.actions.attack);
                damage += t.training.damage.applied_dealt;
                kills += t.training.kills;
                if result.done {
                    break result;
                }
            };
            assert_eq!(
                episode.final_state_hash,
                env.reference_simulation_for_diagnostics()
                    .unwrap()
                    .state_hash()
                    .to_hex()
            );
            assert_eq!(episode.attack, attacks);
            assert_eq!(episode.damage_dealt, damage);
            assert_eq!(episode.kills, kills);
            assert_eq!(episode.steps, terminal.episode_step);
            assert_eq!(episode.outcome, format!("{:?}", terminal.outcome));
            assert_eq!(episode.decisions.illegal, 0);
        }
        assert!(actual.report.damage_dealt > 0);
    }
    #[test]
    fn passive_minds_do_not_pass_combat_gate() {
        let c = config();
        let passive = evaluate_contact_mind(
            &mut ActionBufferMind::new_opponent(OpponentProfile::Wait),
            &c,
            &[702],
        )
        .unwrap();
        assert!(!passive.combat_thresholds_passed);
        assert_eq!(passive.report.attacks_committed, 0);
    }
}
