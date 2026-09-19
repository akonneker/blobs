//! Held-out direct-contact and local-skirmish characterization.

use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::{
    CombatCurriculumConfig, EnvConfig, FeedingCurriculumConfig, FeedingCurriculumStage,
    OpponentProfile, RewardConfig,
};
use crate::env::{BlobEnv, EpisodeOutcome};
use crate::evaluation::greedy_policy_choices;
use crate::model::PolicyValueNet;
use crate::telemetry::TelemetryConfig;

pub const CONTACT_EVALUATION_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContactVariantMetrics {
    pub stage: FeedingCurriculumStage,
    pub cells_per_team: usize,
    pub initial_energy: u32,
    pub opponent: OpponentProfile,
    pub episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub safety_aborts: usize,
    pub attacking_episodes: usize,
    pub damaging_episodes: usize,
    pub attacks_committed: u64,
    pub attacks_succeeded: u64,
    pub attacks_frustrated: u64,
    pub attacks_interrupted: u64,
    pub damage_dealt: u128,
    pub kills: u64,
    pub attacking_episode_rate: f64,
    pub damaging_episode_rate: f64,
    pub attack_success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContactEvaluationReport {
    pub schema_version: u32,
    pub ruleset_hash: String,
    pub seeds: Vec<u64>,
    pub contact_sim_time_limit_quanta: u64,
    pub skirmish_sim_time_limit_quanta: u64,
    pub variants: Vec<ContactVariantMetrics>,
    pub episodes: usize,
    pub attacking_episodes: usize,
    pub damaging_episodes: usize,
    pub attacks_committed: u64,
    pub attacks_succeeded: u64,
    pub damage_dealt: u128,
    pub kills: u64,
    pub attacking_episode_rate: f64,
    pub damaging_episode_rate: f64,
    pub attack_success_rate: f64,
}

impl ContactEvaluationReport {
    /// True when the policy completed at least one attack that applied damage,
    /// with no safety-aborted characterization episode.
    pub fn is_active(&self) -> bool {
        self.variants
            .iter()
            .all(|variant| variant.safety_aborts == 0)
            && CombatCurriculumConfig::combat_stages().iter().all(|stage| {
                let variants = self
                    .variants
                    .iter()
                    .filter(|variant| variant.stage == *stage)
                    .collect::<Vec<_>>();
                !variants.is_empty()
                    && variants
                        .iter()
                        .map(|variant| variant.attacks_succeeded)
                        .sum::<u64>()
                        > 0
                    && variants
                        .iter()
                        .map(|variant| variant.damage_dealt)
                        .sum::<u128>()
                        > 0
            })
    }

    pub fn kills_for_stage(&self, stage: FeedingCurriculumStage) -> u64 {
        self.variants
            .iter()
            .filter(|variant| variant.stage == stage)
            .map(|variant| variant.kills)
            .sum()
    }

    /// Promotion-quality local combat requires base activity plus the exact
    /// configured resolver-attributed kill floors for both populations.
    pub fn meets_promotion_thresholds(&self, combat: &CombatCurriculumConfig) -> bool {
        self.is_active()
            && self.kills_for_stage(FeedingCurriculumStage::Contact)
                >= combat.min_contact_kills_for_promotion
            && self.kills_for_stage(FeedingCurriculumStage::Skirmish)
                >= combat.min_skirmish_kills_for_promotion
    }

    /// Reject stale, truncated, or internally inconsistent contact evidence.
    pub fn validate_against(
        &self,
        expected_ruleset_hash: &str,
        expected_seeds: &[u64],
        combat: &CombatCurriculumConfig,
    ) -> Result<(), String> {
        if self.schema_version != CONTACT_EVALUATION_SCHEMA_VERSION
            || self.ruleset_hash != expected_ruleset_hash
            || self.seeds != expected_seeds
            || self.seeds.is_empty()
            || self.contact_sim_time_limit_quanta != combat.contact_evaluation_sim_time_limit_quanta
            || self.skirmish_sim_time_limit_quanta
                != combat.skirmish_evaluation_sim_time_limit_quanta
        {
            return Err("contact evaluation schema, ruleset, or seeds do not match".into());
        }
        let expected_variants = combat
            .contact_initial_energies
            .len()
            .saturating_mul(combat.contact_opponents.len())
            .saturating_mul(CombatCurriculumConfig::combat_stages().len());
        if self.variants.len() != expected_variants {
            return Err("contact evaluation has the wrong variant count".into());
        }
        for stage in CombatCurriculumConfig::combat_stages() {
            for &energy in &combat.contact_initial_energies {
                for &opponent in &combat.contact_opponents {
                    if self
                        .variants
                        .iter()
                        .filter(|variant| {
                            variant.stage == stage
                                && variant.cells_per_team == combat.combat_cells_per_team(stage)
                                && variant.initial_energy == energy
                                && variant.opponent == opponent
                        })
                        .count()
                        != 1
                    {
                        return Err("contact evaluation variants are missing or duplicated".into());
                    }
                }
            }
        }
        for variant in &self.variants {
            if !CombatCurriculumConfig::combat_stages().contains(&variant.stage)
                || variant.cells_per_team != combat.combat_cells_per_team(variant.stage)
                || variant.episodes != self.seeds.len()
                || variant.wins + variant.losses + variant.timeouts + variant.safety_aborts
                    != variant.episodes
                || variant.attacking_episodes > variant.episodes
                || variant.damaging_episodes > variant.attacking_episodes
                || variant.attacks_succeeded > variant.attacks_committed
                || !close(
                    variant.attacking_episode_rate,
                    ratio(variant.attacking_episodes, variant.episodes),
                )
                || !close(
                    variant.damaging_episode_rate,
                    ratio(variant.damaging_episodes, variant.episodes),
                )
                || !close(
                    variant.attack_success_rate,
                    ratio_u64(variant.attacks_succeeded, variant.attacks_committed),
                )
            {
                return Err("contact evaluation variant totals or rates are inconsistent".into());
            }
        }
        let recomputed = report_from_variants(
            self.ruleset_hash.clone(),
            self.seeds.clone(),
            self.contact_sim_time_limit_quanta,
            self.skirmish_sim_time_limit_quanta,
            self.variants.clone(),
        );
        if self.episodes != recomputed.episodes
            || self.attacking_episodes != recomputed.attacking_episodes
            || self.damaging_episodes != recomputed.damaging_episodes
            || self.attacks_committed != recomputed.attacks_committed
            || self.attacks_succeeded != recomputed.attacks_succeeded
            || self.damage_dealt != recomputed.damage_dealt
            || self.kills != recomputed.kills
            || !close(
                self.attacking_episode_rate,
                recomputed.attacking_episode_rate,
            )
            || !close(self.damaging_episode_rate, recomputed.damaging_episode_rate)
            || !close(self.attack_success_rate, recomputed.attack_success_rate)
        {
            return Err("contact evaluation aggregate totals or rates are inconsistent".into());
        }
        Ok(())
    }
}

fn close(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= 1e-12
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn ratio_u64(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub(crate) fn finalize_variant(mut metrics: ContactVariantMetrics) -> ContactVariantMetrics {
    metrics.attacking_episode_rate = ratio(metrics.attacking_episodes, metrics.episodes);
    metrics.damaging_episode_rate = ratio(metrics.damaging_episodes, metrics.episodes);
    metrics.attack_success_rate = ratio_u64(metrics.attacks_succeeded, metrics.attacks_committed);
    metrics
}

pub(crate) fn report_from_variants(
    ruleset_hash: String,
    seeds: Vec<u64>,
    contact_sim_time_limit_quanta: u64,
    skirmish_sim_time_limit_quanta: u64,
    variants: Vec<ContactVariantMetrics>,
) -> ContactEvaluationReport {
    let episodes = variants.iter().map(|variant| variant.episodes).sum();
    let attacking_episodes = variants
        .iter()
        .map(|variant| variant.attacking_episodes)
        .sum();
    let damaging_episodes = variants
        .iter()
        .map(|variant| variant.damaging_episodes)
        .sum();
    let attacks_committed = variants
        .iter()
        .map(|variant| variant.attacks_committed)
        .sum();
    let attacks_succeeded = variants
        .iter()
        .map(|variant| variant.attacks_succeeded)
        .sum();
    let damage_dealt = variants.iter().map(|variant| variant.damage_dealt).sum();
    let kills = variants.iter().map(|variant| variant.kills).sum();
    ContactEvaluationReport {
        schema_version: CONTACT_EVALUATION_SCHEMA_VERSION,
        ruleset_hash,
        seeds,
        contact_sim_time_limit_quanta,
        skirmish_sim_time_limit_quanta,
        variants,
        episodes,
        attacking_episodes,
        damaging_episodes,
        attacks_committed,
        attacks_succeeded,
        damage_dealt,
        kills,
        attacking_episode_rate: ratio(attacking_episodes, episodes),
        damaging_episode_rate: ratio(damaging_episodes, episodes),
        attack_success_rate: ratio_u64(attacks_succeeded, attacks_committed),
    }
}

/// Exercise every configured energy/opponent combination using greedy policy
/// decisions and the ordinary anonymous Mind boundary.
pub fn evaluate_contact<B: Backend>(
    model: &PolicyValueNet<B>,
    base_env: &EnvConfig,
    reward: &RewardConfig,
    feeding: &FeedingCurriculumConfig,
    combat: &CombatCurriculumConfig,
    seeds: &[u64],
    device: &B::Device,
) -> ContactEvaluationReport
where
    f32: From<B::FloatElem>,
{
    assert!(!seeds.is_empty(), "contact evaluation requires seeds");
    assert!(combat.enabled, "contact evaluation requires its curriculum");
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
                    let mut env = BlobEnv::new(env_config, reward.clone(), seed);
                    env.enable_telemetry(TelemetryConfig {
                        enabled: true,
                        state_sample_interval_steps: u64::MAX,
                        max_state_samples_per_episode: 2,
                        episode_log_stride: 0,
                    });
                    let mut observations = env.get_policy_observations();
                    let mut episode_attacks = 0u64;
                    let mut episode_damage = 0u128;
                    loop {
                        let actions = greedy_policy_choices(model, &observations, device);
                        let result = env.step_with_policy_memory(&actions);
                        let telemetry = result
                            .telemetry
                            .as_ref()
                            .expect("contact evaluation telemetry is enabled");
                        let attack = &telemetry.training.actions.attack;
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
                            break;
                        }
                        observations = result.policy_observations;
                    }
                    metrics.attacking_episodes += usize::from(episode_attacks > 0);
                    metrics.damaging_episodes += usize::from(episode_damage > 0);
                }
                variants.push(finalize_variant(metrics));
            }
        }
    }

    report_from_variants(
        ruleset_hash,
        seeds.to_vec(),
        combat.contact_evaluation_sim_time_limit_quanta,
        combat.skirmish_evaluation_sim_time_limit_quanta,
        variants,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_rates_are_recomputed_from_raw_counts() {
        let variant = finalize_variant(ContactVariantMetrics {
            stage: FeedingCurriculumStage::Contact,
            cells_per_team: 1,
            initial_energy: 100,
            opponent: OpponentProfile::Defensive,
            episodes: 4,
            wins: 1,
            losses: 1,
            timeouts: 2,
            safety_aborts: 0,
            attacking_episodes: 3,
            damaging_episodes: 2,
            attacks_committed: 10,
            attacks_succeeded: 4,
            attacks_frustrated: 5,
            attacks_interrupted: 1,
            damage_dealt: 20,
            kills: 1,
            attacking_episode_rate: 0.0,
            damaging_episode_rate: 0.0,
            attack_success_rate: 0.0,
        });
        let report = report_from_variants(
            "rules".into(),
            vec![1, 2, 3, 4],
            32_768,
            65_536,
            vec![variant],
        );
        assert_eq!(report.attacking_episode_rate, 0.75);
        assert_eq!(report.damaging_episode_rate, 0.5);
        assert_eq!(report.attack_success_rate, 0.4);
        assert_eq!(report.damage_dealt, 20);
    }

    #[test]
    fn validation_rejects_mutated_aggregate() {
        let combat = CombatCurriculumConfig {
            contact_initial_energies: vec![100],
            contact_opponents: vec![OpponentProfile::Defensive],
            ..Default::default()
        };
        let variant = finalize_variant(ContactVariantMetrics {
            stage: FeedingCurriculumStage::Contact,
            cells_per_team: 1,
            initial_energy: 100,
            opponent: OpponentProfile::Defensive,
            episodes: 1,
            wins: 0,
            losses: 0,
            timeouts: 1,
            safety_aborts: 0,
            attacking_episodes: 1,
            damaging_episodes: 1,
            attacks_committed: 1,
            attacks_succeeded: 1,
            attacks_frustrated: 0,
            attacks_interrupted: 0,
            damage_dealt: 5,
            kills: 0,
            attacking_episode_rate: 0.0,
            damaging_episode_rate: 0.0,
            attack_success_rate: 0.0,
        });
        let mut skirmish = variant.clone();
        skirmish.stage = FeedingCurriculumStage::Skirmish;
        skirmish.cells_per_team = combat.skirmish_cells_per_team;
        let mut report = report_from_variants(
            "rules".into(),
            vec![7],
            combat.contact_evaluation_sim_time_limit_quanta,
            combat.skirmish_evaluation_sim_time_limit_quanta,
            vec![variant, skirmish],
        );
        report.damage_dealt = 6;
        assert!(report.validate_against("rules", &[7], &combat).is_err());

        let mut horizon_mutation = report_from_variants(
            "rules".into(),
            vec![7],
            combat.contact_evaluation_sim_time_limit_quanta,
            combat.skirmish_evaluation_sim_time_limit_quanta,
            report.variants.clone(),
        );
        horizon_mutation.contact_sim_time_limit_quanta += 1;
        assert!(horizon_mutation
            .validate_against("rules", &[7], &combat)
            .is_err());
    }

    #[test]
    fn activity_requires_resolver_confirmed_damage_in_contact_and_skirmish() {
        let contact = finalize_variant(ContactVariantMetrics {
            stage: FeedingCurriculumStage::Contact,
            cells_per_team: 1,
            initial_energy: 100,
            opponent: OpponentProfile::Aggressive,
            episodes: 1,
            wins: 0,
            losses: 0,
            timeouts: 1,
            safety_aborts: 0,
            attacking_episodes: 1,
            damaging_episodes: 1,
            attacks_committed: 1,
            attacks_succeeded: 1,
            attacks_frustrated: 0,
            attacks_interrupted: 0,
            damage_dealt: 5,
            kills: 0,
            attacking_episode_rate: 0.0,
            damaging_episode_rate: 0.0,
            attack_success_rate: 0.0,
        });
        let mut skirmish = contact.clone();
        skirmish.stage = FeedingCurriculumStage::Skirmish;
        skirmish.cells_per_team = 4;
        skirmish.attacks_succeeded = 0;
        skirmish.damage_dealt = 0;
        skirmish.damaging_episodes = 0;
        skirmish = finalize_variant(skirmish);
        let report = report_from_variants(
            "rules".into(),
            vec![7],
            32_768,
            65_536,
            vec![contact, skirmish.clone()],
        );
        assert!(!report.is_active());

        skirmish.attacks_succeeded = 1;
        skirmish.damage_dealt = 5;
        skirmish.damaging_episodes = 1;
        let report = report_from_variants(
            "rules".into(),
            vec![7],
            32_768,
            65_536,
            vec![report.variants[0].clone(), finalize_variant(skirmish)],
        );
        assert!(report.is_active());

        let mut combat = CombatCurriculumConfig {
            min_skirmish_kills_for_promotion: 1,
            ..Default::default()
        };
        assert!(!report.meets_promotion_thresholds(&combat));
        let mut lethal = report.clone();
        lethal
            .variants
            .iter_mut()
            .find(|variant| variant.stage == FeedingCurriculumStage::Skirmish)
            .unwrap()
            .kills = 1;
        lethal.kills = 1;
        assert!(lethal.meets_promotion_thresholds(&combat));
        combat.min_contact_kills_for_promotion = 1;
        assert!(!lethal.meets_promotion_thresholds(&combat));
    }
}
