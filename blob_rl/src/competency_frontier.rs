//! Bounded, integrity-checked archive of evaluated ecology/combat trade-offs.
//!
//! The archive is host-side training evidence. It is never projected into a
//! Mind observation, shared between cells, or consulted by canonical physics.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use serde::{Deserialize, Serialize};

use crate::artifact::verify_checkpoint_metadata;
use crate::config::{CombatCurriculumConfig, FeedingCurriculumStage};
use crate::contact_evaluation::ContactEvaluationReport;
use crate::evaluation::EvaluationMetrics;
use crate::feeding_curriculum::FeedingPromotionReport;

pub const COMPETENCY_FRONTIER_SCHEMA_VERSION: u32 = 1;
pub const MAX_COMPETENCY_FRONTIER_ENTRIES: usize = 32;
const MAX_FRONTIER_BYTES: u64 = 1024 * 1024;
static FRONTIER_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompetencyMetrics {
    pub configured_feeding_passed: bool,
    pub retention_feeding_passed: bool,
    pub on_food_survival_rate: f64,
    pub adjacent_food_survival_rate: f64,
    pub on_food_intake_per_initial_cell: f64,
    pub adjacent_food_intake_per_initial_cell: f64,
    pub combat_passed: bool,
    pub contact_damage: u128,
    pub contact_kills: u64,
    pub skirmish_damage: u128,
    pub skirmish_kills: u64,
    pub fixed_worst_case_win_rate: f64,
    pub fixed_win_rate: f64,
    pub fixed_average_reward: f64,
}

impl CompetencyMetrics {
    pub fn from_reports(
        evaluation: &EvaluationMetrics,
        feeding: &FeedingPromotionReport,
        retention: Option<&FeedingPromotionReport>,
        contact: &ContactEvaluationReport,
        combat: &CombatCurriculumConfig,
    ) -> Option<Self> {
        let retention = retention.unwrap_or(feeding);
        let on_food = retention
            .stages
            .iter()
            .find(|stage| stage.stage == FeedingCurriculumStage::OnFood)?;
        let adjacent = retention
            .stages
            .iter()
            .find(|stage| stage.stage == FeedingCurriculumStage::AdjacentFood)?;
        let stage_sum =
            |stage, select: fn(&crate::contact_evaluation::ContactVariantMetrics) -> u128| {
                contact
                    .variants
                    .iter()
                    .filter(|variant| variant.stage == stage)
                    .map(select)
                    .sum()
            };
        let contact_damage = stage_sum(FeedingCurriculumStage::Contact, |variant| {
            variant.damage_dealt
        });
        let skirmish_damage = stage_sum(FeedingCurriculumStage::Skirmish, |variant| {
            variant.damage_dealt
        });
        let metrics = Self {
            configured_feeding_passed: feeding.passed,
            retention_feeding_passed: retention.passed,
            on_food_survival_rate: on_food.survival_rate,
            adjacent_food_survival_rate: adjacent.survival_rate,
            on_food_intake_per_initial_cell: on_food.consumed_energy_per_initial_cell,
            adjacent_food_intake_per_initial_cell: adjacent.consumed_energy_per_initial_cell,
            combat_passed: contact.meets_promotion_thresholds(combat),
            contact_damage,
            contact_kills: contact.kills_for_stage(FeedingCurriculumStage::Contact),
            skirmish_damage,
            skirmish_kills: contact.kills_for_stage(FeedingCurriculumStage::Skirmish),
            fixed_worst_case_win_rate: evaluation.worst_case_win_rate,
            fixed_win_rate: evaluation.win_rate,
            fixed_average_reward: evaluation.average_reward,
        };
        metrics.is_valid().then_some(metrics)
    }

    pub fn joint_qualified(&self) -> bool {
        self.configured_feeding_passed && self.retention_feeding_passed && self.combat_passed
    }

    fn is_valid(&self) -> bool {
        [
            self.on_food_survival_rate,
            self.adjacent_food_survival_rate,
            self.on_food_intake_per_initial_cell,
            self.adjacent_food_intake_per_initial_cell,
            self.fixed_worst_case_win_rate,
            self.fixed_win_rate,
            self.fixed_average_reward,
        ]
        .iter()
        .all(|value| value.is_finite())
            && (0.0..=1.0).contains(&self.on_food_survival_rate)
            && (0.0..=1.0).contains(&self.adjacent_food_survival_rate)
            && self.on_food_intake_per_initial_cell >= 0.0
            && self.adjacent_food_intake_per_initial_cell >= 0.0
            && (0.0..=1.0).contains(&self.fixed_worst_case_win_rate)
            && (0.0..=1.0).contains(&self.fixed_win_rate)
    }

    fn dominates_or_equals(&self, other: &Self) -> bool {
        u8::from(self.configured_feeding_passed) >= u8::from(other.configured_feeding_passed)
            && u8::from(self.retention_feeding_passed) >= u8::from(other.retention_feeding_passed)
            && self.on_food_survival_rate >= other.on_food_survival_rate
            && self.adjacent_food_survival_rate >= other.adjacent_food_survival_rate
            && self.on_food_intake_per_initial_cell >= other.on_food_intake_per_initial_cell
            && self.adjacent_food_intake_per_initial_cell
                >= other.adjacent_food_intake_per_initial_cell
            && u8::from(self.combat_passed) >= u8::from(other.combat_passed)
            && self.contact_damage >= other.contact_damage
            && self.contact_kills >= other.contact_kills
            && self.skirmish_damage >= other.skirmish_damage
            && self.skirmish_kills >= other.skirmish_kills
            && self.fixed_worst_case_win_rate >= other.fixed_worst_case_win_rate
            && self.fixed_win_rate >= other.fixed_win_rate
            && self.fixed_average_reward >= other.fixed_average_reward
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompetencyFrontierEntry {
    pub checkpoint_directory: String,
    pub checkpoint: String,
    pub update: usize,
    pub actions: u64,
    pub metrics: CompetencyMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CompetencyCheckpointIdentity {
    pub checkpoint_directory: String,
    pub checkpoint: String,
    pub update: usize,
    pub actions: u64,
}

impl CompetencyFrontierEntry {
    pub fn identity(&self) -> CompetencyCheckpointIdentity {
        CompetencyCheckpointIdentity {
            checkpoint_directory: self.checkpoint_directory.clone(),
            checkpoint: self.checkpoint.clone(),
            update: self.update,
            actions: self.actions,
        }
    }
}

/// Exact host-side teachers active after an update boundary. A missing field
/// means no checkpoint has yet passed that competency's independent gate.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpecialistTeacherSelection {
    pub ecology: Option<CompetencyCheckpointIdentity>,
    pub combat: Option<CompetencyCheckpointIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompetencyFrontier {
    pub schema_version: u32,
    pub entries: Vec<CompetencyFrontierEntry>,
}

impl Default for CompetencyFrontier {
    fn default() -> Self {
        Self {
            schema_version: COMPETENCY_FRONTIER_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }
}

impl CompetencyFrontier {
    /// Insert one nondominated point. Equal evidence keeps the earlier entry;
    /// the bounded fallback ranks joint qualification, worst ecology, kills,
    /// damage, and fixed reward deterministically.
    pub fn insert(&mut self, candidate: CompetencyFrontierEntry) -> bool {
        if !candidate.metrics.is_valid()
            || self
                .entries
                .iter()
                .any(|entry| entry.metrics.dominates_or_equals(&candidate.metrics))
        {
            return false;
        }
        self.entries
            .retain(|entry| !candidate.metrics.dominates_or_equals(&entry.metrics));
        let candidate_key = (
            candidate.checkpoint_directory.clone(),
            candidate.checkpoint.clone(),
        );
        self.entries.push(candidate);
        if self.entries.len() > MAX_COMPETENCY_FRONTIER_ENTRIES {
            self.entries.sort_by(frontier_priority);
            self.entries.truncate(MAX_COMPETENCY_FRONTIER_ENTRIES);
        }
        self.entries.sort_by(|left, right| {
            (
                left.actions,
                left.update,
                &left.checkpoint_directory,
                &left.checkpoint,
            )
                .cmp(&(
                    right.actions,
                    right.update,
                    &right.checkpoint_directory,
                    &right.checkpoint,
                ))
        });
        self.entries.iter().any(|entry| {
            (
                entry.checkpoint_directory.as_str(),
                entry.checkpoint.as_str(),
            ) == (candidate_key.0.as_str(), candidate_key.1.as_str())
        })
    }

    pub fn specialist_teachers(
        &self,
        combat_precursor_min_skirmish_damage: Option<u64>,
    ) -> SpecialistTeacherSelection {
        let ecology = self
            .entries
            .iter()
            .filter(|entry| {
                entry.metrics.configured_feeding_passed && entry.metrics.retention_feeding_passed
            })
            .max_by(|left, right| ecology_teacher_priority(left, right))
            .map(CompetencyFrontierEntry::identity);
        let qualified_combat = self
            .entries
            .iter()
            .filter(|entry| entry.metrics.combat_passed)
            .max_by(|left, right| combat_teacher_priority(left, right))
            .map(CompetencyFrontierEntry::identity);
        let combat = qualified_combat.or_else(|| {
            let minimum = u128::from(combat_precursor_min_skirmish_damage?);
            self.entries
                .iter()
                .filter(|entry| entry.metrics.skirmish_damage >= minimum)
                .max_by(|left, right| combat_teacher_priority(left, right))
                .map(CompetencyFrontierEntry::identity)
        });
        SpecialistTeacherSelection { ecology, combat }
    }

    pub fn contains_identity(&self, identity: &CompetencyCheckpointIdentity) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.identity() == *identity)
    }
}

fn ecology_teacher_priority(
    left: &CompetencyFrontierEntry,
    right: &CompetencyFrontierEntry,
) -> Ordering {
    let left_worst_survival = left
        .metrics
        .on_food_survival_rate
        .min(left.metrics.adjacent_food_survival_rate);
    let right_worst_survival = right
        .metrics
        .on_food_survival_rate
        .min(right.metrics.adjacent_food_survival_rate);
    let left_worst_intake = left
        .metrics
        .on_food_intake_per_initial_cell
        .min(left.metrics.adjacent_food_intake_per_initial_cell);
    let right_worst_intake = right
        .metrics
        .on_food_intake_per_initial_cell
        .min(right.metrics.adjacent_food_intake_per_initial_cell);
    left_worst_survival
        .total_cmp(&right_worst_survival)
        .then_with(|| left_worst_intake.total_cmp(&right_worst_intake))
        .then_with(|| {
            left.metrics
                .fixed_worst_case_win_rate
                .total_cmp(&right.metrics.fixed_worst_case_win_rate)
        })
        .then_with(|| left.actions.cmp(&right.actions))
        .then_with(|| left.update.cmp(&right.update))
        .then_with(|| left.checkpoint.cmp(&right.checkpoint))
}

fn combat_teacher_priority(
    left: &CompetencyFrontierEntry,
    right: &CompetencyFrontierEntry,
) -> Ordering {
    left.metrics
        .skirmish_kills
        .cmp(&right.metrics.skirmish_kills)
        .then_with(|| left.metrics.contact_kills.cmp(&right.metrics.contact_kills))
        .then_with(|| {
            left.metrics
                .skirmish_damage
                .cmp(&right.metrics.skirmish_damage)
        })
        .then_with(|| {
            left.metrics
                .contact_damage
                .cmp(&right.metrics.contact_damage)
        })
        .then_with(|| {
            left.metrics
                .fixed_worst_case_win_rate
                .total_cmp(&right.metrics.fixed_worst_case_win_rate)
        })
        .then_with(|| left.actions.cmp(&right.actions))
        .then_with(|| left.update.cmp(&right.update))
        .then_with(|| left.checkpoint.cmp(&right.checkpoint))
}

fn frontier_priority(left: &CompetencyFrontierEntry, right: &CompetencyFrontierEntry) -> Ordering {
    let left_worst_ecology = left
        .metrics
        .on_food_survival_rate
        .min(left.metrics.adjacent_food_survival_rate);
    let right_worst_ecology = right
        .metrics
        .on_food_survival_rate
        .min(right.metrics.adjacent_food_survival_rate);
    right
        .metrics
        .joint_qualified()
        .cmp(&left.metrics.joint_qualified())
        .then_with(|| right_worst_ecology.total_cmp(&left_worst_ecology))
        .then_with(|| {
            right
                .metrics
                .skirmish_kills
                .cmp(&left.metrics.skirmish_kills)
        })
        .then_with(|| {
            right
                .metrics
                .skirmish_damage
                .cmp(&left.metrics.skirmish_damage)
        })
        .then_with(|| {
            right
                .metrics
                .contact_damage
                .cmp(&left.metrics.contact_damage)
        })
        .then_with(|| {
            right
                .metrics
                .fixed_average_reward
                .total_cmp(&left.metrics.fixed_average_reward)
        })
        .then_with(|| right.actions.cmp(&left.actions))
        .then_with(|| right.update.cmp(&left.update))
        .then_with(|| left.checkpoint.cmp(&right.checkpoint))
}

pub fn validate_competency_frontier(frontier: &CompetencyFrontier) -> Result<(), String> {
    if frontier.schema_version != COMPETENCY_FRONTIER_SCHEMA_VERSION
        || frontier.entries.len() > MAX_COMPETENCY_FRONTIER_ENTRIES
    {
        return Err("unsupported or oversized competency frontier".into());
    }
    let mut identities = HashSet::new();
    for (index, entry) in frontier.entries.iter().enumerate() {
        if entry.checkpoint_directory.trim().is_empty()
            || entry.checkpoint.trim().is_empty()
            || !entry.metrics.is_valid()
            || !identities.insert((
                entry.checkpoint_directory.as_str(),
                entry.checkpoint.as_str(),
            ))
        {
            return Err("competency frontier contains an invalid or duplicate entry".into());
        }
        if frontier
            .entries
            .iter()
            .enumerate()
            .any(|(other_index, other)| {
                index != other_index && other.metrics.dominates_or_equals(&entry.metrics)
            })
        {
            return Err("competency frontier contains a dominated entry".into());
        }
        let checkpoint = Path::new(&entry.checkpoint_directory).join(&entry.checkpoint);
        let metadata = verify_checkpoint_metadata(&checkpoint)?;
        let metrics = CompetencyMetrics::from_reports(
            metadata
                .evaluation
                .as_ref()
                .ok_or("frontier checkpoint has no fixed evaluation")?,
            metadata
                .feeding_evaluation
                .as_ref()
                .ok_or("frontier checkpoint has no feeding evaluation")?,
            metadata.retention_feeding_evaluation.as_ref(),
            metadata
                .contact_evaluation
                .as_ref()
                .ok_or("frontier checkpoint has no combat evaluation")?,
            &metadata.config.combat_curriculum,
        )
        .ok_or("frontier checkpoint has incomplete competency evidence")?;
        if metadata.update != entry.update
            || metadata.actions != entry.actions
            || metrics != entry.metrics
        {
            return Err("competency frontier entry does not match its immutable checkpoint".into());
        }
    }
    Ok(())
}

pub fn publish_competency_frontier(
    artifact_root: &Path,
    frontier: &CompetencyFrontier,
) -> Result<PathBuf, String> {
    validate_competency_frontier(frontier)?;
    fs::create_dir_all(artifact_root).map_err(|error| {
        format!(
            "failed to create competency-frontier root {}: {error}",
            artifact_root.display()
        )
    })?;
    let path = artifact_root.join("competency-frontier.json");
    let nonce = FRONTIER_TEMP_NONCE.fetch_add(1, AtomicOrdering::Relaxed);
    let temporary = artifact_root.join(format!(
        ".competency-frontier.json.tmp-{}-{nonce}",
        std::process::id()
    ));
    let mut bytes = serde_json::to_vec_pretty(frontier)
        .map_err(|error| format!("failed to encode competency frontier: {error}"))?;
    bytes.push(b'\n');
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_FRONTIER_BYTES {
        return Err("competency frontier exceeds its bounded artifact size".into());
    }
    let result = (|| {
        let mut file = File::create(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        fs::rename(&temporary, &path)
            .map_err(|error| format!("failed to publish {}: {error}", path.display()))?;
        File::open(artifact_root)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", artifact_root.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map(|()| path)
}

pub fn load_competency_frontier(path: &Path) -> Result<CompetencyFrontier, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    if metadata.len() > MAX_FRONTIER_BYTES {
        return Err("competency frontier exceeds its bounded artifact size".into());
    }
    let frontier: CompetencyFrontier = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    validate_competency_frontier(&frontier)?;
    Ok(frontier)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(ecology: f64, kills: u64, damage: u128) -> CompetencyMetrics {
        CompetencyMetrics {
            configured_feeding_passed: ecology >= 0.8,
            retention_feeding_passed: ecology >= 0.8,
            on_food_survival_rate: ecology,
            adjacent_food_survival_rate: ecology,
            on_food_intake_per_initial_cell: 100.0,
            adjacent_food_intake_per_initial_cell: 100.0,
            combat_passed: kills > 0,
            contact_damage: damage,
            contact_kills: 0,
            skirmish_damage: damage,
            skirmish_kills: kills,
            fixed_worst_case_win_rate: 0.0,
            fixed_win_rate: 2.0 / 3.0,
            fixed_average_reward: 100.0,
        }
    }

    fn entry(update: usize, metrics: CompetencyMetrics) -> CompetencyFrontierEntry {
        CompetencyFrontierEntry {
            checkpoint_directory: "/tmp/frontier-test".into(),
            checkpoint: format!("checkpoint-{update:08}"),
            update,
            actions: update as u64 * 10,
            metrics,
        }
    }

    #[test]
    fn frontier_keeps_incomparable_specialists_and_removes_dominated_points() {
        let mut frontier = CompetencyFrontier::default();
        assert!(frontier.insert(entry(1, metrics(0.95, 0, 100))));
        assert!(frontier.insert(entry(2, metrics(0.70, 8, 500))));
        assert!(!frontier.insert(entry(3, metrics(0.60, 0, 50))));
        assert_eq!(frontier.entries.len(), 2);

        assert!(frontier.insert(entry(4, metrics(0.96, 16, 600))));
        assert_eq!(frontier.entries.len(), 1);
        assert_eq!(frontier.entries[0].update, 4);
        assert!(frontier.entries[0].metrics.joint_qualified());
    }

    #[test]
    fn frontier_is_bounded_deterministically() {
        let mut frontier = CompetencyFrontier::default();
        for update in 1..=MAX_COMPETENCY_FRONTIER_ENTRIES + 8 {
            let ecology = 0.5 + update as f64 / 100.0;
            let kills = (MAX_COMPETENCY_FRONTIER_ENTRIES + 8 - update) as u64;
            assert!(frontier.insert(entry(update, metrics(ecology, kills, 100))));
        }
        assert_eq!(frontier.entries.len(), MAX_COMPETENCY_FRONTIER_ENTRIES);
        assert!(frontier.entries.windows(2).all(|pair| {
            (pair[0].actions, pair[0].update) <= (pair[1].actions, pair[1].update)
        }));
    }

    #[test]
    fn specialist_selection_is_qualified_and_stage_local() {
        let mut frontier = CompetencyFrontier::default();
        assert!(frontier.insert(entry(1, metrics(0.95, 0, 100))));
        assert!(frontier.insert(entry(2, metrics(0.70, 8, 500))));

        let teachers = frontier.specialist_teachers(None);
        assert_eq!(
            teachers.ecology.as_ref().map(|teacher| teacher.update),
            Some(1)
        );
        assert_eq!(
            teachers.combat.as_ref().map(|teacher| teacher.update),
            Some(2)
        );
        assert!(frontier.contains_identity(teachers.ecology.as_ref().unwrap()));
        assert!(frontier.contains_identity(teachers.combat.as_ref().unwrap()));
    }

    #[test]
    fn combat_precursor_is_explicit_and_never_displaces_a_qualified_teacher() {
        let mut frontier = CompetencyFrontier::default();
        assert!(frontier.insert(entry(1, metrics(0.95, 0, 400))));

        assert!(frontier.specialist_teachers(None).combat.is_none());
        assert!(frontier.specialist_teachers(Some(500)).combat.is_none());
        assert_eq!(
            frontier
                .specialist_teachers(Some(400))
                .combat
                .as_ref()
                .map(|teacher| teacher.update),
            Some(1)
        );

        assert!(frontier.insert(entry(2, metrics(0.70, 1, 100))));
        assert_eq!(
            frontier
                .specialist_teachers(Some(1))
                .combat
                .as_ref()
                .map(|teacher| teacher.update),
            Some(2)
        );
    }
}
