//! Assign one seed role across every demonstration corpus.

use super::BehaviorCloningConfig;
use crate::demonstration::{DemonstrationSample, LoadedDemonstrations};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};

pub(super) struct DatasetPartition<'a> {
    pub(super) manifest_sha256: String,
    pub(super) training: Vec<&'a DemonstrationSample>,
    pub(super) validation: Vec<&'a DemonstrationSample>,
    pub(super) held_out_seeds: Vec<u64>,
}

fn seed_rank(split_seed: u64, source_seed: u64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"blob-behavior-cloning-validation-seed-v1");
    hasher.update(split_seed.to_le_bytes());
    hasher.update(source_seed.to_le_bytes());
    hasher.finalize().into()
}

pub(super) fn partition_datasets<'a>(
    datasets: &'a [LoadedDemonstrations],
    config: &BehaviorCloningConfig,
) -> Result<Vec<DatasetPartition<'a>>, String> {
    let inherited = super::seed_ledger::inherited_roles(config)?;
    if datasets.iter().any(|dataset| {
        dataset
            .manifest
            .seeds
            .iter()
            .any(|seed| inherited.confirmation.contains(seed))
            || dataset
                .payload
                .samples
                .iter()
                .any(|sample| inherited.confirmation.contains(&sample.source_seed))
    }) {
        return Err("reserved confirmation seeds cannot appear in demonstration datasets, including filtered rows".into());
    }
    // A source seed has one role across the entire corpus, even when datasets
    // have different seed sets or exact-round-trip filtering removes rows.
    let mut ranked = datasets
        .iter()
        .flat_map(|dataset| &dataset.payload.samples)
        .filter(|sample| !config.exact_round_trip_only || sample.exact_round_trip)
        .map(|sample| sample.source_seed)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let held_out_set = if config.validation_fraction == 0.0 {
        if ranked
            .iter()
            .any(|seed| inherited.validation.contains(seed))
        {
            return Err(
                "validation is disabled but the corpus contains inherited validation seeds".into(),
            );
        }
        HashSet::new()
    } else {
        if ranked.len() < 2 {
            return Err(
                "the corpus needs at least two eligible source seeds for held-out validation"
                    .into(),
            );
        }
        let desired = ((ranked.len() as f64 * config.validation_fraction).round() as usize)
            .clamp(1, ranked.len() - 1);
        let mut held_out: HashSet<_> = ranked
            .iter()
            .copied()
            .filter(|seed| inherited.validation.contains(seed))
            .collect();
        ranked.retain(|seed| {
            !inherited.training.contains(seed) && !inherited.validation.contains(seed)
        });
        ranked.sort_unstable_by_key(|seed| seed_rank(config.seed, *seed));
        let count = desired.saturating_sub(held_out.len()).min(ranked.len());
        held_out.extend(&ranked[..count]);
        held_out
    };
    let mut identities = HashSet::with_capacity(datasets.len());
    let mut partitions = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        if !identities.insert(&dataset.manifest_sha256) {
            return Err(format!(
                "duplicate demonstration manifest {}",
                dataset.manifest_sha256
            ));
        }
        let eligible = dataset
            .payload
            .samples
            .iter()
            .filter(|sample| !config.exact_round_trip_only || sample.exact_round_trip)
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            return Err(format!(
                "behavior-cloning filters removed every sample from dataset {}",
                dataset.manifest_sha256
            ));
        }
        let source_seeds = eligible
            .iter()
            .map(|sample| sample.source_seed)
            .collect::<BTreeSet<_>>();
        let held_out_seeds = source_seeds
            .into_iter()
            .filter(|seed| held_out_set.contains(seed))
            .collect::<Vec<_>>();
        let (validation, training): (Vec<_>, Vec<_>) = eligible
            .into_iter()
            .partition(|sample| held_out_set.contains(&sample.source_seed));
        if training.is_empty() || (config.validation_fraction > 0.0 && validation.is_empty()) {
            return Err(format!(
                "dataset {} produced an empty training or validation partition",
                dataset.manifest_sha256
            ));
        }
        partitions.push(DatasetPartition {
            manifest_sha256: dataset.manifest_sha256.clone(),
            training,
            validation,
            held_out_seeds,
        });
    }
    Ok(partitions)
}
