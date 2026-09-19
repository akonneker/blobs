//! Cumulative host-only seed roles. These never enter a Mind observation.

use super::{BehaviorCloningConfig, BehaviorCloningDatasetPartition};
use crate::config::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SeedLedger {
    pub training: BTreeSet<u64>,
    pub validation: BTreeSet<u64>,
    pub confirmation: BTreeSet<u64>,
}

impl SeedLedger {
    pub fn validate(&self) -> Result<(), String> {
        if !self.training.is_disjoint(&self.validation)
            || !self.training.is_disjoint(&self.confirmation)
            || !self.validation.is_disjoint(&self.confirmation)
        {
            return Err(
                "a lineage seed cannot have multiple training, validation, or confirmation roles"
                    .into(),
            );
        }
        if self.training.len() + self.validation.len() + self.confirmation.len() > 65_536 {
            return Err("seed ledger exceeds 65536 distinct seeds".into());
        }
        Ok(())
    }
}

/// A parent's cumulative ledger, obtained from its hash-verified artifact.
/// Private fields prevent accidental construction from an unrelated seed list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InheritedSeedLedger {
    artifact_sha256: String,
    ledger: SeedLedger,
    #[serde(default)]
    legacy_audit: Option<LegacySeedAudit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct LegacySeedAudit {
    artifacts: BTreeSet<String>,
    corpus_manifests: BTreeSet<String>,
}

impl InheritedSeedLedger {
    /// Read verified roles without allowing callers to construct or mutate lineage.
    pub fn ledger(&self) -> &SeedLedger {
        &self.ledger
    }

    pub fn from_artifact(
        directory: &Path,
        artifact_sha256: &str,
        model: &ModelConfig,
    ) -> Result<Self, String> {
        let artifact = super::verify_behavior_clone_metadata(directory, artifact_sha256, model)?;
        let ledger = artifact.seed_ledger.ok_or_else(|| {
            "parent artifact has no cumulative seed ledger; historical lineage must be audited before further training (evaluation imports remain supported)".to_string()
        })?;
        // Older metadata did not enforce the ledger contract, even if a field was added.
        if artifact.schema_version < 36 {
            return Err("training parents require seed-ledger schema 36 or newer".into());
        }
        Ok(Self {
            artifact_sha256: artifact_sha256.into(),
            ledger,
            legacy_audit: None,
        })
    }

    /// Audit a pre-ledger chain using every hash-bound ancestor and corpus.
    /// All historical corpus seeds are conservatively treated as exposed: old
    /// validation results may have influenced model selection or later corpora.
    pub fn audit_legacy(
        directory: &Path,
        artifact_sha256: &str,
        model: &ModelConfig,
        ancestors: &[std::path::PathBuf],
        corpus_directories: &[std::path::PathBuf],
    ) -> Result<Self, String> {
        use std::collections::{HashMap, HashSet};
        // Verify one corpus at a time; retain only its manifest, not historical
        // observation tensors from every generation at once.
        let mut corpora = HashMap::new();
        for path in corpus_directories {
            let data = crate::demonstration::load_demonstrations(path)?;
            corpora.insert(data.manifest_sha256, data.manifest);
        }
        let mut artifacts = HashMap::new();
        artifacts.insert(artifact_sha256.to_owned(), directory.to_owned());
        for path in ancestors {
            artifacts.insert(super::behavior_clone_artifact_sha256(path)?, path.clone());
        }
        let mut next = Some(artifact_sha256.to_owned());
        let mut seen = HashSet::new();
        let mut ledger = SeedLedger::default();
        let mut used_corpora = BTreeSet::new();
        while let Some(hash) = next {
            if !seen.insert(hash.clone()) || seen.len() > 256 {
                return Err("historical lineage is cyclic or exceeds 256 ancestors".into());
            }
            let path = artifacts.get(&hash).ok_or_else(|| {
                format!("missing historical ancestor {hash}; supply --lineage-artifact")
            })?;
            let artifact = super::verify_behavior_clone_metadata(path, &hash, model)?;
            if artifact.schema_version >= 36 {
                let known = artifact
                    .seed_ledger
                    .expect("verified current artifact has a ledger");
                ledger.training.extend(known.training);
                ledger.validation.extend(known.validation);
                ledger.confirmation.extend(known.confirmation);
                ledger.validate()?;
                break;
            }
            if artifact.datasets.is_empty() {
                return Err("historical artifact has no corpus identities to audit".into());
            }
            for identity in &artifact.datasets {
                let data = corpora.get(&identity.manifest_sha256).ok_or_else(|| {
                    format!(
                        "missing historical corpus {}; supply --lineage-dataset",
                        identity.manifest_sha256
                    )
                })?;
                if data.payload_sha256 != identity.payload_sha256 {
                    return Err(
                        "historical corpus payload identity does not match its policy".into(),
                    );
                }
                used_corpora.insert(identity.manifest_sha256.clone());
                ledger.training.extend(&data.seeds);
            }
            ledger.validate()?;
            next = artifact.config.initial_artifact_sha256;
        }
        Ok(Self {
            artifact_sha256: artifact_sha256.into(),
            ledger,
            legacy_audit: Some(LegacySeedAudit {
                artifacts: seen.into_iter().collect(),
                corpus_manifests: used_corpora,
            }),
        })
    }
}

pub(super) fn inherited_roles(config: &BehaviorCloningConfig) -> Result<SeedLedger, String> {
    let mut ledger = match (&config.initial_artifact_sha256, &config.inherited_seed_ledger) {
        (None, None) => SeedLedger::default(),
        (Some(hash), Some(parent)) if *hash == parent.artifact_sha256 => parent.ledger.clone(),
        _ => return Err("initial artifact and its verified inherited seed ledger must be supplied together and match".into()),
    };
    ledger.confirmation.extend(&config.confirmation_seeds);
    ledger.validate()?;
    Ok(ledger)
}

pub(super) fn ledger_from_metrics(
    config: &BehaviorCloningConfig,
    partitions: &[BehaviorCloningDatasetPartition],
) -> Result<SeedLedger, String> {
    let mut ledger = inherited_roles(config)?;
    for partition in partitions {
        if partition.training_seeds.is_empty()
            || partition.training_samples == 0
            || partition.training_seeds.len() > partition.training_samples
            || partition.held_out_seeds.len() > partition.validation_samples
            || partition
                .training_seeds
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || partition
                .held_out_seeds
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || (config.validation_fraction == 0.0 && partition.validation_samples != 0)
            || (partition.validation_samples == 0) != partition.held_out_seeds.is_empty()
            || (config.validation_fraction > 0.0 && partition.held_out_seeds.is_empty())
        {
            return Err(
                "seed ledger requires complete training and validation partition identities".into(),
            );
        }
        ledger.training.extend(&partition.training_seeds);
        ledger.validation.extend(&partition.held_out_seeds);
    }
    ledger.validate()?;
    Ok(ledger)
}
