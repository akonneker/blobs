//! Shared policy admission and behavior-preserving model-layout migration.
//!
//! Legacy artifacts are imported under the current execution contract; their
//! old ecological qualification must not be reused as a new qualification.

use crate::behavior_cloning::verify_behavior_clone_artifact_with_schema;
use crate::config::ModelConfig;
use crate::model::{PolicyValueNet, PolicyValueNetConfig};
use burn::prelude::*;
use burn::record::CompactRecorder;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Semantic identity of the code interpreting saved policy parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PolicyExecutionIdentity {
    pub mind_abi_sha256: String,
    pub observation_encoding: String,
    pub action_encoding: String,
    pub private_memory_encoding: String,
    pub expert_routing: String,
}

impl PolicyExecutionIdentity {
    pub fn current() -> Self {
        Self {
            mind_abi_sha256: blob_interface::abi::reference_mind_abi_hash()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
            observation_encoding: "anonymous-slots33-header38-random32-v1".into(),
            action_encoding: "factored-abi8-v1".into(),
            private_memory_encoding: "BRM1-i16-v1".into(),
            expert_routing: "threat-plant-capacity-loose-neighbor-v2".into(),
        }
    }
}

/// Load every supported cloning schema through the same migration path.
pub fn load_behavior_clone<B: Backend>(
    directory: &Path,
    artifact_sha256: &str,
    model: &ModelConfig,
    device: &B::Device,
) -> Result<PolicyValueNet<B>, String> {
    let (model_path, schema_version) =
        verify_behavior_clone_artifact_with_schema(directory, artifact_sha256, model)?;
    let config = PolicyValueNetConfig {
        hidden1: model.hidden1,
        hidden2: model.hidden2,
        recurrent_size: model.recurrent_size,
    };
    if schema_version < 25 {
        let legacy = config
            .init_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load legacy behavior clone: {error}"))?;
        Ok(config.migrate_legacy(legacy, device))
    } else if schema_version < 28 {
        let legacy = config
            .init_foraging_adapter_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load foraging-adapter behavior clone: {error}"))?;
        Ok(config.migrate_foraging_adapter(legacy, device))
    } else if schema_version < 30 {
        let legacy = config
            .init_header_context_adapter_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load header-context-adapter clone: {error}"))?;
        Ok(config.migrate_header_context_adapter(legacy, device))
    } else if schema_version < 31 {
        Err("experimental schema-30 slot-only clones are not migratable; retrain the unpromoted treatment from its verified schema-28/29 parent".into())
    } else if schema_version < 34 {
        let legacy = config
            .init_slot_context_adapter_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load slot-context-adapter clone: {error}"))?;
        Ok(config.migrate_slot_context_adapter(legacy, device))
    } else if schema_version < 37 {
        let legacy = config
            .init_pre_target_residual_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load pre-target-residual clone: {error}"))?;
        Ok(config.migrate_pre_target_residual(legacy, device))
    } else {
        config
            .init::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load behavior clone: {error}"))
    }
}
