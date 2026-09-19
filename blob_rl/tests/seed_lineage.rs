use blob_rl::behavior_cloning::seed_ledger::InheritedSeedLedger;
use blob_rl::behavior_cloning::{
    behavior_clone_artifact_sha256, behavior_clone_from_model, publish_behavior_clone,
    verify_behavior_clone_metadata, BehaviorCloningArtifact, BehaviorCloningConfig,
};
use blob_rl::config::{ModelConfig, TrainingConfig};
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::demonstration::{
    generate_demonstrations, load_demonstrations, publish_demonstrations, DemonstrationOptions,
    LoadedDemonstrations,
};
use blob_rl::policy_artifact::load_behavior_clone;
use burn::backend::{Autodiff, NdArray};
use std::path::Path;

type B = Autodiff<NdArray<f32>>;

fn corpus(path: &Path, seeds: Vec<u64>) -> LoadedDemonstrations {
    let mut config = TrainingConfig::default();
    config.env.world_size = 4;
    config.env.cells_per_team = 1;
    config.env.max_episode_len = 1024;
    config.env.victory.sim_time_limit_quanta = 16384;
    config.env.num_scattered_energy = 2;
    config.env.num_plants = 1;
    let (manifest, payload) = generate_demonstrations(
        &config,
        "a".repeat(64),
        &DemonstrationOptions {
            teacher: MaintainedMindProfile::Simple,
            max_samples: seeds.len() * 4,
            seeds,
        },
    )
    .unwrap();
    publish_demonstrations(path, &manifest, &payload).unwrap();
    load_demonstrations(path).unwrap()
}

fn train(
    output: &Path,
    parent: Option<&Path>,
    model: &ModelConfig,
    config: &BehaviorCloningConfig,
    datasets: &[LoadedDemonstrations],
) -> BehaviorCloningArtifact {
    let device = Default::default();
    let initial = parent.map(|path| {
        load_behavior_clone::<B>(
            path,
            config.initial_artifact_sha256.as_deref().unwrap(),
            model,
            &device,
        )
        .unwrap()
    });
    let (net, metrics) =
        behavior_clone_from_model::<B>(datasets, model, config, initial, device).unwrap();
    publish_behavior_clone(output, &net, model, config, datasets, &metrics).unwrap();
    verify_behavior_clone_metadata(
        output,
        &behavior_clone_artifact_sha256(output).unwrap(),
        model,
    )
    .unwrap()
}

fn child_config(
    parent: &Path,
    model: &ModelConfig,
    config: &BehaviorCloningConfig,
) -> BehaviorCloningConfig {
    let hash = behavior_clone_artifact_sha256(parent).unwrap();
    BehaviorCloningConfig {
        initial_artifact_sha256: Some(hash.clone()),
        inherited_seed_ledger: Some(
            InheritedSeedLedger::from_artifact(parent, &hash, model).unwrap(),
        ),
        ..config.clone()
    }
}

#[test]
fn lineage_roles_survive_subset_corpora_new_seeds_and_changed_split_settings() {
    let tmp = tempfile::tempdir().unwrap();
    let model = ModelConfig {
        hidden1: 8,
        hidden2: 8,
        recurrent_size: 4,
    };
    let config = BehaviorCloningConfig {
        epochs: 1,
        minibatch_size: 4,
        recurrent_unroll_steps: 1,
        validation_fraction: 0.5,
        confirmation_seeds: vec![900, 901],
        ..Default::default()
    };
    let root_data = [corpus(&tmp.path().join("root-data"), (1..=8).collect())];
    let root = tmp.path().join("root");
    let first = train(&root, None, &model, &config, &root_data);
    let original = first.seed_ledger.as_ref().unwrap();
    assert_eq!(original.training.len(), 4);
    assert_eq!(original.validation.len(), 4);
    assert_eq!(original.confirmation, [900, 901].into());

    // Omit old seeds, add new ones, and change the requested ratio and ranking.
    let subset: Vec<_> = original
        .training
        .iter()
        .take(2)
        .chain(original.validation.iter().take(2))
        .copied()
        .chain([9, 10])
        .collect();
    let child_data = [corpus(&tmp.path().join("child-data"), subset)];
    let mut child = child_config(&root, &model, &config);
    child.seed = 123;
    child.validation_fraction = 0.25;
    child.confirmation_seeds = vec![]; // Inherited reservations cannot be removed.
    let second_path = tmp.path().join("child");
    let second = train(&second_path, Some(&root), &model, &child, &child_data);
    let second_ledger = second.seed_ledger.as_ref().unwrap();
    assert!(original.training.is_subset(&second_ledger.training));
    assert!(original.validation.is_subset(&second_ledger.validation));
    assert_eq!(second_ledger.confirmation, original.confirmation);

    let grandchild_data = [corpus(
        &tmp.path().join("grandchild-data"),
        (1..=12).collect(),
    )];
    let grandchild = child_config(&second_path, &model, &config);
    let third = train(
        &tmp.path().join("grandchild"),
        Some(&second_path),
        &model,
        &grandchild,
        &grandchild_data,
    );
    let third_ledger = third.seed_ledger.unwrap();
    assert!(second_ledger.training.is_subset(&third_ledger.training));
    assert!(second_ledger.validation.is_subset(&third_ledger.validation));
    assert!(third_ledger.training.is_disjoint(&third_ledger.validation));

    let mut wrong_parent = child.clone();
    wrong_parent.initial_artifact_sha256 = Some("a".repeat(64));
    assert!(wrong_parent.validate().is_err());
    let mut retroactive_reservation = child.clone();
    retroactive_reservation
        .confirmation_seeds
        .push(*original.training.first().unwrap());
    assert!(retroactive_reservation.validate().is_err());
    let mut disabled = child.clone();
    disabled.validation_fraction = 0.0;
    let initial = load_behavior_clone::<B>(
        &root,
        child.initial_artifact_sha256.as_deref().unwrap(),
        &model,
        &Default::default(),
    )
    .unwrap();
    assert!(behavior_clone_from_model::<B>(
        &root_data,
        &model,
        &disabled,
        Some(initial.clone()),
        Default::default()
    )
    .err()
    .unwrap()
    .contains("validation is disabled"));
    let mut reserved = root_data.clone();
    reserved[0].payload.samples[0].source_seed = 900;
    reserved[0].payload.samples[0].exact_round_trip = false;
    let mut filtered = child.clone();
    filtered.exact_round_trip_only = true;
    assert!(behavior_clone_from_model::<B>(
        &reserved,
        &model,
        &filtered,
        Some(initial),
        Default::default()
    )
    .err()
    .unwrap()
    .contains("confirmation seeds"));

    // Repaired metadata hashes must not hide an inconsistent cumulative ledger.
    let metadata_path = root.join("behavior-cloning.json");
    let bytes = std::fs::read(&metadata_path).unwrap();
    let mut tampered: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    tampered["seed_ledger"]["training"] = serde_json::json!([]);
    std::fs::write(&metadata_path, serde_json::to_vec(&tampered).unwrap()).unwrap();
    assert!(verify_behavior_clone_metadata(
        &root,
        &behavior_clone_artifact_sha256(&root).unwrap(),
        &model
    )
    .is_err());
    std::fs::write(&metadata_path, bytes).unwrap();
}

#[test]
fn legacy_audit_requires_complete_hash_bound_history_and_exposes_old_holdouts() {
    let tmp = tempfile::tempdir().unwrap();
    let model = ModelConfig {
        hidden1: 8,
        hidden2: 8,
        recurrent_size: 4,
    };
    let config = BehaviorCloningConfig {
        epochs: 1,
        minibatch_size: 4,
        recurrent_unroll_steps: 1,
        validation_fraction: 0.5,
        ..Default::default()
    };
    let corpus_path = tmp.path().join("corpus");
    let datasets = [corpus(&corpus_path, (1..=8).collect())];
    let root = tmp.path().join("root");
    let artifact = train(&root, None, &model, &config, &datasets);
    let mut legacy = serde_json::to_value(&artifact).unwrap();
    legacy["schema_version"] = 35.into();
    legacy.as_object_mut().unwrap().remove("seed_ledger");
    legacy["config"]
        .as_object_mut()
        .unwrap()
        .remove("inherited_seed_ledger");
    legacy["config"]
        .as_object_mut()
        .unwrap()
        .remove("confirmation_seeds");
    for partition in legacy["metrics"]["partitions"].as_array_mut().unwrap() {
        partition.as_object_mut().unwrap().remove("training_seeds");
    }
    std::fs::write(
        root.join("behavior-cloning.json"),
        serde_json::to_vec(&legacy).unwrap(),
    )
    .unwrap();
    let root_hash = behavior_clone_artifact_sha256(&root).unwrap();
    load_behavior_clone::<B>(&root, &root_hash, &model, &Default::default()).unwrap();
    assert!(InheritedSeedLedger::from_artifact(&root, &root_hash, &model).is_err());
    assert!(InheritedSeedLedger::audit_legacy(&root, &root_hash, &model, &[], &[]).is_err());

    let leaf = tmp.path().join("legacy-leaf");
    std::fs::create_dir(&leaf).unwrap();
    std::fs::copy(root.join("model.mpk"), leaf.join("model.mpk")).unwrap();
    legacy["config"]["initial_artifact_sha256"] = root_hash.clone().into();
    std::fs::write(
        leaf.join("behavior-cloning.json"),
        serde_json::to_vec(&legacy).unwrap(),
    )
    .unwrap();
    let leaf_hash = behavior_clone_artifact_sha256(&leaf).unwrap();
    assert!(InheritedSeedLedger::audit_legacy(
        &leaf,
        &leaf_hash,
        &model,
        &[],
        std::slice::from_ref(&corpus_path)
    )
    .err()
    .unwrap()
    .contains("missing historical ancestor"));
    let audited =
        InheritedSeedLedger::audit_legacy(&leaf, &leaf_hash, &model, &[root], &[corpus_path])
            .unwrap();
    let next = BehaviorCloningConfig {
        initial_artifact_sha256: Some(leaf_hash),
        inherited_seed_ledger: Some(audited),
        ..config
    };
    let next_data = [corpus(&tmp.path().join("new-corpus"), (1..=16).collect())];
    let bridge = train(
        &tmp.path().join("bridge"),
        Some(&leaf),
        &model,
        &next,
        &next_data,
    );
    let ledger = bridge.seed_ledger.unwrap();
    assert!((1..=8).all(|seed| ledger.training.contains(&seed)));
    assert!((9..=16).all(|seed| ledger.validation.contains(&seed)));
}
