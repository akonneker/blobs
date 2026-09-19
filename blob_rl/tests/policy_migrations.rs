use blob_rl::behavior_cloning::{
    behavior_clone, behavior_clone_artifact_sha256, publish_behavior_clone,
    verify_behavior_clone_artifact, BehaviorCloningConfig,
};
use blob_rl::config::{ModelConfig, TrainingConfig};
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::demonstration::{generate_demonstrations, DemonstrationOptions, LoadedDemonstrations};
use blob_rl::model::{PolicyValueNet, PolicyValueNetConfig};
use blob_rl::observation::OBS_DIM;
use blob_rl::policy_artifact::load_behavior_clone;
use burn::{
    backend::{Autodiff, NdArray},
    prelude::*,
    record::CompactRecorder,
};
use sha2::{Digest, Sha256};

type B = Autodiff<NdArray<f32>>;
fn outputs(model: &PolicyValueNet<B>) -> Vec<f32> {
    let output = model.forward(Tensor::from_data(
        TensorData::new(vec![0.2; OBS_DIM], [1, OBS_DIM]),
        &Default::default(),
    ));
    Tensor::cat(
        vec![
            output.action_kind_logits,
            output.action_kind_expert_logits,
            output.phase_gate_logits,
            output.target_logits,
            output.effort_logits,
            output.amount_logits,
            output.signal_logits,
            output.signal_strength_logits,
            output.values,
            output.next_memory,
        ],
        1,
    )
    .into_data()
    .to_vec::<f32>()
    .unwrap()
}

#[test]
fn all_supported_layouts_share_verified_migration_and_execution_identity() {
    let mut config = TrainingConfig::default();
    config.env.world_size = 4;
    config.env.cells_per_team = 1;
    config.env.max_episode_len = 1024;
    config.env.victory.sim_time_limit_quanta = 16384;
    config.env.num_scattered_energy = 2;
    config.env.num_plants = 1;
    let (manifest, payload) = generate_demonstrations(
        &config,
        "config".into(),
        &DemonstrationOptions {
            teacher: MaintainedMindProfile::Simple,
            seeds: vec![3, 2, 4, 1],
            max_samples: 16,
        },
    )
    .unwrap();
    let dataset = LoadedDemonstrations {
        directory: "corpus".into(),
        manifest_sha256: "corpus".into(),
        manifest,
        payload,
    };
    let datasets = [dataset];
    let bc = BehaviorCloningConfig {
        epochs: 1,
        minibatch_size: 4,
        recurrent_unroll_steps: 1,
        validation_fraction: 0.5,
        ..Default::default()
    };
    let model_config = ModelConfig {
        hidden1: 8,
        hidden2: 8,
        recurrent_size: 4,
    };
    let device = Default::default();
    let (model, metrics) = behavior_clone::<B>(&datasets, &model_config, &bc, device).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("clone");
    publish_behavior_clone(&output, &model, &model_config, &bc, &datasets, &metrics).unwrap();
    let net = PolicyValueNetConfig {
        hidden1: 8,
        hidden2: 8,
        recurrent_size: 4,
    };
    let meta_path = output.join("behavior-cloning.json");
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
    let publish_metadata = |schema: u32, mutate: fn(&mut serde_json::Value)| {
        let mut metadata = metadata.clone();
        metadata["schema_version"] = serde_json::json!(schema);
        if schema < 36 {
            metadata.as_object_mut().unwrap().remove("seed_ledger");
        }
        if schema < 35 {
            metadata.as_object_mut().unwrap().remove("execution");
            metadata.as_object_mut().unwrap().remove("code_revision");
        }
        metadata["model_sha256"] = serde_json::json!(format!(
            "{:x}",
            Sha256::digest(std::fs::read(output.join("model.mpk")).unwrap())
        ));
        mutate(&mut metadata);
        std::fs::write(&meta_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
        behavior_clone_artifact_sha256(&output).unwrap()
    };
    macro_rules! legacy {
        ($schemas:expr, $init:ident, $migrate:ident) => {
            for schema in $schemas {
                net.$init::<B>(&device)
                    .save_file(output.join("model"), &CompactRecorder::new())
                    .unwrap();
                let stored = net
                    .$init::<B>(&device)
                    .load_file(output.join("model"), &CompactRecorder::new(), &device)
                    .unwrap();
                let expected = net.$migrate(stored, &device);
                let digest = publish_metadata(schema, |_| {});
                let actual =
                    load_behavior_clone::<B>(&output, &digest, &model_config, &device).unwrap();
                assert_eq!(outputs(&actual), outputs(&expected), "schema {schema}");
            }
        };
    }
    legacy!(22..25, init_legacy, migrate_legacy);
    legacy!(
        25..28,
        init_foraging_adapter_legacy,
        migrate_foraging_adapter
    );
    legacy!(
        28..30,
        init_header_context_adapter_legacy,
        migrate_header_context_adapter
    );
    legacy!(
        31..34,
        init_slot_context_adapter_legacy,
        migrate_slot_context_adapter
    );
    legacy!(
        34..37,
        init_pre_target_residual_legacy,
        migrate_pre_target_residual
    );
    {
        let schema = 37;
        net.init::<B>(&device)
            .save_file(output.join("model"), &CompactRecorder::new())
            .unwrap();
        let expected = net
            .init::<B>(&device)
            .load_file(output.join("model"), &CompactRecorder::new(), &device)
            .unwrap();
        let digest = publish_metadata(schema, |_| {});
        assert_eq!(
            outputs(&load_behavior_clone::<B>(&output, &digest, &model_config, &device).unwrap()),
            outputs(&expected)
        );
    }
    for schema in [21, 30, 38] {
        let digest = publish_metadata(schema, |_| {});
        assert!(verify_behavior_clone_artifact(&output, &digest, &model_config).is_err());
    }
    for mutate in [
        (|m: &mut serde_json::Value| {
            m["execution"]["expert_routing"] = "old-diffuse-router".into();
        }) as fn(&mut serde_json::Value),
        |m| {
            m.as_object_mut().unwrap().remove("execution");
        },
    ] {
        let digest = publish_metadata(36, mutate);
        assert!(load_behavior_clone::<B>(&output, &digest, &model_config, &device).is_err());
    }
}
