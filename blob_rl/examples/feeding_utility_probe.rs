//! Matched target-utility transfer on verified frozen-policy observations.
//! This is a conditional offline gate, not a deployed composite policy.
#[path = "feeding_utility/data.rs"]
mod data;
#[path = "feeding_utility/evaluate.rs"]
mod evaluate;
#[path = "feeding_utility/loss.rs"]
mod loss;
#[path = "feeding_utility/model.rs"]
mod model;
use blob_rl::behavior_cloning::{
    behavior_clone_artifact_sha256, seed_ledger::InheritedSeedLedger, BehaviorCloningArtifact,
};
use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{grad_clipping::GradientClipping, AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
const TRAIN_SEED: u64 = 1433000101;
const OLD_VALIDATION: u64 = 1433000202;
const FRESH: [u64; 2] = [1435400201, 1435400202];
const INITIALIZATIONS: [u64; 3] = [1435400301, 1435400302, 1435400303];
const SAMPLING_SEED: u64 = 1435400101;
const STEPS: usize = 1024;
const BATCH: usize = 128;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    pairs_root: PathBuf,
    #[arg(long)]
    fresh_pairs_root: PathBuf,
    #[arg(long)]
    parent: PathBuf,
    #[arg(long)]
    lineage_inventory: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    objective: Option<loss::Objective>,
    #[arg(long)]
    save_records: bool,
}
fn read(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    fs::create_dir(&args.output)?;
    let parent_hash = behavior_clone_artifact_sha256(&args.parent)?;
    let parent: BehaviorCloningArtifact =
        serde_json::from_slice(&fs::read(args.parent.join("behavior-cloning.json"))?)?;
    let inventory = read(&args.lineage_inventory)?;
    let paths = |key: &str| -> Result<Vec<PathBuf>, String> {
        inventory[key]
            .as_array()
            .ok_or("invalid lineage inventory")?
            .iter()
            .map(|x| {
                x.as_str()
                    .map(PathBuf::from)
                    .ok_or("invalid lineage path".into())
            })
            .collect()
    };
    let inherited = InheritedSeedLedger::audit_legacy(
        &args.parent,
        &parent_hash,
        &parent.model,
        &paths("chain")?,
        &paths("corpora")?,
    )?;
    let reserved = [1434999901, 1434999902];
    let used = [
        TRAIN_SEED,
        OLD_VALIDATION,
        SAMPLING_SEED,
        FRESH[0],
        FRESH[1],
        INITIALIZATIONS[0],
        INITIALIZATIONS[1],
        INITIALIZATIONS[2],
    ];
    if used.iter().any(|s| {
        reserved.contains(s)
            || inherited.ledger().training.contains(s)
            || inherited.ledger().validation.contains(s)
            || inherited.ledger().confirmation.contains(s)
    }) {
        return Err("probe seed overlaps reserved/parent lineage".into());
    }
    let (old, audit) = data::load(&args.pairs_root)?;
    let (fresh, fresh_audit) = data::load(&args.fresh_pairs_root)?;
    for (before, after) in audit["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .zip(fresh_audit["datasets"].as_array().unwrap())
    {
        for field in [
            "layout",
            "source_config_sha256",
            "effective_config_sha256",
            "semantic_ruleset_hash",
            "compiled_ruleset_hash",
        ] {
            if before[field] != after[field] {
                return Err(format!("fresh corpus changes {field}").into());
            }
        }
        for corpus in [before, after] {
            if corpus["collection"]["behavior_clone_metadata_sha256"] != parent_hash {
                return Err("corpus parent mismatch".into());
            }
        }
    }
    if old
        .iter()
        .any(|r| ![TRAIN_SEED, OLD_VALIDATION].contains(&r.seed))
        || fresh.iter().any(|r| !FRESH.contains(&r.seed))
    {
        return Err("unexpected corpus seed".into());
    }
    let train: Vec<_> = old
        .iter()
        .filter(|r| r.seed == TRAIN_SEED)
        .cloned()
        .collect();
    let validation: Vec<_> = old
        .iter()
        .filter(|r| r.seed == OLD_VALIDATION)
        .cloned()
        .collect();
    if train.is_empty() || validation.is_empty() || fresh.is_empty() {
        return Err("empty partition".into());
    }
    let sources: &[(&str, &[u8])] = &[
        ("probe.rs", include_bytes!("feeding_utility_probe.rs")),
        ("loss.rs", include_bytes!("feeding_utility/loss.rs")),
        ("data.rs", include_bytes!("feeding_utility/data.rs")),
        ("model.rs", include_bytes!("feeding_utility/model.rs")),
        ("evaluate.rs", include_bytes!("feeding_utility/evaluate.rs")),
        (
            "portable_sampling.rs",
            include_bytes!("../../blob_policy/src/sampling.rs"),
        ),
        ("Cargo.lock", include_bytes!("../../Cargo.lock")),
        ("blob_rl.Cargo.toml", include_bytes!("../Cargo.toml")),
    ];
    let mut hashes = std::collections::BTreeMap::new();
    for (name, bytes) in sources {
        hashes.insert(name, format!("{:x}", Sha256::digest(bytes)));
        fs::write(args.output.join(name), bytes)?;
    }
    let objectives = args.objective.map_or_else(
        || {
            vec![
                loss::Objective::CrossEntropy,
                loss::Objective::QuantileInterval,
            ]
        },
        |objective| vec![objective],
    );
    let plan = json!({"scope":"Offline adjunct Move-utility gate on encoded ordinary observations. Parent choices supply fixed kind/effort and other outputs; no parent parameter updates, deployable checkpoint, fresh ABI export or ecology.","parent_sha256":parent_hash,"inherited_seed_ledger":inherited,"train_seed":TRAIN_SEED,"old_development_seed":OLD_VALIDATION,"fresh_development_seeds":FRESH,"initialization_seeds":INITIALIZATIONS,"sampling_seed":SAMPLING_SEED,"reserved_confirmation_seeds":reserved,"steps":STEPS,"batch":BATCH,"learning_rate":0.005,"sampling":"uniform with replacement; identical stream in every arm","arms":["control","treatment"],"objectives":objectives.iter().map(|objective|format!("{objective:?}")).collect::<Vec<_>>(),"save_full_precision_diagnostic_records":args.save_records,"code_revision":env!("BLOB_CODE_REVISION"),"revision":3,"revision_reason":"Retain exact-f32 diagnostic records after all three interval-loss initializations pass the transfer gate. Models, labels, loss, seeds and training budget are unchanged. Reloaded records must reproduce both development evaluations exactly; these are not deployed Mind artifacts.","features":"33 raw slot features, dx/dy restored to tile scale; all random/header/host keys excluded. 33->16->16 tanh; sorted legal sum/32; local+encoding+summary ->64 ReLU->1 zero head. Exact fixed-parent-effort target mask.","gate":"Every initialization and arm: >=95% Move agreement, >=90% active-slice agreement, >=95% inactive Move retention, on old and new development cohorts; finite scores and exact row/slot permutation. Non-Move choices pass through by construction, not tested as live learned-policy behavior.","source_sha256":hashes,"old_audit":audit,"fresh_audit":fresh_audit});
    fs::write(
        args.output.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    let device = Default::default();
    let mut results = Vec::new();
    for seed in INITIALIZATIONS {
        for control in [true, false] {
            for &objective in &objectives {
                NdArray::<f32>::seed(&device, seed);
                let mut utility = model::Utility::<Autodiff<NdArray>>::new(&device);
                let initial = evaluate::evaluate(&utility.valid(), &validation, control);
                let mut optimizer = AdamWConfig::new()
                    .init()
                    .with_grad_clipping(GradientClipping::Norm(1.));
                let mut rng = ChaCha8Rng::seed_from_u64(SAMPLING_SEED);
                let mut losses = Vec::new();
                for step in 0..STEPS {
                    let selected: Vec<_> = (0..BATCH)
                        .map(|_| train[rng.random_range(0..train.len())].clone())
                        .collect();
                    let (local, legal) = model::batch(&selected, &device);
                    let loss =
                        loss::loss(utility.forward(local, legal), &selected, control, objective);
                    if step % 256 == 0 || step == STEPS - 1 {
                        losses.push(json!({"step":step,"objective_value":loss.clone().into_data().to_vec::<f32>()?[0]}));
                    }
                    let grads = GradientsParams::from_grads(loss.backward(), &utility);
                    utility = optimizer.step(0.005, utility, grads);
                }
                let utility = utility.valid();
                let old = evaluate::evaluate(&utility, &validation, control);
                let new = evaluate::evaluate(&utility, &fresh, control);
                let record = if args.save_records {
                    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
                    let filename = format!(
                        "{seed}-{}-{objective:?}",
                        if control { "control" } else { "treatment" }
                    );
                    let path = args.output.join(&filename);
                    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
                    utility.clone().save_file(&path, &recorder)?;
                    let restored = model::Utility::<NdArray>::new(&device)
                        .load_file(&path, &recorder, &device)?;
                    assert_eq!(utility.parameter_bits(), restored.parameter_bits());
                    assert_eq!(evaluate::evaluate(&restored, &validation, control), old);
                    assert_eq!(evaluate::evaluate(&restored, &fresh, control), new);
                    let bytes = fs::read(path.with_extension("mpk"))?;
                    Some(
                        json!({"file":format!("{filename}.mpk"),"sha256":format!("{:x}",Sha256::digest(&bytes)),"bytes":bytes.len(),"precision":"f32","reload_evaluations_identical":true,"reload_parameter_bits_identical":true,"kind":"diagnostic utility record, not a policy checkpoint"}),
                    )
                } else {
                    None
                };
                let result = json!({"initialization_seed":seed,"objective":format!("{objective:?}"),"arm":if control{"control"}else{"treatment"},"parameters":utility.num_params(),"record":record,"initial_old":initial,"loss_checkpoints":losses,"old_development":old,"fresh_development":new});
                println!("{}", serde_json::to_string(&result)?);
                results.push(result);
                fs::write(
                    args.output.join("progress.json"),
                    serde_json::to_vec_pretty(&results)?,
                )?;
            }
        }
    }
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&json!({"complete":true,"plan":plan,"results":results}))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_rl::observation::SLOT_FEATURES;
    fn row() -> data::Row {
        let mut local = vec![0.; 32 * SLOT_FEATURES];
        for i in 0..8 {
            local[i * SLOT_FEATURES] = 1.;
            local[i * SLOT_FEATURES + 1] = 1.;
            local[i * SLOT_FEATURES + 2] = (i % 3) as f32 - 1.;
            local[i * SLOT_FEATURES + 3] = (i / 3) as f32 - 1.;
            local[i * SLOT_FEATURES + 8] = i as f32 / 32.;
        }
        data::Row {
            seed: 1,
            layout: 0,
            local,
            legal: std::array::from_fn(|i| i < 8),
            geometry_order: (0..8).collect(),
            word: 0,
            control: 0,
            treatment: 1,
            teacher: Some(1),
            active: true,
        }
    }
    #[test]
    fn active_network_excludes_labels_private_random_and_host_keys() {
        let device = Default::default();
        let mut model = model::Utility::<NdArray>::new(&device);
        model.activate_for_test(&device);
        let input = row();
        let mut changed = input.clone();
        changed.seed = 99;
        changed.layout = 4;
        changed.word = u64::MAX;
        changed.control = 3;
        changed.treatment = 4;
        changed.teacher = None;
        changed.active = false;
        let (local, legal) = model::batch(&[input, changed], &device);
        let scores = model
            .forward(local, legal)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(&scores[..32], &scores[32..]);
        assert!(scores[..8]
            .windows(2)
            .any(|p| p[0].to_bits() != p[1].to_bits()));
    }
    #[test]
    fn illegal_competitors_cannot_change_legal_scores_or_other_rows() {
        let device = Default::default();
        let mut model = model::Utility::<NdArray>::new(&device);
        model.activate_for_test(&device);
        let input = row();
        let mut changed = input.clone();
        changed.local[8 * SLOT_FEATURES..].fill(9.);
        let (local, legal) = model::batch(&[input.clone(), changed], &device);
        let scores = model
            .forward(local, legal)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(&scores[..8], &scores[32..40]);
        let (local, legal) = model::batch(&[input], &device);
        let alone = model
            .forward(local, legal)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(&scores[..32], alone);
    }
}
