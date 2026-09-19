//! Matched per-slot interaction residual fit on frozen ordinary prefixes.
#[path = "interaction_fit/data.rs"]
mod data;
#[path = "relational_fit/model.rs"]
mod model;
#[path = "interaction_fit/partition.rs"]
mod partition;
#[path = "interaction_fit/sampling.rs"]
mod sampling;
use blob_policy::{
    composite::{CompositePolicy, DeployedPolicy},
    observation::OBS_DIM,
    relational::{FrozenResidual, RelationalPolicy, EXECUTION_CONTRACT, PARAMETERS},
};
use blob_rl::deployed_artifact::{load_deployed_policy, read_bounded, sha};
use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{grad_clipping::GradientClipping, AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use clap::Parser;
use data::{load, read, Row};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    study: PathBuf,
    #[arg(long)]
    plan_sha256: String,
}
fn observations<B: Backend>(rows: &[&Row], device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(
            rows.iter()
                .flat_map(|r| r.observation.clone())
                .collect::<Vec<_>>(),
            [rows.len(), OBS_DIM],
        ),
        device,
    )
}
fn background(r: &Row) -> Vec<f32> {
    (0..10)
        .map(|k| (r.base_kind[k] + r.context_delta[k]) + r.slot_delta[k])
        .collect()
}
fn scores<B: Backend>(model: &model::Residual<B>, rows: &[&Row], mask: bool) -> Tensor<B, 2> {
    let device = model.head.weight.device();
    let raw = Tensor::from_data(
        TensorData::new(
            rows.iter().flat_map(|r| background(r)).collect::<Vec<_>>(),
            [rows.len(), 10],
        ),
        &device,
    ) + model.forward(observations(rows, &device));
    if mask {
        raw + Tensor::from_data(
            TensorData::new(
                rows.iter()
                    .flat_map(|r| r.legal_kinds.iter().map(|&v| if v { 0. } else { -1e9 }))
                    .collect::<Vec<_>>(),
                [rows.len(), 10],
            ),
            &device,
        )
    } else {
        raw
    }
}
fn choose(values: &[f32], mask: &[bool]) -> usize {
    (0..10)
        .filter(|&k| mask[k])
        .max_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
fn evaluate(
    model: &model::Residual<NdArray>,
    policy: &RelationalPolicy,
    rows: &[Row],
    teacher: bool,
) -> Value {
    let (mut correct, mut attacks, mut hits, mut mismatch) = (0, 0, 0, 0);
    let mut max_error = 0_f32;
    let mut predictions = vec![];
    for chunk in rows.chunks(128) {
        let refs: Vec<_> = chunk.iter().collect();
        let burn = burn::tensor::activation::softmax(scores(model, &refs, false), 1)
            .clamp_min(1e-20)
            .log()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (i, r) in chunk.iter().enumerate() {
            let out = policy.forward(&r.observation, &r.memory).unwrap();
            let k = choose(&out.kind, &r.legal_kinds);
            let label = if teacher {
                r.teacher_kind
            } else {
                r.selected_kind
            };
            correct += usize::from(k == label);
            attacks += usize::from(label == 4);
            hits += usize::from(k == 4 && label == 4);
            mismatch += usize::from(k != choose(&burn[i * 10..i * 10 + 10], &r.legal_kinds));
            for j in 0..10 {
                max_error = max_error.max((out.kind[j] - burn[i * 10 + j]).abs());
            }
            predictions.push(k);
        }
    }
    json!({"rows":rows.len(),"correct":correct,"agreement":correct as f64/rows.len() as f64,"attack_labels":attacks,"attack_correct":hits,"attack_agreement":if attacks==0{1.}else{hits as f64/attacks as f64},"portable_burn_kind_mismatches":mismatch,"max_log_probability_difference":max_error,"predictions":predictions})
}
fn export(
    dir: &Path,
    policy: &RelationalPolicy,
    manifest: &Value,
    plan: &Value,
    plan_sha: &str,
    ledger: &Value,
) -> Result<String, Box<dyn std::error::Error>> {
    fs::create_dir(dir)?;
    let bytes = policy.to_bytes();
    assert_eq!(RelationalPolicy::from_bytes(&bytes)?.to_bytes(), bytes);
    fs::write(dir.join("weights.bin"), &bytes)?;
    let m = json!({"schema_version":1,"weight_format":1,"execution_contract":EXECUTION_CONTRACT,"reference_mind_abi":manifest["reference_mind_abi"],"weight_bytes":bytes.len(),"weights_sha256":sha(&bytes),"source_seed_ledger":ledger,"parent_manifest_sha256":plan["parent_manifest_sha256"],"parent_weights_sha256":plan["parent_weights_sha256"],"study_plan_sha256":plan_sha,"qualification":"Unpromoted diagnostic residual; native/Burn results reported separately, WASM and ecology require separate evidence"});
    fs::write(dir.join("export.json"), serde_json::to_vec_pretty(&m)?)?;
    Ok(sha(&bytes))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    let bytes = read_bounded(&a.study.join("plan.json"), 1024 * 1024)?;
    assert_eq!(sha(&bytes), a.plan_sha256);
    let p: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(p["intervention"], "relational_residual");
    assert_eq!(p["steps"], 1024);
    assert_eq!(p["batch"], 128);
    assert_eq!(p["learning_rate"], 0.005);
    let out = a.study.join("fits");
    if out.exists() {
        return Err("immutable fit output exists".into());
    }
    let source = PathBuf::from(p["corpus_source"].as_str().unwrap());
    assert_eq!(
        sha(&fs::read(source.join("plan.json"))?),
        p["corpus_source_plan_sha256"].as_str().unwrap()
    );
    let audit = read(&a.study.join("seed-audit.json"));
    assert_eq!(
        sha(&fs::read(a.study.join("seed-audit.json"))?),
        p["seed_audit_sha256"].as_str().unwrap()
    );
    let ledger = &audit["seed_ledger"];
    let train_seeds = partition::training_seeds(&p)?;
    partition::validate(&p, ledger)?;
    let validation: Vec<u64> = serde_json::from_value(p["validation_seeds"].clone())?;
    let seeds: Vec<u64> = serde_json::from_value(p["sampling_seeds"].clone())?;
    for s in train_seeds.iter().chain(&seeds) {
        assert!(ledger["training"].as_array().unwrap().contains(&json!(s)));
    }
    for s in &validation {
        assert!(ledger["validation"].as_array().unwrap().contains(&json!(s)));
        assert!(!ledger["training"].as_array().unwrap().contains(&json!(s)));
    }
    let (deployed, manifest) = load_deployed_policy(
        Path::new(p["parent"].as_str().unwrap()),
        p["parent_manifest_sha256"].as_str().unwrap(),
        &validation,
    )?;
    assert_eq!(manifest["weights_sha256"], p["parent_weights_sha256"]);
    let base: CompositePolicy = match deployed {
        DeployedPolicy::Composite(c) => c,
        _ => return Err("requires declared composite parent".into()),
    };
    let base_bytes = base.to_bytes();
    let mut train = data::Corpus {
        combat: vec![],
        feeding: vec![],
        audit: Value::Null,
    };
    let mut partitions = vec![];
    for &seed in &train_seeds {
        let corpus = load(&partition::root(&p, &a.study, seed)?, seed);
        partitions.push(json!({"seed":seed,"files":corpus.audit}));
        train.combat.extend(corpus.combat);
        train.feeding.extend(corpus.feeding);
    }
    train.audit = if train_seeds.len() == 1 {
        partitions[0]["files"].clone()
    } else {
        json!({"partitions":partitions})
    };
    let feeding_pool = if let Some(config) = p.get("feeding_sampling") {
        assert_eq!(config["hard_per_batch"], 32);
        let path = a
            .study
            .join(config["pool_file"].as_str().ok_or("missing pool file")?);
        assert_eq!(sha(&fs::read(&path)?), config["pool_sha256"]);
        let pool = read(&path);
        assert_eq!(pool["training_seeds"], p["training_seeds"]);
        assert_eq!(pool["sampling_seeds"], p["sampling_seeds"]);
        assert_eq!(pool["feeding_rows"], train.feeding.len());
        assert_eq!(pool["source_study"], p["sampling_source"]);
        assert_eq!(
            pool["source_replay_sha256"],
            p["sampling_source_replay_sha256"]
        );
        let source = Path::new(pool["source_study"].as_str().unwrap());
        assert_eq!(
            sha(&fs::read(source.join("portable-replay.json"))?),
            pool["source_replay_sha256"]
        );
        assert_eq!(
            sha(&fs::read(source.join("fits/corpus-audit.json"))?),
            pool["source_corpus_audit_sha256"]
        );
        Some(sampling::FeedingPool::new(
            serde_json::from_value(pool["hard_indices"].clone())?,
            train.feeding.len(),
        )?)
    } else {
        None
    };
    let val: Vec<_> = validation
        .iter()
        .map(|&s| load(&partition::root(&p, &a.study, s).unwrap(), s))
        .collect();
    let all: Vec<_> = train
        .combat
        .iter()
        .chain(&train.feeding)
        .chain(val.iter().flat_map(|c| c.combat.iter().chain(&c.feeding)))
        .collect();
    // Verify recorded frozen representations and exact zero migration before fitting.
    let zero = RelationalPolicy {
        base: base.clone(),
        residual: FrozenResidual::new(vec![0.; PARAMETERS])?,
    };
    for r in &all {
        let (old, trace) = base.parent.forward_with_trace(&r.observation, &r.memory)?;
        assert_eq!(trace.raw_kind, background(r));
        assert_eq!(trace.hidden, r.hidden);
        let new = zero.forward(&r.observation, &r.memory)?;
        for (a, b) in [
            &old.kind,
            &old.target,
            &old.effort,
            &old.amount,
            &old.signal,
            &old.signal_strength,
            &old.next_memory,
        ]
        .into_iter()
        .zip([
            &new.kind,
            &new.target,
            &new.effort,
            &new.amount,
            &new.signal,
            &new.signal_strength,
            &new.next_memory,
        ]) {
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    fs::create_dir(&out)?;
    fs::write(
        out.join("corpus-audit.json"),
        serde_json::to_vec_pretty(
            &json!({"train":train.audit,"validation":val.iter().map(|v|v.audit.clone()).collect::<Vec<_>>(),"zero_migration_rows":all.len(),"all_outputs_bit_identical":true}),
        )?,
    )?;
    export(
        &out.join("zero"),
        &zero,
        &manifest,
        &p,
        &a.plan_sha256,
        ledger,
    )?;
    let device = Default::default();
    let mut results = vec![];
    for seed in seeds {
        for treatment in [false, true] {
            NdArray::<f32>::seed(&device, seed);
            let mut m = model::Residual::<Autodiff<NdArray>>::new(&device);
            assert_eq!(m.num_params(), PARAMETERS);
            let initial = FrozenResidual::new(m.values())?;
            for r in &all {
                assert!(initial.forward(&r.observation)?.iter().all(|x| *x == 0.));
            }
            let mut optimizer = AdamWConfig::new()
                .with_weight_decay(0.)
                .init()
                .with_grad_clipping(GradientClipping::Norm(1.));
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut losses = vec![];
            for step in 0..1024 {
                let mut rows = vec![];
                let mut labels = vec![];
                for feeding in [false, true] {
                    let set = if feeding {
                        &train.feeding
                    } else {
                        &train.combat
                    };
                    for position in 0..64 {
                        let index = if feeding {
                            feeding_pool
                                .as_ref()
                                .map(|pool| pool.sample(&mut rng, position))
                        } else {
                            None
                        }
                        .unwrap_or_else(|| rng.random_range(0..set.len()));
                        let r = &set[index];
                        rows.push(r);
                        labels.push(if treatment && !feeding {
                            r.teacher_kind as i32
                        } else {
                            r.selected_kind as i32
                        });
                    }
                }
                let labels = Tensor::<Autodiff<NdArray>, 2, Int>::from_data(
                    TensorData::new(labels, [128, 1]),
                    &device,
                );
                let loss = burn::tensor::activation::log_softmax(scores(&m, &rows, true), 1)
                    .gather(1, labels)
                    .mean()
                    .neg();
                if step % 256 == 0 || step == 1023 {
                    losses.push(
                        json!({"step":step,"loss":loss.clone().into_data().to_vec::<f32>()?[0]}),
                    );
                }
                let grads = GradientsParams::from_grads(loss.backward(), &m);
                m = optimizer.step(0.005, m, grads);
            }
            let m = m.valid();
            let policy = RelationalPolicy {
                base: base.clone(),
                residual: FrozenResidual::new(m.values())?,
            };
            assert_eq!(policy.base.to_bytes(), base_bytes);
            let mut evaluations = vec![];
            let mut pass = true;
            for (&seed, v) in validation.iter().zip(&val) {
                let combat = evaluate(&m, &policy, &v.combat, treatment);
                let feeding = evaluate(&m, &policy, &v.feeding, false);
                pass &= combat["agreement"].as_f64().unwrap() >= 0.95
                    && (!treatment || combat["attack_agreement"].as_f64().unwrap() >= 0.9)
                    && feeding["agreement"].as_f64().unwrap() >= 0.99
                    && combat["portable_burn_kind_mismatches"] == 0
                    && feeding["portable_burn_kind_mismatches"] == 0;
                evaluations.push(json!({"seed":seed,"combat":combat,"feeding":feeding}));
            }
            let arm = if treatment { "treatment" } else { "control" };
            let dir = out.join(format!("{seed}-{arm}"));
            let digest = export(&dir, &policy, &manifest, &p, &a.plan_sha256, ledger)?;
            let result = json!({"sampling_seed":seed,"initialization_seed":seed,"arm":arm,"added_parameters":PARAMETERS,"parent_and_utility_bytes_identical":true,"weights_sha256":digest,"losses":losses,"validation":evaluations,"offline_gate_pass":pass});
            fs::write(dir.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
            println!("{seed}-{arm}: gate={pass}");
            results.push(result);
        }
    }
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"plan_sha256":a.plan_sha256,"all_pass":results.iter().all(|r|r["offline_gate_pass"]==true),"results":results}),
        )?,
    )?;
    Ok(())
}
