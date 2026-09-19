//! Training-only direct private-context diagnostic, never a deployed export.
#[path = "interaction_fit/data.rs"]
mod data;
#[path = "context_fit/evaluation.rs"]
mod evaluation;
#[path = "context_fit/model.rs"]
mod model;
#[path = "interaction_fit/partition.rs"]
mod partition;
#[path = "context_fit/scalar.rs"]
mod scalar;
use blob_rl::deployed_artifact::{load_deployed_policy, read_bounded, sha};
use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{grad_clipping::GradientClipping, AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use clap::Parser;
use data::read;
use evaluation::{background, scores, Evaluation};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    study: PathBuf,
    #[arg(long)]
    plan_sha256: String,
    #[arg(long)]
    verify: bool,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    assert_eq!(sha(&fs::read(a.study.join("plan.json"))?), a.plan_sha256);
    let p = read(&a.study.join("plan.json"));
    assert_eq!(p["arms"], json!(["observation", "memory", "shuffled"]));
    assert_eq!(p["steps"], 2048);
    assert_eq!(p["checkpoints"], json!([512, 1024, 2048]));
    assert_eq!(p["batch"], 128);
    assert_eq!(p["learning_rate"], 0.005);
    assert_eq!(p["parameters"], scalar::PARAMETERS);
    let audit = read(&a.study.join("seed-audit.json"));
    assert_eq!(
        sha(&fs::read(a.study.join("seed-audit.json"))?),
        p["seed_audit_sha256"]
    );
    partition::validate(&p, &audit["seed_ledger"])?;
    let validation: Vec<u64> = serde_json::from_value(p["validation_seeds"].clone())?;
    let (parent, _) = load_deployed_policy(
        std::path::Path::new(p["parent"].as_str().unwrap()),
        p["parent_manifest_sha256"].as_str().unwrap(),
        &validation,
    )?;
    let mut rows = vec![];
    let mut feeding = vec![];
    let mut corpus_audit = vec![];
    for seed in partition::training_seeds(&p)? {
        let c = data::load(&partition::root(&p, &a.study, seed)?, seed);
        corpus_audit.push(json!({"seed":seed,"files":c.audit}));
        rows.extend(c.combat);
        feeding.extend(c.feeding);
    }
    let combat = rows.len();
    rows.extend(feeding);
    assert_eq!(
        sha(&fs::read(a.study.join("memory-donors.json"))?),
        p["donor_sha256"]
    );
    let donors: Vec<usize> =
        serde_json::from_value(read(&a.study.join("memory-donors.json"))["donors"].clone())?;
    assert_eq!(donors.len(), rows.len());
    let mut sorted = donors.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..rows.len()).collect::<Vec<_>>());
    assert!(donors.iter().enumerate().all(|(i, &d)| i != d));
    for r in &rows {
        assert_eq!(r.memory.len(), 128);
        let (_, t) = parent
            .parent()
            .forward_with_trace(&r.observation, &r.memory)?;
        assert_eq!(t.raw_kind, background(r));
        assert_eq!(t.hidden, r.hidden);
    }
    let e = Evaluation {
        rows: &rows,
        combat,
        donors: &donors,
        parent: parent.parent(),
    };
    if a.verify {
        let report = read(&a.study.join("fits/report.json"));
        let mut checks = vec![];
        for fit in report["results"].as_array().unwrap() {
            let arm = fit["arm"].as_str().unwrap();
            let file = a.study.join(fit["weights_file"].as_str().unwrap());
            let bytes = read_bounded(&file, 1024 * 1024)?;
            assert_eq!(sha(&bytes), fit["weights_sha256"]);
            assert_eq!(bytes.len(), scalar::PARAMETERS * 4);
            let frozen = scalar::Frozen::new(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            )?;
            let metrics = e.evaluate(&frozen, None, arm);
            for kind in ["combat", "feeding"] {
                for key in [
                    "rows",
                    "correct",
                    "attack_labels",
                    "attack_correct",
                    "predictions",
                ] {
                    assert_eq!(metrics[kind][key], fit["metrics"][kind][key]);
                }
            }
            checks.push(json!({"seed":fit["seed"],"arm":arm,"step":fit["step"],"weights_sha256":fit["weights_sha256"],"metrics":metrics}));
        }
        let path = a.study.join("scalar-replay.json");
        if path.exists() {
            return Err("immutable scalar replay exists".into());
        }
        fs::write(
            path,
            serde_json::to_vec(
                &json!({"complete":true,"fit_report_sha256":sha(&fs::read(a.study.join("fits/report.json"))?),"results":checks}),
            )?,
        )?;
        return Ok(());
    }
    let out = a.study.join("fits");
    fs::create_dir(&out)?;
    fs::write(
        out.join("corpus-audit.json"),
        serde_json::to_vec_pretty(
            &json!({"training":corpus_audit,"combat_rows":combat,"feeding_rows":rows.len()-combat,"validation_rows_loaded":0}),
        )?,
    )?;
    let device = Default::default();
    let seeds: Vec<u64> = serde_json::from_value(p["sampling_seeds"].clone())?;
    let mut results = vec![];
    for seed in seeds {
        for arm in ["observation", "memory", "shuffled"] {
            NdArray::<f32>::seed(&device, seed);
            let mut m = model::Residual::<Autodiff<NdArray>>::new(&device);
            assert_eq!(m.num_params(), scalar::PARAMETERS);
            let initial_sha = sha(&m
                .values()
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>());
            let mut optimizer = AdamWConfig::new()
                .with_weight_decay(0.)
                .init()
                .with_grad_clipping(GradientClipping::Norm(1.));
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut losses = vec![];
            let mut stream = Sha256::new();
            for step in 1..=2048 {
                let mut ids: Vec<_> = (0..64).map(|_| rng.random_range(0..combat)).collect();
                ids.extend((0..64).map(|_| rng.random_range(combat..rows.len())));
                for &i in &ids {
                    stream.update((i as u64).to_le_bytes());
                }
                let labels: Vec<i32> = ids
                    .iter()
                    .map(|&i| {
                        if i < combat {
                            rows[i].teacher_kind as i32
                        } else {
                            rows[i].selected_kind as i32
                        }
                    })
                    .collect();
                let labels = Tensor::<Autodiff<NdArray>, 2, Int>::from_data(
                    TensorData::new(labels, [128, 1]),
                    &device,
                );
                let loss = burn::tensor::activation::log_softmax(
                    scores(&m, &rows, &ids, arm, &donors, true),
                    1,
                )
                .gather(1, labels)
                .mean()
                .neg();
                if step == 1 || step % 128 == 0 {
                    losses.push(
                        json!({"step":step,"loss":loss.clone().into_data().to_vec::<f32>()?[0]}),
                    );
                }
                let grads = GradientsParams::from_grads(loss.backward(), &m);
                m = optimizer.step(0.005, m, grads);
                if [512, 1024, 2048].contains(&step) {
                    let valid = m.valid();
                    let values = valid.values();
                    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let file = format!("fits/{seed}-{arm}-{step}.f32");
                    fs::write(a.study.join(&file), &bytes)?;
                    let metrics = e.evaluate(&scalar::Frozen::new(values)?, Some(&valid), arm);
                    println!(
                        "{seed}-{arm}-{step}: combat={} feeding={}",
                        metrics["combat"]["agreement"], metrics["feeding"]["agreement"]
                    );
                    results.push(json!({"seed":seed,"arm":arm,"step":step,"initial_weights_sha256":initial_sha,"batch_stream_sha256":format!("{:x}",stream.clone().finalize()),"weights_file":file,"weights_sha256":sha(&bytes),"losses":losses,"metrics":metrics}));
                }
            }
        }
    }
    fs::write(
        out.join("report.json"),
        serde_json::to_vec(
            &json!({"complete":true,"plan_sha256":a.plan_sha256,"results":results,"scope":"Training-only diagnostic; no deployment artifact or generalization claim."}),
        )?,
    )?;
    Ok(())
}
