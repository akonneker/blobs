//! Fixed-batch overfit check on predeclared training-only hard examples.
#[path = "interaction_fit/row.rs"]
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
use data::{read, Row};
#[path = "context_fit/normalization.rs"]
mod normalization;
use evaluation::{background, choose, resolved_context, scores_with_contexts, Evaluation};
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
    #[arg(long)]
    verify: bool,
}
fn activation(
    f: &scalar::Frozen,
    rows: &[Row],
    arm: &str,
    donors: &[usize],
    overrides: Option<&[Vec<f32>]>,
) -> Value {
    let mut totals = [0_f64; 6];
    for (i, r) in rows.iter().enumerate() {
        let (_, stats) = f
            .forward_with_stats(
                &r.observation,
                resolved_context(arm, i, rows, donors, overrides),
            )
            .unwrap();
        for j in 0..5 {
            totals[j] += stats[j];
        }
        totals[5] = totals[5].max(stats[5]);
    }
    json!({"encoded_units":totals[0],"first_tanh_saturation":totals[1]/totals[0],"second_tanh_saturation":totals[2]/totals[0],"active_relu_fraction":totals[3]/totals[4],"max_first_preactivation":totals[5]})
}
fn scalar_loss(
    f: &scalar::Frozen,
    rows: &[Row],
    arm: &str,
    donors: &[usize],
    overrides: Option<&[Vec<f32>]>,
) -> f64 {
    rows.iter()
        .enumerate()
        .map(|(i, r)| {
            let delta = f
                .forward(
                    &r.observation,
                    resolved_context(arm, i, rows, donors, overrides),
                )
                .unwrap();
            let raw: Vec<f32> = background(r)
                .iter()
                .zip(delta)
                .map(|(&a, b)| a + b)
                .collect();
            let max = (0..10)
                .filter(|&k| r.legal_kinds[k])
                .map(|k| raw[k])
                .fold(f32::NEG_INFINITY, f32::max) as f64;
            let label = if i < 64 {
                r.teacher_kind
            } else {
                r.selected_kind
            };
            (0..10)
                .filter(|&k| r.legal_kinds[k])
                .map(|k| (raw[k] as f64 - max).exp())
                .sum::<f64>()
                .ln()
                + max
                - raw[label] as f64
        })
        .sum::<f64>()
        / rows.len() as f64
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    assert_eq!(sha(&fs::read(a.study.join("plan.json"))?), a.plan_sha256);
    let p = read(&a.study.join("plan.json"));
    let normalized = p.get("normalization").is_some();
    assert_eq!(
        p["arms"],
        if normalized {
            json!(["memory", "standardized"])
        } else {
            json!(["observation", "memory"])
        }
    );
    let arms: Vec<String> = serde_json::from_value(p["arms"].clone())?;
    assert_eq!(p["steps"], 4096);
    assert_eq!(p["checkpoints"], json!([128, 512, 2048, 4096]));
    assert_eq!(p["batch"], 128);
    assert_eq!(p["parameters"], 7514);
    assert_eq!(p["learning_rate"], 0.005);
    assert_eq!(
        sha(&fs::read(a.study.join("seed-audit.json"))?),
        p["seed_audit_sha256"]
    );
    partition::validate(&p, &read(&a.study.join("seed-audit.json"))["seed_ledger"])?;
    assert_eq!(
        sha(&fs::read(a.study.join("fixture.json"))?),
        p["fixture_sha256"]
    );
    let rows: Vec<Row> =
        serde_json::from_value(read(&a.study.join("fixture.json"))["rows"].clone())?;
    assert_eq!(rows.len(), 128);
    let standardized = if normalized {
        let config = &p["normalization"];
        let path = a.study.join(config["file"].as_str().unwrap());
        assert_eq!(sha(&fs::read(&path)?), config["sha256"]);
        let value = read(&path);
        assert_eq!(value["fixture_sha256"], p["fixture_sha256"]);
        assert_eq!(value["rows"], 128);
        let normalizer = normalization::MemoryNormalizer::new(
            serde_json::from_value(value["means"].clone())?,
            serde_json::from_value(value["scales"].clone())?,
        )?;
        Some(
            rows.iter()
                .map(|r| normalizer.apply(&r.memory))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let contexts = |arm: &str| {
        if arm == "standardized" {
            Some(
                standardized
                    .as_deref()
                    .expect("missing standardized context"),
            )
        } else {
            None
        }
    };
    let validation: Vec<u64> = serde_json::from_value(p["validation_seeds"].clone())?;
    let (parent, _) = load_deployed_policy(
        Path::new(p["parent"].as_str().unwrap()),
        p["parent_manifest_sha256"].as_str().unwrap(),
        &validation,
    )?;
    let parent = parent.parent();
    let donors: Vec<usize> = (0..128).collect();
    let ids = donors.clone();
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r.context, "Interaction");
        assert_eq!(r.memory.len(), 128);
        assert_eq!(r.legal_kinds.len(), 10);
        let (_, t) = parent.forward_with_trace(&r.observation, &r.memory)?;
        assert_eq!(t.raw_kind, background(r));
        assert_eq!(t.hidden, r.hidden);
        assert_eq!(
            if i < 64 {
                r.teacher_kind
            } else {
                r.selected_kind
            },
            if i < 64 { 4 } else { 3 }
        );
    }
    let context_audit=standardized.as_ref().map(|values| json!({
        "fixture_sha256":p["fixture_sha256"],"rows":128,"context_width":128,
        "parent_memory_sha256":sha(&rows.iter().flat_map(|r|r.memory.iter().flat_map(|x|x.to_le_bytes())).collect::<Vec<_>>()),
        "standardized_context_sha256":sha(&values.iter().flat_map(|r|r.iter().flat_map(|x|x.to_le_bytes())).collect::<Vec<_>>())
    }));
    let e = Evaluation {
        rows: &rows,
        combat: 64,
        donors: &donors,
        parent,
    };
    if a.verify {
        if let Some(ref audit) = context_audit {
            assert_eq!(*audit, read(&a.study.join("fits/context-audit.json")));
        }
        let report = read(&a.study.join("fits/report.json"));
        let mut checks = vec![];
        for fit in report["results"].as_array().unwrap() {
            let bytes = read_bounded(
                &a.study.join(fit["weights_file"].as_str().unwrap()),
                1024 * 1024,
            )?;
            assert_eq!(sha(&bytes), fit["weights_sha256"]);
            assert_eq!(bytes.len(), 7514 * 4);
            let frozen = scalar::Frozen::new(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            )?;
            let arm = fit["arm"].as_str().unwrap();
            let metrics = if normalized {
                e.evaluate_with_contexts(&frozen, None, arm, contexts(arm))
            } else {
                e.evaluate(&frozen, None, arm)
            };
            for kind in ["combat", "feeding"] {
                for key in ["rows", "correct", "predictions"] {
                    assert_eq!(metrics[kind][key], fit["metrics"][kind][key]);
                }
            }
            assert_eq!(
                activation(&frozen, &rows, arm, &donors, contexts(arm)),
                fit["activation"]
            );
            let loss = scalar_loss(&frozen, &rows, arm, &donors, contexts(arm));
            assert!((loss - fit["loss"].as_f64().unwrap()).abs() < 1e-4);
            checks.push(json!({"seed":fit["seed"],"arm":arm,"step":fit["step"],"weights_sha256":fit["weights_sha256"],"metrics":metrics,"activation":fit["activation"],"scalar_loss":loss}));
        }
        let path = a.study.join("scalar-replay.json");
        if path.exists() {
            return Err("immutable replay exists".into());
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
    if let Some(audit) = context_audit {
        fs::write(
            out.join("context-audit.json"),
            serde_json::to_vec_pretty(&audit)?,
        )?;
    }
    for (i, r) in rows.iter().enumerate() {
        let label = if i < 64 {
            r.teacher_kind
        } else {
            r.selected_kind
        };
        let raw = background(r);
        let delta: Vec<_> = (0..10)
            .map(|k| {
                if k == label {
                    32. - raw[k]
                } else {
                    -32. - raw[k]
                }
            })
            .collect();
        let output = parent.forward_with_kind_delta(&r.observation, &r.memory, &delta)?;
        assert_eq!(choose(&output.kind, &r.legal_kinds), label);
    }
    fs::write(
        out.join("decoder-sanity.json"),
        serde_json::to_vec_pretty(
            &json!({"rows":128,"correct":128,"scope":"Label-coded score control bypasses learning; verifies legal labels and inherited kind decoding only."}),
        )?,
    )?;
    let device = Default::default();
    let seeds: Vec<u64> = serde_json::from_value(p["sampling_seeds"].clone())?;
    let mut results = vec![];
    for seed in seeds {
        for arm in &arms {
            let arm = arm.as_str();
            NdArray::<f32>::seed(&device, seed);
            let mut m = model::Residual::<Autodiff<NdArray>>::new(&device);
            assert_eq!(m.num_params(), 7514);
            let initial_sha = sha(&m
                .values()
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>());
            let mut optimizer = AdamWConfig::new()
                .with_weight_decay(0.)
                .init()
                .with_grad_clipping(GradientClipping::Norm(1.));
            for step in 1..=4096 {
                let label_values: Vec<i32> = (0..128).map(|i| if i < 64 { 4 } else { 3 }).collect();
                let labels = Tensor::<Autodiff<NdArray>, 2, Int>::from_data(
                    TensorData::new(label_values.clone(), [128, 1]),
                    &device,
                );
                let loss = burn::tensor::activation::log_softmax(
                    scores_with_contexts(&m, &rows, &ids, arm, &donors, true, contexts(arm)),
                    1,
                )
                .gather(1, labels)
                .mean()
                .neg();
                let grads = GradientsParams::from_grads(loss.backward(), &m);
                m = optimizer.step(0.005, m, grads);
                if [128, 512, 2048, 4096].contains(&step) {
                    let valid = m.valid();
                    let values = valid.values();
                    let frozen = scalar::Frozen::new(values.clone())?;
                    let labels = Tensor::<NdArray, 2, Int>::from_data(
                        TensorData::new(label_values, [128, 1]),
                        &device,
                    );
                    let loss = burn::tensor::activation::log_softmax(
                        scores_with_contexts(
                            &valid,
                            &rows,
                            &ids,
                            arm,
                            &donors,
                            true,
                            contexts(arm),
                        ),
                        1,
                    )
                    .gather(1, labels)
                    .mean()
                    .neg()
                    .into_data()
                    .to_vec::<f32>()?[0];
                    let metrics = if normalized {
                        e.evaluate_with_contexts(&frozen, Some(&valid), arm, contexts(arm))
                    } else {
                        e.evaluate(&frozen, Some(&valid), arm)
                    };
                    let stats = activation(&frozen, &rows, arm, &donors, contexts(arm));
                    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let file = format!("fits/{seed}-{arm}-{step}.f32");
                    fs::write(a.study.join(&file), &bytes)?;
                    println!(
                        "{seed}-{arm}-{step}: combat={} feeding={} loss={loss}",
                        metrics["combat"]["correct"], metrics["feeding"]["correct"]
                    );
                    results.push(json!({"seed":seed,"arm":arm,"step":step,"initial_weights_sha256":initial_sha,"weights_file":file,"weights_sha256":sha(&bytes),"metrics":metrics,"activation":stats,"loss":loss}));
                }
            }
        }
    }
    fs::write(
        out.join("report.json"),
        serde_json::to_vec(
            &json!({"complete":true,"plan_sha256":a.plan_sha256,"results":results}),
        )?,
    )?;
    Ok(())
}
