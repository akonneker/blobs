//! Matched existing-head or local-adapter fits; portable offline gate, no promotion.
#[path = "interaction_fit/data.rs"]
mod data;
#[path = "interaction_fit/model.rs"]
mod model;
use blob_policy::{
    composite::{CompositePolicy, DeployedPolicy},
    runtime::FrozenPolicy,
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
use model::{Fit, Update};
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
fn scores<B: Backend>(head: &Fit<B>, rows: &[&Row], mask: bool) -> Tensor<B, 2> {
    let device = head.head.weight.device();
    let adapter = head.fc.is_some();
    let n = rows.len();
    let x = Tensor::from_data(
        TensorData::new(
            rows.iter()
                .flat_map(|r| {
                    if adapter {
                        model::local(&r.observation)
                    } else {
                        r.hidden.clone()
                    }
                })
                .collect::<Vec<_>>(),
            [n, if adapter { 71 } else { 128 }],
        ),
        &device,
    );
    let c = Tensor::from_data(
        TensorData::new(
            rows.iter()
                .flat_map(|r| r.context_delta.clone())
                .collect::<Vec<_>>(),
            [n, 10],
        ),
        &device,
    );
    let s = Tensor::from_data(
        TensorData::new(
            rows.iter()
                .flat_map(|r| {
                    if adapter {
                        r.base_kind.clone()
                    } else {
                        r.slot_delta.clone()
                    }
                })
                .collect::<Vec<_>>(),
            [n, 10],
        ),
        &device,
    );
    let y = if adapter {
        s + c + head.forward(x)
    } else {
        head.forward(x) + c + s
    };
    if mask {
        y + Tensor::from_data(
            TensorData::new(
                rows.iter()
                    .flat_map(|r| r.legal_kinds.iter().map(|&v| if v { 0. } else { -1e9 }))
                    .collect::<Vec<_>>(),
                [n, 10],
            ),
            &device,
        )
    } else {
        y
    }
}
fn choose(scores: &[f32], mask: &[bool]) -> usize {
    (0..10)
        .filter(|&k| mask[k])
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
fn evaluate(head: &Fit<NdArray>, policy: &FrozenPolicy, rows: &[Row], teacher: bool) -> Value {
    let (mut correct, mut attack, mut attack_correct, mut mismatch) = (0, 0, 0, 0);
    let mut max_error = 0_f32;
    for chunk in rows.chunks(128) {
        let refs: Vec<_> = chunk.iter().collect();
        let raw = scores(head, &refs, false);
        let burn = burn::tensor::activation::softmax(raw, 1)
            .clamp_min(1e-20)
            .log()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (i, r) in chunk.iter().enumerate() {
            let (out, trace) = policy
                .forward_with_trace(&r.observation, &r.memory)
                .unwrap();
            // A head update cannot change the representation or either inherited adapter.
            assert_eq!(trace.hidden, r.hidden);
            assert_eq!(trace.context_delta, r.context_delta);
            if head.fc.is_some() {
                assert_eq!(trace.base_kind, r.base_kind);
            } else {
                assert_eq!(trace.slot_delta, r.slot_delta);
            }
            let selected = choose(&out.kind, &r.legal_kinds);
            let label = if teacher {
                r.teacher_kind
            } else {
                r.selected_kind
            };
            correct += usize::from(selected == label);
            attack += usize::from(label == 4);
            attack_correct += usize::from(label == 4 && selected == 4);
            mismatch += usize::from(selected != choose(&burn[i * 10..i * 10 + 10], &r.legal_kinds));
            for k in 0..10 {
                max_error = max_error.max((out.kind[k] - burn[i * 10 + k]).abs());
            }
        }
    }
    json!({"rows":rows.len(),"correct":correct,"agreement":correct as f64/rows.len() as f64,"attack_labels":attack,"attack_correct":attack_correct,"attack_agreement":if attack==0 {1.}else{attack_correct as f64/attack as f64},"portable_burn_kind_mismatches":mismatch,"max_log_probability_difference":max_error})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    let bytes = read_bounded(&a.study.join("plan.json"), 1024 * 1024)?;
    assert_eq!(sha(&bytes), a.plan_sha256);
    let p: Value = serde_json::from_slice(&bytes)?;
    let out = a.study.join("fits");
    if out.exists() {
        return Err("immutable fits already exist".into());
    }
    assert_eq!(p["steps"], 1024);
    assert_eq!(p["batch"], 128);
    assert_eq!(p["learning_rate"], 0.005);
    let train_seed = p["training_seed"].as_u64().unwrap();
    let validations: Vec<u64> = serde_json::from_value(p["validation_seeds"].clone())?;
    let seeds: Vec<u64> = serde_json::from_value(p["sampling_seeds"].clone())?;
    let audit = read(&a.study.join("seed-audit.json"));
    assert_eq!(
        sha(&fs::read(a.study.join("seed-audit.json"))?),
        p["seed_audit_sha256"].as_str().unwrap()
    );
    assert!(audit["seed_ledger"]["training"]
        .as_array()
        .unwrap()
        .contains(&json!(train_seed)));
    for s in &validations {
        assert!(audit["seed_ledger"]["validation"]
            .as_array()
            .unwrap()
            .contains(&json!(s)));
        assert!(!audit["seed_ledger"]["training"]
            .as_array()
            .unwrap()
            .contains(&json!(s)));
    }
    let (deployed, manifest) = load_deployed_policy(
        Path::new(p["parent"].as_str().unwrap()),
        p["parent_manifest_sha256"].as_str().unwrap(),
        &validations,
    )?;
    assert_eq!(manifest["weights_sha256"], p["parent_weights_sha256"]);
    let parent = match &deployed {
        DeployedPolicy::Composite(c) => &c.parent,
        _ => return Err("requires declared composite parent".into()),
    };
    let original = parent.to_bytes();
    let update = Update::from_plan(&p["intervention"]);
    let range = update.range(&original);
    let values: Vec<f32> = original[range.clone()]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let corpus_root = p["corpus_source"]
        .as_str()
        .map_or(a.study.clone(), PathBuf::from);
    let train = load(&corpus_root, train_seed);
    let val: Vec<_> = validations.iter().map(|&s| load(&corpus_root, s)).collect();
    fs::create_dir(&out)?;
    let corpus_audit = json!({"train":train.audit,"validation":val.iter().map(|v|v.audit.clone()).collect::<Vec<_>>()});
    fs::write(
        out.join("corpus-audit.json"),
        serde_json::to_vec_pretty(&corpus_audit)?,
    )?;
    let device = Default::default();
    let mut results = vec![];
    for seed in seeds {
        for treatment in [false, true] {
            let mut h = Fit::<Autodiff<NdArray>>::new(update, &values, &device);
            assert_eq!(h.values(), values);
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
                    let source = if feeding {
                        &train.feeding
                    } else {
                        &train.combat
                    };
                    for _ in 0..64 {
                        let r = &source[rng.random_range(0..source.len())];
                        labels.push(if treatment && !feeding {
                            r.teacher_kind as i32
                        } else {
                            r.selected_kind as i32
                        });
                        rows.push(r);
                    }
                }
                let y = Tensor::<Autodiff<NdArray>, 2, Int>::from_data(
                    TensorData::new(labels, [128, 1]),
                    &device,
                );
                let loss = burn::tensor::activation::log_softmax(scores(&h, &rows, true), 1)
                    .gather(1, y)
                    .mean()
                    .neg();
                if step % 256 == 0 || step == 1023 {
                    losses.push(
                        json!({"step":step,"loss":loss.clone().into_data().to_vec::<f32>()?[0]}),
                    );
                }
                let gradients = GradientsParams::from_grads(loss.backward(), &h);
                h = optimizer.step(0.005, h, gradients);
            }
            let h = h.valid();
            let fitted = h.values();
            let mut updated = original.clone();
            let encoded: Vec<u8> = fitted.iter().flat_map(|x| x.to_le_bytes()).collect();
            updated[range.clone()].copy_from_slice(&encoded);
            assert_eq!(&original[..range.start], &updated[..range.start]);
            assert_eq!(&original[range.end..], &updated[range.end..]);
            let frozen = FrozenPolicy::from_bytes(&updated)?;
            assert_eq!(frozen.to_bytes(), updated);
            let mut evaluations = vec![];
            let mut pass = true;
            for (&seed, v) in validations.iter().zip(&val) {
                let combat = evaluate(&h, &frozen, &v.combat, treatment);
                let feeding = evaluate(&h, &frozen, &v.feeding, false);
                pass &= combat["agreement"].as_f64().unwrap() >= 0.95
                    && (!treatment || combat["attack_agreement"].as_f64().unwrap() >= 0.9)
                    && feeding["agreement"].as_f64().unwrap() >= 0.99
                    && combat["portable_burn_kind_mismatches"] == 0
                    && feeding["portable_burn_kind_mismatches"] == 0;
                evaluations.push(json!({"seed":seed,"combat":combat,"feeding":feeding}));
            }
            let arm = if treatment { "treatment" } else { "control" };
            let dir = out.join(format!("{seed}-{arm}"));
            fs::create_dir(&dir)?;
            let bytes = match &deployed {
                DeployedPolicy::Composite(c) => CompositePolicy {
                    parent: frozen,
                    utility: c.utility.clone(),
                }
                .to_bytes(),
                _ => unreachable!(),
            };
            fs::write(dir.join("weights.bin"), &bytes)?;
            let export = json!({"schema_version":manifest["schema_version"],"weight_format":manifest["weight_format"],"execution_contract":manifest["execution_contract"],"reference_mind_abi":manifest["reference_mind_abi"],"weight_bytes":bytes.len(),"weights_sha256":sha(&bytes),"source_seed_ledger":audit["seed_ledger"],"parent_manifest_sha256":p["parent_manifest_sha256"],"study_plan_sha256":a.plan_sha256,"qualification":"Native offline diagnostic only; no WASM qualification or ecological promotion","updated_parameter_range_in_parent": [range.start,range.end]});
            fs::write(dir.join("export.json"), serde_json::to_vec_pretty(&export)?)?;
            let result = json!({"sampling_seed":seed,"arm":arm,"updated_parameters":fitted.len(),"all_other_parent_bytes_identical":true,"utility_bytes_identical":true,"weights_sha256":sha(&bytes),"losses":losses,"validation":evaluations,"offline_gate_pass":pass});
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
