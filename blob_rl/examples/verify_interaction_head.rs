//! Independently replay archived ordinary observations through saved frozen fits.
#[path = "interaction_fit/partition.rs"]
mod partition;
use blob_policy::{
    composite::DeployedPolicy,
    runtime::{layer_shapes, FrozenPolicy},
};
use blob_rl::deployed_artifact::{load_deployed_policy, read_bounded, sha};
use clap::Parser;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    study: PathBuf,
}
fn read(path: &Path) -> Value {
    serde_json::from_slice(&read_bounded(path, 128 * 1024 * 1024).unwrap()).unwrap()
}
fn parent(p: &DeployedPolicy) -> &FrozenPolicy {
    p.parent()
}
fn floats(v: &Value) -> Vec<f32> {
    serde_json::from_value(v.clone()).unwrap()
}
fn check(policy: &DeployedPolicy, rows: &[Value], teacher: bool, expected: &Value) -> Value {
    let (mut correct, mut attacks, mut hits) = (0, 0, 0);
    let mut predictions = vec![];
    for r in rows {
        let output = policy
            .forward(&floats(&r["observation"]), &floats(&r["memory"]))
            .unwrap();
        let masks: Vec<bool> = serde_json::from_value(r["legal_kinds"].clone()).unwrap();
        let chosen = (0..10)
            .filter(|&k| masks[k])
            .max_by(|&a, &b| {
                output.kind[a]
                    .total_cmp(&output.kind[b])
                    .then_with(|| b.cmp(&a))
            })
            .unwrap();
        let label = r[if teacher {
            "teacher_kind"
        } else {
            "selected_kind"
        }]
        .as_u64()
        .unwrap() as usize;
        correct += usize::from(chosen == label);
        attacks += usize::from(label == 4);
        hits += usize::from(chosen == 4 && label == 4);
        predictions.push(chosen);
    }
    if !expected.is_null() {
        assert_eq!(expected["rows"], rows.len());
        assert_eq!(expected["correct"], correct);
        assert_eq!(expected["attack_labels"], attacks);
        assert_eq!(expected["attack_correct"], hits);
    }
    json!({"rows":rows.len(),"correct":correct,"attack_labels":attacks,"attack_correct":hits,"predictions":predictions})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    let plan = read(&a.study.join("plan.json"));
    let report = read(&a.study.join("fits/report.json"));
    assert_eq!(
        report["plan_sha256"],
        sha(&fs::read(a.study.join("plan.json"))?)
    );
    let seeds: Vec<u64> = serde_json::from_value(plan["validation_seeds"].clone())?;
    partition::validate(
        &plan,
        &read(&a.study.join("seed-audit.json"))["seed_ledger"],
    )?;
    let (original, _) = load_deployed_policy(
        Path::new(plan["parent"].as_str().unwrap()),
        plan["parent_manifest_sha256"].as_str().unwrap(),
        &seeds,
    )?;
    let old = parent(&original).to_bytes();
    let mut evidence = vec![];
    for result in report["results"].as_array().unwrap() {
        let arm = result["arm"].as_str().unwrap();
        let seed = result["sampling_seed"].as_u64().unwrap();
        let dir = a.study.join(format!("fits/{seed}-{arm}"));
        assert_eq!(*result, read(&dir.join("result.json")));
        let manifest_bytes = fs::read(dir.join("export.json"))?;
        let (policy, manifest) = load_deployed_policy(&dir, &sha(&manifest_bytes), &seeds)?;
        assert_eq!(manifest["weights_sha256"], result["weights_sha256"]);
        let new = parent(&policy).to_bytes();
        let relational = plan["intervention"] == "relational_residual";
        if relational {
            match (&original, &policy) {
                (DeployedPolicy::Composite(a), DeployedPolicy::Relational(b)) => {
                    assert_eq!(a.to_bytes(), b.base.to_bytes())
                }
                _ => panic!("requires exact nested composite"),
            }
            assert_eq!(
                result["added_parameters"],
                blob_policy::relational::PARAMETERS
            );
        } else {
            let start = manifest["updated_parameter_range_in_parent"][0]
                .as_u64()
                .unwrap() as usize;
            let end = manifest["updated_parameter_range_in_parent"][1]
                .as_u64()
                .unwrap() as usize;
            let shapes = layer_shapes(256, 128, 128)?;
            let (first, last) = if plan["intervention"] == "slot_adapter" {
                (10, 12)
            } else {
                (7, 8)
            };
            let size = |ss: &[(&str, usize, usize)]| {
                ss.iter().map(|(_, i, o)| (i * o + o) * 4).sum::<usize>()
            };
            assert_eq!(start, 20 + size(&shapes[..first]));
            assert_eq!(end, start + size(&shapes[first..last]));
            assert_eq!(old.len(), new.len());
            assert_eq!(&old[..start], &new[..start]);
            assert_eq!(&old[end..], &new[end..]);
            match (&original, &policy) {
                (DeployedPolicy::Composite(a), DeployedPolicy::Composite(b)) => {
                    assert_eq!(
                        a.utility
                            .values()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>(),
                        b.utility
                            .values()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>()
                    )
                }
                _ => panic!("lost composite"),
            };
        }
        let mut evaluations = vec![];
        for (index, seed) in seeds.iter().enumerate() {
            let root = partition::root(&plan, &a.study, *seed)?.join(seed.to_string());
            let combat = read(&root.join("combat.json"))["rows"]
                .as_array()
                .unwrap()
                .clone();
            let mut feeding = vec![];
            for layout in ["Line", "Checkerboard", "Ring", "LooseRandom", "Random"] {
                feeding.extend(
                    read(&root.join(format!("feeding-{layout}.json")))["rows"]
                        .as_array()
                        .unwrap()
                        .clone(),
                );
            }
            let expected = &result["validation"][index];
            assert_eq!(expected["seed"], *seed);
            evaluations.push(json!({"seed":seed,"combat":check(&policy,&combat,arm=="treatment",&expected["combat"]),"feeding":check(&policy,&feeding,false,&expected["feeding"])}));
        }
        let training = if relational {
            let train_seeds = partition::training_seeds(&plan)?;
            let mut combat = vec![];
            let mut feeding = vec![];
            for &seed in &train_seeds {
                let root = partition::root(&plan, &a.study, seed)?.join(seed.to_string());
                combat.extend(
                    read(&root.join("combat.json"))["rows"]
                        .as_array()
                        .unwrap()
                        .clone(),
                );
                for layout in ["Line", "Checkerboard", "Ring", "LooseRandom", "Random"] {
                    feeding.extend(
                        read(&root.join(format!("feeding-{layout}.json")))["rows"]
                            .as_array()
                            .unwrap()
                            .clone(),
                    );
                }
            }
            let mut report = json!({"seed":plan["training_seed"],"combat":check(&policy,&combat,arm=="treatment",&Value::Null),"feeding":check(&policy,&feeding,false,&Value::Null)});
            if train_seeds.len() > 1 {
                report["seeds"] = json!(train_seeds);
            }
            report
        } else {
            Value::Null
        };
        evidence.push(json!({"arm":arm,"sampling_seed":seed,"manifest_sha256":sha(&manifest_bytes),"weights_sha256":manifest["weights_sha256"],"only_declared_head_changed":!relational,"parent_and_utility_bytes_identical":relational,"training":training,"utility_identical":true,"validation":evaluations}));
    }
    let output = a.study.join("portable-replay.json");
    if output.exists() {
        return Err("immutable replay exists".into());
    }
    fs::write(
        output,
        serde_json::to_vec(
            &json!({"complete":true,"fit_report_sha256":sha(&fs::read(a.study.join("fits/report.json"))?),"results":evidence}),
        )?,
    )?;
    Ok(())
}
