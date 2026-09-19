//! Freeze the six archived utility fits without retraining or selecting a best seed.
#[path = "feeding_utility/data.rs"]
mod data;
#[path = "feeding_utility/evaluate.rs"]
mod evaluate;
#[path = "feeding_utility/model.rs"]
#[allow(dead_code)]
mod model;
use blob_policy::{composite::*, runtime::*};
use blob_rl::behavior_cloning::seed_ledger::SeedLedger;
use burn::{
    backend::NdArray,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use clap::Parser;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    study: PathBuf,
    #[arg(long)]
    report_sha256: String,
    #[arg(long)]
    parent_export: PathBuf,
    #[arg(long)]
    pairs_root: PathBuf,
    #[arg(long)]
    fresh_pairs_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(path: &Path, limit: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("artifact exceeds byte budget".into());
    }
    Ok(bytes)
}
fn compare(
    utility: &model::Utility<NdArray>,
    frozen: &FrozenUtility,
    rows: &[data::Row],
    control: bool,
) -> Value {
    let expected = evaluate::evaluate(utility, rows, control);
    let mut mismatches = 0;
    let mut max_error = 0_f32;
    let mut correct = 0;
    let mut active_correct = 0;
    let mut retention_correct = 0;
    for chunk in rows.chunks(128) {
        let (local, legal) = model::batch(chunk, &Default::default());
        let burn = utility
            .forward(local, legal)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (index, row) in chunk.iter().enumerate() {
            let scores = frozen.forward(&row.local, &row.legal).unwrap();
            let native = &burn[index * 32..(index + 1) * 32];
            for i in 0..32 {
                if row.legal[i] {
                    max_error = max_error.max((scores[i] - native[i]).abs());
                }
            }
            let predicted = evaluate::predict(&scores, row);
            mismatches += usize::from(predicted != evaluate::predict(native, row));
            let hit = predicted == if control { row.control } else { row.treatment };
            correct += usize::from(hit);
            active_correct += usize::from(hit && row.active);
            retention_correct += usize::from(hit && !row.active);
        }
    }
    assert_eq!(
        mismatches, 0,
        "portable utility changes a development target"
    );
    assert_eq!(correct, expected["correct"].as_u64().unwrap() as usize);
    assert!(expected["gate_pass"].as_bool().unwrap());
    json!({"rows":rows.len(),"sampled_choice_mismatches":mismatches,"max_legal_logit_absolute_difference":max_error,"correct":correct,"active_correct":active_correct,"retention_correct":retention_correct,"burn_evaluation":expected})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.exists() {
        return Err("immutable output already exists".into());
    }
    let report_bytes = read(&args.study.join("report.json"), 1024 * 1024)?;
    if hash(&report_bytes) != args.report_sha256 {
        return Err("study report SHA mismatch".into());
    }
    let report: Value = serde_json::from_slice(&report_bytes)?;
    let plan = &report["plan"];
    if report["complete"] != true
        || plan["revision"] != 3
        || plan["objectives"] != json!(["QuantileInterval"])
    {
        return Err("requires completed archived interval study".into());
    }
    let parent_manifest_bytes = read(&args.parent_export.join("export.json"), 1024 * 1024)?;
    let parent_manifest: Value = serde_json::from_slice(&parent_manifest_bytes)?;
    let parent_bytes = read(
        &args.parent_export.join("weights.bin"),
        MAX_WEIGHT_BYTES as u64,
    )?;
    if parent_manifest["execution_contract"] != EXECUTION_CONTRACT
        || parent_manifest["weights_sha256"] != hash(&parent_bytes)
        || parent_manifest["source_artifact_sha256"] != plan["parent_sha256"]
    {
        return Err("parent export identity mismatch".into());
    }
    let parent = FrozenPolicy::from_bytes(&parent_bytes)?;
    let (old, old_audit) = data::load(&args.pairs_root)?;
    let (fresh, fresh_audit) = data::load(&args.fresh_pairs_root)?;
    if old_audit != plan["old_audit"] || fresh_audit != plan["fresh_audit"] {
        return Err("study corpus audit changed".into());
    }
    let old: Vec<_> = old.into_iter().filter(|r| r.seed == 1433000202).collect();
    let mut ledger: SeedLedger =
        serde_json::from_value(plan["inherited_seed_ledger"]["ledger"].clone())?;
    ledger
        .training
        .extend([1433000101, 1435400101, 1435400301, 1435400302, 1435400303]);
    ledger
        .validation
        .extend([1433000202, 1435400201, 1435400202]);
    // Earlier representation studies informed architecture selection, even
    // though their weights are not ancestors of this utility record.
    ledger.validation.extend([
        1435000101, 1435000202, 1435000303, 1435100101, 1435100202, 1435100301, 1435100302,
        1435100303, 1435200101, 1435200202, 1435200301, 1435200302, 1435200303,
    ]);
    ledger
        .validation
        .extend(blob_policy::qualification::DEVELOPMENT_SEEDS);
    ledger.confirmation.extend([1434999901, 1434999902]);
    ledger.validate()?;
    blob_policy::qualification::check_confirmation_reservations(
        ledger.confirmation.iter().copied(),
    )?;
    let fits = report["results"].as_array().ok_or("missing fits")?;
    let mut seen = std::collections::BTreeSet::new();
    let mut outputs = Vec::new();
    let sources: &[(&str, &[u8])] = &[
        ("exporter.rs", include_bytes!("export_feeding_composite.rs")),
        ("model.rs", include_bytes!("feeding_utility/model.rs")),
        ("data.rs", include_bytes!("feeding_utility/data.rs")),
        ("evaluate.rs", include_bytes!("feeding_utility/evaluate.rs")),
        (
            "composite.rs",
            include_bytes!("../../blob_policy/src/composite.rs"),
        ),
        (
            "runtime.rs",
            include_bytes!("../../blob_policy/src/runtime.rs"),
        ),
        (
            "sampling.rs",
            include_bytes!("../../blob_policy/src/sampling.rs"),
        ),
        ("Cargo.lock", include_bytes!("../../Cargo.lock")),
    ];
    let source_hashes: std::collections::BTreeMap<_, _> = sources
        .iter()
        .map(|(name, bytes)| (*name, hash(bytes)))
        .collect();
    for fit in fits {
        let seed = fit["initialization_seed"]
            .as_u64()
            .ok_or("missing initialization")?;
        let arm = fit["arm"].as_str().ok_or("missing arm")?;
        if ![1435400301, 1435400302, 1435400303].contains(&seed)
            || !["control", "treatment"].contains(&arm)
            || !seen.insert((seed, arm.to_owned()))
        {
            return Err("invalid fit grid".into());
        }
        let record = &fit["record"];
        let filename = record["file"].as_str().ok_or("missing record")?;
        if Path::new(filename).file_name().and_then(|s| s.to_str()) != Some(filename) {
            return Err("invalid record path".into());
        }
        let bytes = read(&args.study.join(filename), 1024 * 1024)?;
        if record["sha256"] != hash(&bytes)
            || record["precision"] != "f32"
            || record["reload_parameter_bits_identical"] != true
        {
            return Err("record identity mismatch".into());
        }
        let snapshot = tempfile::tempdir()?;
        fs::write(snapshot.path().join("utility.mpk"), &bytes)?;
        let utility = model::Utility::<NdArray>::new(&Default::default()).load_file(
            snapshot.path().join("utility"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            &Default::default(),
        )?;
        let frozen = FrozenUtility::new(
            utility
                .parameter_bits()
                .into_iter()
                .map(f32::from_bits)
                .collect(),
        )?;
        let old_fidelity = compare(&utility, &frozen, &old, arm == "control");
        let fresh_fidelity = compare(&utility, &frozen, &fresh, arm == "control");
        if old_fidelity["burn_evaluation"] != fit["old_development"]
            || fresh_fidelity["burn_evaluation"] != fit["fresh_development"]
        {
            return Err("record no longer reproduces archived evaluation".into());
        }
        let composite = CompositePolicy {
            parent: parent.clone(),
            utility: frozen,
        };
        let weights = composite.to_bytes();
        let decoded = CompositePolicy::from_bytes(&weights)?;
        assert_eq!(decoded.parent.to_bytes(), parent_bytes);
        assert_eq!(decoded.to_bytes(), weights);
        let manifest = json!({"schema_version":POLICY_EXPORT_SCHEMA_VERSION,"weight_format":POLICY_WEIGHT_FORMAT_VERSION,"execution_contract":COMPOSITE_EXECUTION_CONTRACT,"reference_mind_abi":blob_interface::abi::reference_mind_abi_hash(),"weights_sha256":hash(&weights),"weight_bytes":weights.len(),"parent_export_sha256":hash(&parent_manifest_bytes),"parent_weights_sha256":hash(&parent_bytes),"source_artifact_sha256":plan["parent_sha256"],"source_utility_record_sha256":hash(&bytes),"source_study_report_sha256":args.report_sha256,"source_seed_ledger":ledger,"initialization_seed":seed,"arm":arm,"export_build":env!("BLOB_CODE_REVISION"),"parent_parameters_bit_identical":true,"old_fidelity":old_fidelity,"fresh_fidelity":fresh_fidelity,"qualification":"Conditional recorded-frontier fidelity only. Composite reserves parent signal cost; live retention, WASM parity and ecology pending."});
        let mut manifest = manifest;
        manifest["export_source_sha256"] = serde_json::to_value(&source_hashes)?;
        outputs.push((format!("{seed}-{arm}"), weights, manifest));
    }
    if seen.len() != 6 {
        return Err("incomplete fit grid".into());
    }
    fs::create_dir(&args.output)?;
    fs::create_dir(args.output.join("sources"))?;
    for (name, bytes) in sources {
        fs::write(args.output.join("sources").join(name), bytes)?;
    }
    for (name, weights, manifest) in outputs {
        let path = args.output.join(name);
        fs::create_dir(&path)?;
        fs::write(path.join("weights.bin"), weights)?;
        fs::write(
            path.join("export.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
    }
    println!(
        "exported all six composite fits; exact sampled choices on {} recorded Move frontiers each",
        old.len() + fresh.len()
    );
    Ok(())
}
