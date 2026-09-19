//! Explicit simulation-seed partitions shared by fitting and replay commands.
use serde_json::Value;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
pub fn training_seeds(plan: &Value) -> Result<Vec<u64>, String> {
    let seeds = if let Some(value) = plan.get("training_seeds") {
        serde_json::from_value::<Vec<u64>>(value.clone()).map_err(|e| e.to_string())?
    } else {
        vec![plan["training_seed"]
            .as_u64()
            .ok_or("missing training seed")?]
    };
    if seeds.is_empty() || seeds.iter().collect::<BTreeSet<_>>().len() != seeds.len() {
        return Err("training seeds must be nonempty and unique".into());
    }
    if plan["training_seed"].as_u64() != Some(seeds[0]) {
        return Err("primary training seed must be first partition".into());
    }
    Ok(seeds)
}
pub fn validate(plan: &Value, ledger: &Value) -> Result<(), String> {
    let training = training_seeds(plan)?;
    let validation: Vec<u64> =
        serde_json::from_value(plan["validation_seeds"].clone()).map_err(|e| e.to_string())?;
    let sampling: Vec<u64> =
        serde_json::from_value(plan["sampling_seeds"].clone()).map_err(|e| e.to_string())?;
    let train_ledger: Vec<u64> =
        serde_json::from_value(ledger["training"].clone()).map_err(|e| e.to_string())?;
    let val_ledger: Vec<u64> =
        serde_json::from_value(ledger["validation"].clone()).map_err(|e| e.to_string())?;
    let confirmation: Vec<u64> =
        serde_json::from_value(ledger["confirmation"].clone()).map_err(|e| e.to_string())?;
    if validation.is_empty() || validation.iter().collect::<BTreeSet<_>>().len() != validation.len()
    {
        return Err("validation seeds must be nonempty and unique".into());
    }
    for s in training.iter().chain(&sampling) {
        if !train_ledger.contains(s) || val_ledger.contains(s) || confirmation.contains(s) {
            return Err("training or sampling seed violates inherited partition".into());
        }
    }
    for s in &validation {
        if !val_ledger.contains(s) || train_ledger.contains(s) || confirmation.contains(s) {
            return Err("validation seed violates inherited partition".into());
        }
    }
    for s in training.iter().chain(&validation) {
        root(plan, Path::new("."), *s)?;
    }
    Ok(())
}
pub fn root(plan: &Value, study: &Path, seed: u64) -> Result<PathBuf, String> {
    if let Some(roots) = plan.get("corpus_roots") {
        let path = roots
            .get(seed.to_string())
            .and_then(Value::as_str)
            .filter(|x| !x.is_empty())
            .ok_or("missing explicit corpus root")?;
        Ok(PathBuf::from(path))
    } else {
        Ok(plan["corpus_source"]
            .as_str()
            .map_or(study.to_path_buf(), PathBuf::from))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn partition_order_and_legacy_fallback_are_explicit() {
        let mut p = json!({"training_seed":1,"validation_seeds":[9],"sampling_seeds":[8],"corpus_source":"old"});
        let l = json!({"training":[1,2,8],"validation":[9],"confirmation":[10]});
        validate(&p, &l).unwrap();
        assert_eq!(training_seeds(&p).unwrap(), vec![1]);
        p["training_seeds"] = json!([1, 2]);
        p["corpus_roots"] = json!({"1":"old","2":"new","9":"old"});
        validate(&p, &l).unwrap();
        assert_eq!(
            root(&p, Path::new("ignored"), 2).unwrap(),
            PathBuf::from("new")
        );
        p["corpus_roots"].as_object_mut().unwrap().remove("2");
        assert!(validate(&p, &l).is_err());
    }
    #[test]
    fn rejects_empty_duplicate_reclassified_and_reserved_training_seeds() {
        let p = json!({"training_seed":1,"training_seeds":[1,2],"validation_seeds":[9],"sampling_seeds":[8]});
        let l = json!({"training":[1,2,8],"validation":[9],"confirmation":[10]});
        for seeds in [
            json!([]),
            json!([1, 1]),
            json!([2, 1]),
            json!([1, 9]),
            json!([1, 10]),
        ] {
            let mut bad = p.clone();
            bad["training_seeds"] = seeds;
            assert!(validate(&bad, &l).is_err());
        }
        let mut bad = l.clone();
        bad["training"] = json!([1, 2, 8, 9]);
        assert!(validate(&p, &bad).is_err());
    }
}
