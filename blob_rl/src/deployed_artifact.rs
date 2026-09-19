//! Bounded, hash-bound deployment loading shared by evaluation commands.
use blob_policy::{
    composite::DeployedPolicy,
    runtime::{MAX_WEIGHT_BYTES, POLICY_EXPORT_SCHEMA_VERSION, POLICY_WEIGHT_FORMAT_VERSION},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path};
pub fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("input exceeds byte budget".into());
    }
    Ok(bytes)
}
pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn load_deployed_policy(
    export: &Path,
    expected_manifest_sha256: &str,
    seeds: &[u64],
) -> Result<(DeployedPolicy, Value), Box<dyn std::error::Error>> {
    if seeds.is_empty()
        || seeds
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != seeds.len()
    {
        return Err("evaluation seeds must be nonempty and unique".into());
    }
    let manifest_bytes = read_bounded(&export.join("export.json"), 1024 * 1024)?;
    if sha(&manifest_bytes) != expected_manifest_sha256 {
        return Err("export manifest SHA mismatch".into());
    }
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let weights = read_bounded(&export.join("weights.bin"), MAX_WEIGHT_BYTES)?;
    let policy = DeployedPolicy::from_bytes(&weights)?;
    if manifest["schema_version"] != POLICY_EXPORT_SCHEMA_VERSION
        || manifest["weight_format"] != POLICY_WEIGHT_FORMAT_VERSION
        || manifest["weights_sha256"] != sha(&weights)
        || manifest["weight_bytes"] != weights.len()
        || manifest["execution_contract"] != policy.execution_contract()
        || manifest["reference_mind_abi"] != json!(blob_interface::abi::reference_mind_abi_hash())
    {
        return Err("deployment identity mismatch".into());
    }
    for seed in seeds {
        if [1434999901, 1434999902].contains(seed) {
            return Err("reserved feeding confirmation seed".into());
        }
        for pointer in [
            "/source_seed_ledger/confirmation",
            "/source_training_config/confirmation_seeds",
        ] {
            if let Some(value) = manifest.pointer(pointer) {
                for reserved in value.as_array().ok_or("malformed confirmation ledger")? {
                    if reserved.as_u64().ok_or("malformed confirmation seed")? == *seed {
                        return Err("evaluation seed is reserved for confirmation".into());
                    }
                }
            }
        }
    }
    Ok((policy, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_policy::runtime::{layer_shapes, FrozenPolicy, Linear, EXECUTION_CONTRACT};
    fn fixture(path: &Path) -> Value {
        let policy = FrozenPolicy::new(
            4,
            3,
            2,
            layer_shapes(4, 3, 2)
                .unwrap()
                .into_iter()
                .map(|(_, i, o)| Linear::new(i, o, vec![0.; i * o], vec![0.; o]).unwrap())
                .collect(),
        )
        .unwrap();
        let weights = policy.to_bytes();
        fs::write(path.join("weights.bin"), &weights).unwrap();
        json!({"schema_version":POLICY_EXPORT_SCHEMA_VERSION,"weight_format":POLICY_WEIGHT_FORMAT_VERSION,"execution_contract":EXECUTION_CONTRACT,"weights_sha256":sha(&weights),"weight_bytes":weights.len(),"reference_mind_abi":blob_interface::abi::reference_mind_abi_hash(),"source_seed_ledger":{"confirmation":[77]},"source_training_config":{"confirmation_seeds":[88]}})
    }
    fn publish(path: &Path, manifest: &Value) -> String {
        let bytes = serde_json::to_vec(manifest).unwrap();
        fs::write(path.join("export.json"), &bytes).unwrap();
        sha(&bytes)
    }
    #[test]
    fn deployed_loader_rejects_identity_and_weight_mutations() {
        let dir = tempfile::tempdir().unwrap();
        let m = fixture(dir.path());
        let hash = publish(dir.path(), &m);
        assert!(load_deployed_policy(dir.path(), &hash, &[99]).is_ok());
        assert!(load_deployed_policy(dir.path(), "wrong", &[99]).is_err());
        let mut altered = m.clone();
        altered["execution_contract"] = json!("wrong");
        let bad = publish(dir.path(), &altered);
        assert!(load_deployed_policy(dir.path(), &bad, &[99]).is_err());
        publish(dir.path(), &m);
        let path = dir.path().join("weights.bin");
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(path, bytes).unwrap();
        assert!(load_deployed_policy(dir.path(), &hash, &[99]).is_err());
    }
    #[test]
    fn deployed_loader_rejects_reserved_duplicate_and_malformed_seeds() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = fixture(dir.path());
        let hash = publish(dir.path(), &m);
        for seeds in [
            vec![],
            vec![99, 99],
            vec![77],
            vec![88],
            vec![1434999901],
            vec![1434999902],
        ] {
            assert!(load_deployed_policy(dir.path(), &hash, &seeds).is_err());
        }
        m["source_seed_ledger"]["confirmation"] = json!(["not a seed"]);
        let hash = publish(dir.path(), &m);
        assert!(load_deployed_policy(dir.path(), &hash, &[99]).is_err());
    }
}
