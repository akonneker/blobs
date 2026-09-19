//! Small nonzero synthetic network for repeatable CI; never a trained policy.
use blob_policy::runtime::*;
use sha2::{Digest, Sha256};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::path::PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected new output directory")?,
    );
    let composite = std::env::args().nth(2).as_deref() == Some("--composite");
    let layers = layer_shapes(16, 8, 4)?
        .into_iter()
        .enumerate()
        .map(|(n, (name, i, o))| {
            Linear::new(
                i,
                o,
                (0..i * o)
                    .map(|j| ((j * 13 + n * 7) % 31) as f32 * 0.002 - 0.03)
                    .collect(),
                (0..o)
                    .map(|j| {
                        if composite && name.ends_with("action_kind_head") && j == 3 {
                            10.
                        } else {
                            (j % 5) as f32 * 0.003
                        }
                    })
                    .collect(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let parent = FrozenPolicy::new(16, 8, 4, layers)?;
    let (bytes, contract) = if composite {
        use blob_policy::composite::*;
        (
            CompositePolicy {
                parent,
                utility: FrozenUtility::new(
                    (0..UTILITY_PARAMETERS)
                        .map(|i| ((i * 7 % 23) as f32 - 11.) * 0.002)
                        .collect(),
                )?,
            }
            .to_bytes(),
            COMPOSITE_EXECUTION_CONTRACT,
        )
    } else {
        (parent.to_bytes(), EXECUTION_CONTRACT)
    };
    let manifest = serde_json::json!({"schema_version":POLICY_EXPORT_SCHEMA_VERSION,"weight_format":POLICY_WEIGHT_FORMAT_VERSION,"execution_contract":contract,"source_kind":"synthetic CI network, not learned weights","reference_mind_abi":blob_interface::abi::reference_mind_abi_hash(),"weights_sha256":format!("{:x}",Sha256::digest(&bytes)),"weight_bytes":bytes.len()});
    std::fs::create_dir(&output)?;
    std::fs::write(output.join("weights.bin"), bytes)?;
    std::fs::write(
        output.join("export.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
