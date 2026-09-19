//! Verified real-observation audit before utility transfer training.
#[path = "feeding_utility/data.rs"]
mod data;
use blob_rl::observation::SLOT_FEATURES;
use clap::Parser;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    pairs_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    fs::create_dir(&args.output)?;
    let (rows, audit) = data::load(&args.pairs_root)?;
    let mut food_oracle = 0;
    let mut uniform = 0;
    let mut canonical_teacher_mismatches = 0;
    let mut unique = std::collections::BTreeSet::new();
    for row in &rows {
        let legal: Vec<_> = row
            .geometry_order
            .iter()
            .copied()
            .filter(|i| row.legal[*i])
            .collect();
        let choice = legal[((u128::from(row.word) * legal.len() as u128) >> 64) as usize];
        uniform += usize::from(choice == row.treatment);
        let best = legal
            .iter()
            .filter(|i| row.local[**i * SLOT_FEATURES + 13] == 0.)
            .map(|i| row.local[*i * SLOT_FEATURES + 8])
            .fold(f32::NEG_INFINITY, f32::max);
        let candidates: Vec<_> = legal
            .iter()
            .copied()
            .filter(|i| {
                row.local[*i * SLOT_FEATURES + 13] == 0.
                    && row.local[*i * SLOT_FEATURES + 8] == best
            })
            .collect();
        if !candidates.is_empty() {
            let choice =
                candidates[((u128::from(row.word) * candidates.len() as u128) >> 64) as usize];
            food_oracle += usize::from(choice == row.treatment);
            canonical_teacher_mismatches += usize::from(Some(choice) != row.teacher);
        }
        unique.insert(row.local.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
    let report = json!({"audit":audit,"uniform_agreement":uniform,"plant_best_vacant_agreement":food_oracle,"plant_oracle_teacher_target_mismatches":canonical_teacher_mismatches,"unique_random_free_slot_inputs":unique.len(),"control_agreement":rows.iter().filter(|r|r.control==r.treatment).count(),"active_rows":rows.iter().filter(|r|r.active).count(),"source_sha256":format!("{:x}",Sha256::digest(include_bytes!("feeding_utility_audit.rs"))),"data_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("feeding_utility/data.rs")))});
    fs::write(
        args.output.join("audit.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
