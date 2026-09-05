use std::path::PathBuf;

use blob_rl::demonstration::{load_demonstrations, verify_feeding_correction_pair};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "feeding-correction-pair-verify",
    about = "Verify that paired feeding datasets differ only at active labels"
)]
struct Args {
    #[arg(long)]
    control: PathBuf,

    #[arg(long)]
    treatment: PathBuf,
}

fn main() {
    let args = Args::parse();
    let control = load_demonstrations(&args.control)
        .unwrap_or_else(|error| panic!("invalid control dataset: {error}"));
    let treatment = load_demonstrations(&args.treatment)
        .unwrap_or_else(|error| panic!("invalid treatment dataset: {error}"));
    let eligible = verify_feeding_correction_pair(&control, &treatment)
        .unwrap_or_else(|error| panic!("invalid feeding-correction pair: {error}"));
    println!(
        "Verified {} matched samples with {} active feeding label changes",
        control.manifest.samples, eligible
    );
}
