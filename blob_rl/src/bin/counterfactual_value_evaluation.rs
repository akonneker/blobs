//! Apply an explicit lexicographic-Pareto value objective to branch evidence.

use std::path::PathBuf;

use blob_rl::counterfactual_branch::{load_counterfactual_branch_artifact, BranchContinuation};
use blob_rl::counterfactual_value::{
    publish_counterfactual_value_artifact, CounterfactualValueArtifact, CounterfactualValueOptions,
    EnergyValuation, ValuePerspective,
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PerspectiveArg {
    Cell,
    Colony,
    Conservative,
}

impl From<PerspectiveArg> for ValuePerspective {
    fn from(value: PerspectiveArg) -> Self {
        match value {
            PerspectiveArg::Cell => Self::Cell,
            PerspectiveArg::Colony => Self::Colony,
            PerspectiveArg::Conservative => Self::Conservative,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ContinuationArg {
    FrozenPolicy,
    Teacher,
}

impl From<ContinuationArg> for BranchContinuation {
    fn from(value: ContinuationArg) -> Self {
        match value {
            ContinuationArg::FrozenPolicy => Self::FrozenPolicy,
            ContinuationArg::Teacher => Self::Teacher,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "counterfactual-value-evaluation",
    about = "Classify immutable counterfactual branches under explicit Pareto objectives"
)]
struct Args {
    /// Verified schema-3 counterfactual branch artifact.
    #[arg(long)]
    counterfactual: PathBuf,

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "cell,colony,conservative"
    )]
    perspectives: Vec<PerspectiveArg>,

    /// Simulation-clock horizons already present in the branch artifact.
    #[arg(long, value_delimiter = ',', required = true)]
    horizon_quanta: Vec<u64>,

    /// Continuations that must agree for a robust preference.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "frozen-policy,teacher"
    )]
    continuations: Vec<ContinuationArg>,

    /// Exact energy-compartment weights. One million means full weight.
    #[arg(long, default_value_t = 1_000_000)]
    core_weight_ppm: u32,

    #[arg(long, default_value_t = 1_000_000)]
    assimilated_weight_ppm: u32,

    /// Gut contents default to no biological value.
    #[arg(long, default_value_t = 0)]
    gut_weight_ppm: u32,

    /// New immutable value-analysis artifact path.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let source = load_counterfactual_branch_artifact(&args.counterfactual)
        .unwrap_or_else(|error| panic!("invalid counterfactual source: {error}"));
    let options = CounterfactualValueOptions {
        perspectives: args.perspectives.into_iter().map(Into::into).collect(),
        horizon_quanta: args.horizon_quanta,
        continuations: args.continuations.into_iter().map(Into::into).collect(),
        energy: EnergyValuation {
            core_weight_ppm: args.core_weight_ppm,
            assimilated_weight_ppm: args.assimilated_weight_ppm,
            gut_weight_ppm: args.gut_weight_ppm,
        },
    };
    let artifact = CounterfactualValueArtifact::new(&source, options)
        .unwrap_or_else(|error| panic!("counterfactual value evaluation failed: {error}"));
    publish_counterfactual_value_artifact(&args.output, &source, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish value artifact: {error}"));
    println!(
        "Published {} objective evaluations to {}",
        artifact.report.evaluations.len(),
        args.output.display()
    );
    for evaluation in &artifact.report.evaluations {
        let frontier_actions = evaluation
            .states
            .iter()
            .map(|state| state.pareto_frontier.len())
            .sum::<usize>();
        let max_frontier = evaluation
            .states
            .iter()
            .map(|state| state.pareto_frontier.len())
            .max()
            .unwrap_or(0);
        let max_archetypes = evaluation
            .states
            .iter()
            .map(|state| state.pareto_archetypes.len())
            .max()
            .unwrap_or(0);
        println!(
            "  {:?} +{}q states T/P/tie/inc {}/{}/{}/{}, seeds {}/{}/{}/{}, frontier total/max {}/{}, max archetypes {}",
            evaluation.perspective,
            evaluation.requested_elapsed_quanta,
            evaluation.state_counts.teacher,
            evaluation.state_counts.policy,
            evaluation.state_counts.ties,
            evaluation.state_counts.incomparable,
            evaluation.seed_counts.teacher,
            evaluation.seed_counts.policy,
            evaluation.seed_counts.ties,
            evaluation.seed_counts.incomparable,
            frontier_actions,
            max_frontier,
            max_archetypes,
        );
    }
    println!("Artifact hash: {}", artifact.artifact_hash);
}
