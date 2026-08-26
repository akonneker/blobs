//! Publish deterministic ecological and combat seam characterizations.

use std::path::PathBuf;

use blob_rl::config::TrainingConfig;
use blob_rl::ecological_characterization::{
    characterize_ecology, publish_ecological_characterization, EcologicalCharacterizationOptions,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_ecological_characterization",
    about = "Measure lifecycle, travel, food, terrain, signal, reproduction, and combat seams under canonical rules"
)]
struct Args {
    /// Training TOML supplying the scenario and canonical rules.
    #[arg(long)]
    config: PathBuf,

    /// Unique world-generation seeds, accepted as repeated or comma-delimited values.
    #[arg(long, required = true, value_delimiter = ',')]
    seeds: Vec<u64>,

    /// Safety cap for repeated micro-world actions.
    #[arg(long, default_value_t = 1_000_000)]
    max_micro_actions: usize,

    /// New immutable JSON report path.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let config = TrainingConfig::from_file(&args.config.to_string_lossy())
        .unwrap_or_else(|error| panic!("failed to load characterization config: {error}"));
    let report = characterize_ecology(
        &config.env,
        &EcologicalCharacterizationOptions {
            seeds: args.seeds,
            max_micro_actions: args.max_micro_actions,
        },
    )
    .unwrap_or_else(|error| panic!("ecological characterization failed: {error}"));
    let output = publish_ecological_characterization(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish characterization: {error}"));
    let quanta_per_unit = report.rules.time.arithmetic_quanta_per_unit as f64;

    println!("Ecological characterization: {}", output.display());
    println!(
        "  stationary lifetime: {}",
        report.stationary_lifetime_quanta.map_or_else(
            || "unbounded".into(),
            |value| format!(
                "{value} quanta ({:.3} time units)",
                value as f64 / quanta_per_unit
            )
        )
    );
    for travel in &report.travel {
        println!(
            "  {:?} travel: {} moves / {:.3} distance units, death at {}",
            travel.effort,
            travel.completed_moves,
            travel.completed_distance_q10 as f64 / 1024.0,
            travel.death_time_quanta.map_or_else(
                || "unbounded or censored".into(),
                |value| format!("{:.3} time units", value as f64 / quanta_per_unit),
            ),
        );
    }
    println!(
        "  plant distance: mean {:?}, p90 {:?} movement units",
        report.distance_to_plants.weighted_distance_units.mean,
        report.distance_to_plants.weighted_distance_units.p90,
    );
    for volley in &report.attack_volleys {
        println!(
            "  {:?} full-strength volley: unguarded {:?}, guarded {:?} attackers",
            volley.effort, volley.unguarded_attackers_required, volley.guarded_attackers_required,
        );
    }
    println!(
        "  feeding cycle: consumed {}, energy {:?} after digestion at {:?} quanta",
        report.feeding_cycle.consumed_energy,
        report.feeding_cycle.energy_after_digestion,
        report.feeding_cycle.digestion_completion_quanta,
    );
    println!(
        "  reproduction break-even: commit {}, successful parent {:?} energy",
        report.reproduction_break_even.minimum_commit_energy,
        report
            .reproduction_break_even
            .minimum_successful_parent_energy,
    );
    println!(
        "  terrain cycle: excavated {}, deposited {}, final energy {:?}",
        report.terrain_cycle.excavation_succeeded,
        report.terrain_cycle.deposition_succeeded,
        report.terrain_cycle.final_energy,
    );
    println!(
        "  signals: quantum {}, {} neighbor slots / {} directed edges",
        report.signals.emission_quantum,
        report.signals.neighbor_observation_slots,
        report.signals.directed_observation_edges,
    );
    for trial in &report.signals.impulse_trials {
        println!(
            "    {}x: accepted {}, field {} -> {}, half {:?}, extinct {:?} quanta",
            trial.strength_multiplier,
            trial.accepted,
            trial.field_energy_after_commit,
            trial.field_energy_after_completion,
            trial.time_to_half_energy_quanta,
            trial.time_to_extinction_quanta,
        );
    }
    println!("    mass-shedding travel (0x is duration-matched wait):");
    for trial in &report.signals.mass_shedding_trials {
        println!(
            "      {}x: accepted {}, mass {:?}, first move effort/duration {:?}/{:?}, {} moves in {} quanta, stationary death {:?}",
            trial.strength_multiplier,
            trial.preparation_accepted,
            trial.total_mass_after_preparation,
            trial.first_move_effort_spent,
            trial.first_move_duration_quanta,
            trial.completed_moves,
            trial.movement_elapsed_quanta,
            trial.stationary_death_time_quanta,
        );
    }
    println!(
        "    terrain disruption: {}",
        report.signals.terrain_disruption.as_ref().map_or_else(
            || "unavailable".into(),
            |probe| format!(
                "{} energy erased after {} decayed",
                probe.erased_by_terrain_action, probe.decayed_during_terrain_action
            )
        )
    );
    println!(
        "  sustained plant siege: {:?} of {} local attackers required",
        report.sustained_siege.minimum_attackers_to_kill,
        report.sustained_siege.maximum_local_attackers,
    );
}
