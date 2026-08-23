//! World generation utilities for blob_engine.
//! Lightweight versions that don't depend on external noise/probability crates.

use blob_interface::world::EnergySource;

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Generate a simple terrain with random elevation levels.
pub fn generate_flat_terrain(width: usize, height: usize, num_levels: usize) -> Vec<i32> {
    vec![(num_levels / 2) as i32; width * height]
}

/// Scatter random energy sources across the world.
#[allow(clippy::too_many_arguments)]
pub fn scatter_energy(
    width: usize,
    height: usize,
    num_scattered: usize,
    scattered_amount: u32,
    num_plants: usize,
    plant_rate: u32,
    plant_max_energy: u32,
    seed: u64,
) -> Vec<Option<EnergySource>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let total = width * height;
    let mut energy = vec![None; total];

    for _ in 0..num_scattered {
        let idx = rng.random_range(0..total);
        if energy[idx].is_none() {
            energy[idx] = Some(EnergySource::Scattered(scattered_amount));
        }
    }

    for _ in 0..num_plants {
        let idx = rng.random_range(0..total);
        if energy[idx].is_none() {
            energy[idx] = Some(EnergySource::Plant {
                rate: plant_rate,
                current_energy: plant_max_energy / 2,
                max_energy: plant_max_energy,
            });
        }
    }

    energy
}
