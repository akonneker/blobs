use std::collections::HashSet;
use libnoise::prelude::*;
use probability::prelude::*;
use rand::prelude::*;
use blob_interface::types::EnergyDistribution;
use blob_interface::world::EnergySource;
use crate::config::EnergyConfig;


// Helper function to get a value from a distribution using a seeded RNG
fn sample_from_distribution(dist: &EnergyDistribution, seed: Option<u64>, num_samples: usize) -> Vec<u32> {
    let mut source = source::default(seed.unwrap_or(42));
    
    match dist {
        EnergyDistribution::Constant(val) => vec![*val; num_samples],
        EnergyDistribution::Uniform(min, max) => {
            let distribution = Uniform::new(*min as f64, *max as f64);
            let sampler = Independent(&distribution, &mut source);
            let samples = sampler.take(num_samples).collect::<Vec<_>>();
            samples.iter().map(|s| s.round().max(0.0) as u32).collect()
        }
        EnergyDistribution::PseudoNormal(mean, std_dev) => {
            let distribution = Gaussian::new(*mean as f64, *std_dev as f64);
            let sampler = Independent(&distribution, &mut source);
            let samples = sampler.take(num_samples).collect::<Vec<_>>();
            samples.iter().map(|s| s.round().max(0.0) as u32).collect()
        }
        EnergyDistribution::Binomial(n, p) => {
            let distribution = Binomial::new(*n as usize, *p as f64);
            let sampler = Independent(&distribution, &mut source);
            let samples = sampler.take(num_samples).collect::<Vec<_>>();
            samples.iter().map(|s| *s as u32).collect()
        }
    }
}

pub fn generate_terrain(width: usize, height: usize, seed: Option<u64>, num_levels: usize) -> Vec<i32> {
    let generator = Source::simplex(seed.unwrap_or(42))                 // start with simplex noise
    .fbm(5, 0.013, 2.0, 0.5)                        // apply fractal brownian motion
    .blend(                                         // apply blending...
        Source::worley(43).scale([0.05, 0.05]),     // ...with scaled worley noise
        Source::worley(44).scale([0.02, 0.02]))     // ...controlled by other worley noise
    .lambda(|f| (f * 2.0).sin() * 0.3 + f * 0.7);   // apply a closure to the noise

    let mut terrain = Vec::with_capacity(width * height);
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    let mut raw_values = Vec::with_capacity(width * height);

    for y_idx in 0..height {
        for x_idx in 0..width {
            let nx = x_idx as f64 / width as f64;
            let ny = y_idx as f64 / height as f64;
            let value = generator.sample([nx, ny]);
            min_value = min_value.min(value);
            max_value = max_value.max(value);
            raw_values.push(value);
        }
    }

    let range = if (max_value - min_value).abs() < f64::EPSILON {
        1.0
    } else {
        max_value - min_value
    };

    for value in raw_values {
        let normalized = (value - min_value) / range;
        let num_levels_float = (num_levels.saturating_sub(1)) as f64;
        let quantized = (normalized * num_levels_float).round() as i32;
        terrain.push(quantized);
    }
    terrain
}

pub fn generate_energy(width: usize, height: usize, seed: Option<u64>, config: &EnergyConfig) -> Vec<Option<EnergySource>> {
    let mut energy = vec![None; width * height];
    let mut occupied_coords = HashSet::new();
    let total_cells = width * height;

    let scattered_energy_values = sample_from_distribution(&config.scattered_energy, seed, config.num_scattered);
    let plant_rate_values = sample_from_distribution(&config.plant_rate, seed, config.num_plants);
    let plant_max_energy_values = sample_from_distribution(&config.plant_max_energy, seed, config.num_plants);
    let plant_current_energy_values = sample_from_distribution(&config.plant_current_energy, seed, config.num_plants);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));

    // Generate Scattered Energy
    for i in 0..config.num_scattered {
        if occupied_coords.len() >= total_cells { break; }
        let mut coord_idx: usize;
        let mut attempts = 0;
        let max_attempts = total_cells.saturating_mul(2); // Prevent excessive looping

        loop {
            if attempts >= max_attempts {
                 coord_idx = usize::MAX; // Indicate failure
                 break;
            }
            coord_idx = rng.random_range(0..height*width);
            if !occupied_coords.contains(&coord_idx) {
                occupied_coords.insert(coord_idx);
                break;
            }
            attempts += 1;
        }
        
        if coord_idx != usize::MAX {
            energy[coord_idx] = Some(EnergySource::Scattered( scattered_energy_values[i]));
        }
    }

    // Generate Plants
    for i in 0..config.num_plants {
        if occupied_coords.len() >= total_cells { break; }
        let mut coord_idx: usize;
        let mut attempts = 0;
        let max_attempts = total_cells.saturating_mul(2);

        loop {
            if attempts >= max_attempts {
                coord_idx = usize::MAX; // Indicate failure
                break;
            }
            coord_idx = rng.random_range(0..height*width);
            if !occupied_coords.contains(&coord_idx) {
                occupied_coords.insert(coord_idx);
                break;
            }
            attempts += 1;
        }

        if coord_idx != usize::MAX {
            let rate = plant_rate_values[i];
            let max_e = plant_max_energy_values[i];
            let mut current_e = plant_current_energy_values[i];

            if current_e > max_e {
                current_e = max_e;
            }
            energy[coord_idx] = Some(EnergySource::Plant { rate, current_energy: current_e, max_energy: max_e });
        }
    }
    energy
}