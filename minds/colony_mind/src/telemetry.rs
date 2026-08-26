//! Trusted host-side diagnostics for colony scenarios.
//!
//! These metrics inspect canonical state after resolution. They are never
//! included in a Mind input, physics, hashes, or replay commitments.

use std::collections::BTreeSet;

use blob_engine::resolution::{BoundaryRule, ReferenceSimulation, TileIndex};
use serde::{Deserialize, Serialize};

use super::{ColonyMemory, Role};

pub const COLONY_SCENARIO_TELEMETRY_SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColonyScenarioTelemetry {
    pub schema_version: u32,
    pub colony_cells: usize,
    pub cells_by_role: [usize; 5],
    pub cells_with_valid_memory: usize,
    /// Aggregate occupancy of cell-private rolling visitation windows. This is
    /// a density diagnostic, not a canonical or globally mergeable map.
    pub private_coverage_bits: usize,
    pub explorer_coverage_bits: usize,
    pub maximum_private_coverage_bits: u32,
    pub outcome_successes: [usize; 8],
    pub outcome_setbacks: [usize; 8],
    pub outcome_contentions: [usize; 8],
    pub guarded_loss_weighted_sum: u64,
    pub guarded_loss_samples: usize,
    pub unguarded_loss_weighted_sum: u64,
    pub unguarded_loss_samples: usize,
    pub digestion_progress_weighted_sum: u64,
    pub digestion_samples: usize,
    pub signal_retention_weighted_sum_q8: u64,
    pub signal_samples: usize,
    /// Sum of private plant records. Coordinates are lineage-local, so this is
    /// intentionally not mislabeled as a globally unique plant count.
    pub mapped_plant_records: usize,
    pub maximum_records_in_one_cell: usize,
    /// Private records whose local builder plan reached the two-layer target.
    /// Multiple lineages may report the same world plant independently.
    pub completed_wall_plan_records: usize,
    pub excavation_pit_pressure: usize,
    pub maximum_excavation_pit_pressure: u8,
    /// Distinct canonical tiles reconstructed by trusted telemetry from each
    /// cell's private frame. This translation is never revealed to a Mind.
    pub mapped_world_plant_tiles: usize,
    pub confirmed_mapped_plant_tiles: usize,
    pub builders_carrying_material: usize,
    pub defenders_on_station: usize,
    pub world_plant_tiles: usize,
    pub plant_ring_tiles: usize,
    pub elevated_ring_tiles: usize,
    pub impassable_ring_tiles: usize,
    pub occupied_ring_tiles: usize,
    pub signal_energy: [u128; 4],
}

impl ColonyScenarioTelemetry {
    pub fn elevated_wall_coverage(&self) -> f64 {
        ratio(self.elevated_ring_tiles, self.plant_ring_tiles)
    }

    pub fn impassable_wall_coverage(&self) -> f64 {
        ratio(self.impassable_ring_tiles, self.plant_ring_tiles)
    }

    pub fn traversable_ring_tiles(&self) -> usize {
        self.plant_ring_tiles
            .saturating_sub(self.impassable_ring_tiles)
    }

    pub fn plant_discovery_coverage(&self) -> f64 {
        ratio(self.confirmed_mapped_plant_tiles, self.world_plant_tiles)
    }

    pub fn capture(simulation: &ReferenceSimulation, width: usize, height: usize) -> Self {
        assert_eq!(
            simulation.tiles().len(),
            width.saturating_mul(height),
            "telemetry dimensions must match the canonical simulation"
        );
        let mut sample = Self {
            schema_version: COLONY_SCENARIO_TELEMETRY_SCHEMA_VERSION,
            ..Self::default()
        };
        let plants: Vec<_> = simulation
            .tiles()
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                (tile.plant_capacity > 0 || tile.plant_growth_rate > 0).then_some(TileIndex(index))
            })
            .collect();
        let plant_set: BTreeSet<_> = plants.iter().copied().collect();
        sample.world_plant_tiles = plants.len();

        let mut ring_tiles = BTreeSet::new();
        for plant in &plants {
            let plant_x = plant.0 % width;
            let plant_y = plant.0 / width;
            for (dx, dy) in [
                (-1_isize, -1_isize),
                (0, -1),
                (1, -1),
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
            ] {
                if let Some(tile) = offset_tile(
                    plant_x,
                    plant_y,
                    dx,
                    dy,
                    width,
                    height,
                    simulation.rules().neighborhood.boundary_rule,
                ) {
                    ring_tiles.insert(tile);
                }
            }
        }
        sample.plant_ring_tiles = ring_tiles.len();
        for ring in &ring_tiles {
            let ring_state = &simulation.tiles()[ring.0];
            let nearest_plant_elevation = plants
                .iter()
                .filter(|plant| {
                    adjacent(
                        **plant,
                        *ring,
                        width,
                        height,
                        simulation.rules().neighborhood.boundary_rule,
                    )
                })
                .map(|plant| simulation.tiles()[plant.0].elevation)
                .min_by_key(|elevation| {
                    (i32::from(ring_state.elevation) - i32::from(*elevation)).unsigned_abs()
                });
            if let Some(plant_elevation) = nearest_plant_elevation {
                let delta = i32::from(ring_state.elevation) - i32::from(plant_elevation);
                if delta > 0 {
                    sample.elevated_ring_tiles = sample.elevated_ring_tiles.saturating_add(1);
                }
                if delta.abs() > i32::from(simulation.rules().maximum_elevation_delta) {
                    sample.impassable_ring_tiles = sample.impassable_ring_tiles.saturating_add(1);
                }
            }
            if ring_state.occupant.is_some() {
                sample.occupied_ring_tiles = sample.occupied_ring_tiles.saturating_add(1);
            }
        }

        for tile in simulation.tiles() {
            for (channel, energy) in tile.signal_energy.into_iter().enumerate() {
                sample.signal_energy[channel] =
                    sample.signal_energy[channel].saturating_add(u128::from(energy));
            }
        }

        let mut mapped_world_plants = BTreeSet::new();
        for (_, cell) in simulation.cells() {
            let Some(role) = Role::from_marker(cell.marker) else {
                continue;
            };
            sample.colony_cells = sample.colony_cells.saturating_add(1);
            sample.cells_by_role[role as usize] =
                sample.cells_by_role[role as usize].saturating_add(1);
            if role == Role::Builder && cell.carried_material_mass > 0 {
                sample.builders_carrying_material =
                    sample.builders_carrying_material.saturating_add(1);
            }
            if role == Role::Defender
                && plants.iter().any(|plant| {
                    adjacent_or_same(
                        *plant,
                        cell.position,
                        width,
                        height,
                        simulation.rules().neighborhood.boundary_rule,
                    )
                })
            {
                sample.defenders_on_station = sample.defenders_on_station.saturating_add(1);
            }
            if cell.private_memory.get(0..4) == Some(super::MAGIC.as_slice()) {
                let memory = ColonyMemory::decode(&cell.private_memory, role);
                sample.cells_with_valid_memory = sample.cells_with_valid_memory.saturating_add(1);
                let coverage_bits = memory.coverage.count_ones();
                sample.private_coverage_bits = sample
                    .private_coverage_bits
                    .saturating_add(coverage_bits as usize);
                if role == Role::Explorer {
                    sample.explorer_coverage_bits = sample
                        .explorer_coverage_bits
                        .saturating_add(coverage_bits as usize);
                }
                sample.maximum_private_coverage_bits =
                    sample.maximum_private_coverage_bits.max(coverage_bits);
                for family in 0..8 {
                    sample.outcome_successes[family] = sample.outcome_successes[family]
                        .saturating_add(usize::from(memory.outcome_successes[family]));
                    sample.outcome_setbacks[family] = sample.outcome_setbacks[family]
                        .saturating_add(usize::from(memory.outcome_setbacks[family]));
                    sample.outcome_contentions[family] = sample.outcome_contentions[family]
                        .saturating_add(usize::from(memory.outcome_contentions[family]));
                }
                sample.guarded_loss_weighted_sum = sample.guarded_loss_weighted_sum.saturating_add(
                    u64::from(memory.guarded_loss_ema)
                        .saturating_mul(u64::from(memory.guarded_loss_samples)),
                );
                sample.guarded_loss_samples = sample
                    .guarded_loss_samples
                    .saturating_add(usize::from(memory.guarded_loss_samples));
                sample.unguarded_loss_weighted_sum =
                    sample.unguarded_loss_weighted_sum.saturating_add(
                        u64::from(memory.unguarded_loss_ema)
                            .saturating_mul(u64::from(memory.unguarded_loss_samples)),
                    );
                sample.unguarded_loss_samples = sample
                    .unguarded_loss_samples
                    .saturating_add(usize::from(memory.unguarded_loss_samples));
                sample.digestion_progress_weighted_sum =
                    sample.digestion_progress_weighted_sum.saturating_add(
                        u64::from(memory.digestion_progress_ema)
                            .saturating_mul(u64::from(memory.digestion_samples)),
                    );
                sample.digestion_samples = sample
                    .digestion_samples
                    .saturating_add(usize::from(memory.digestion_samples));
                sample.signal_retention_weighted_sum_q8 =
                    sample.signal_retention_weighted_sum_q8.saturating_add(
                        u64::from(memory.signal_retention_q8)
                            .saturating_mul(u64::from(memory.signal_samples)),
                    );
                sample.signal_samples = sample
                    .signal_samples
                    .saturating_add(usize::from(memory.signal_samples));
                sample.mapped_plant_records = sample
                    .mapped_plant_records
                    .saturating_add(memory.plants.len());
                sample.maximum_records_in_one_cell =
                    sample.maximum_records_in_one_cell.max(memory.plants.len());
                for plant in &memory.plants {
                    if plant.wall_layer >= 2 {
                        sample.completed_wall_plan_records =
                            sample.completed_wall_plan_records.saturating_add(1);
                    }
                    sample.excavation_pit_pressure = sample
                        .excavation_pit_pressure
                        .saturating_add(usize::from(plant.pit_pressure));
                    sample.maximum_excavation_pit_pressure = sample
                        .maximum_excavation_pit_pressure
                        .max(plant.pit_pressure);
                    if let Some(tile) = translate_private_coordinate(
                        cell.position,
                        memory.x,
                        memory.y,
                        plant.x,
                        plant.y,
                        width,
                        height,
                        simulation.rules().neighborhood.boundary_rule,
                    ) {
                        mapped_world_plants.insert(tile);
                    }
                }
            }
        }
        sample.mapped_world_plant_tiles = mapped_world_plants.len();
        sample.confirmed_mapped_plant_tiles = mapped_world_plants.intersection(&plant_set).count();
        sample
    }
}

#[allow(clippy::too_many_arguments)]
fn translate_private_coordinate(
    actual: TileIndex,
    private_x: i16,
    private_y: i16,
    mapped_x: i16,
    mapped_y: i16,
    width: usize,
    height: usize,
    boundary: BoundaryRule,
) -> Option<TileIndex> {
    let actual_x = i64::try_from(actual.0 % width).ok()?;
    let actual_y = i64::try_from(actual.0 / width).ok()?;
    let world_x = actual_x - i64::from(private_x) + i64::from(mapped_x);
    let world_y = actual_y - i64::from(private_y) + i64::from(mapped_y);
    let width_i64 = i64::try_from(width).ok()?;
    let height_i64 = i64::try_from(height).ok()?;
    let (world_x, world_y) = match boundary {
        BoundaryRule::Bounded => {
            if world_x < 0 || world_y < 0 || world_x >= width_i64 || world_y >= height_i64 {
                return None;
            }
            (world_x, world_y)
        }
        BoundaryRule::Wrap => (
            world_x.rem_euclid(width_i64),
            world_y.rem_euclid(height_i64),
        ),
    };
    Some(TileIndex(
        usize::try_from(world_y)
            .ok()?
            .checked_mul(width)?
            .checked_add(usize::try_from(world_x).ok()?)?,
    ))
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn offset_tile(
    x: usize,
    y: usize,
    dx: isize,
    dy: isize,
    width: usize,
    height: usize,
    boundary: BoundaryRule,
) -> Option<TileIndex> {
    if width == 0 || height == 0 {
        return None;
    }
    let target = match boundary {
        BoundaryRule::Bounded => {
            let target_x = isize::try_from(x).ok()?.checked_add(dx)?;
            let target_y = isize::try_from(y).ok()?.checked_add(dy)?;
            if target_x < 0
                || target_y < 0
                || target_x >= isize::try_from(width).ok()?
                || target_y >= isize::try_from(height).ok()?
            {
                return None;
            }
            (
                usize::try_from(target_x).ok()?,
                usize::try_from(target_y).ok()?,
            )
        }
        BoundaryRule::Wrap => {
            let width = isize::try_from(width).ok()?;
            let height = isize::try_from(height).ok()?;
            (
                usize::try_from((isize::try_from(x).ok()? + dx).rem_euclid(width)).ok()?,
                usize::try_from((isize::try_from(y).ok()? + dy).rem_euclid(height)).ok()?,
            )
        }
    };
    Some(TileIndex(
        target.1.checked_mul(width)?.checked_add(target.0)?,
    ))
}

fn adjacent(
    left: TileIndex,
    right: TileIndex,
    width: usize,
    height: usize,
    boundary: BoundaryRule,
) -> bool {
    let left_x = left.0 % width;
    let left_y = left.0 / width;
    [
        (-1_isize, -1_isize),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
    ]
    .into_iter()
    .filter_map(|(dx, dy)| offset_tile(left_x, left_y, dx, dy, width, height, boundary))
    .any(|tile| tile == right)
}

fn adjacent_or_same(
    left: TileIndex,
    right: TileIndex,
    width: usize,
    height: usize,
    boundary: BoundaryRule,
) -> bool {
    left == right || adjacent(left, right, width, height, boundary)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use blob_engine::resolution::{NeighborhoodSpec, ReferenceRuleset, ReferenceSimulation};

    use super::*;

    fn simulation() -> ReferenceSimulation {
        let rules = ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
            metabolism_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut simulation = ReferenceSimulation::new(5, 5, rules).unwrap();
        let plant = simulation.tile(2, 2).unwrap();
        simulation.tile_state_mut(plant).unwrap().plant_capacity = 100;
        let defender = simulation
            .add_cell(plant, 10, 100, Role::Defender.marker())
            .unwrap();
        let mut state = simulation.canonical_state();
        let cell = state
            .cells
            .iter_mut()
            .find(|(key, _)| *key == defender)
            .unwrap();
        let mut memory = ColonyMemory::new(Role::Defender);
        memory.observe_plant(0, 0, 100, 5);
        memory.plants[0].wall_layer = 2;
        memory.plants[0].pit_pressure = 3;
        memory.outcome_successes[crate::EstimateFamily::Guard as usize] = 3;
        memory.outcome_contentions[crate::EstimateFamily::Move as usize] = 2;
        memory.guarded_loss_ema = 5;
        memory.guarded_loss_samples = 2;
        memory.digestion_progress_ema = 4;
        memory.digestion_samples = 3;
        memory.signal_retention_q8 = 192;
        memory.signal_samples = 4;
        cell.1.private_memory = Arc::from(memory.encode(2048).unwrap());
        ReferenceSimulation::from_canonical_state(5, 5, simulation.rules().clone(), state).unwrap()
    }

    #[test]
    fn reports_role_map_station_and_wall_semantics() {
        let mut simulation = simulation();
        for (x, y) in [
            (1, 1),
            (2, 1),
            (3, 1),
            (3, 2),
            (3, 3),
            (2, 3),
            (1, 3),
            (1, 2),
        ] {
            simulation
                .tile_state_mut(simulation.tile(x, y).unwrap())
                .unwrap()
                .elevation = 1;
        }
        let one_layer = ColonyScenarioTelemetry::capture(&simulation, 5, 5);
        assert_eq!(one_layer.schema_version, 4);
        assert_eq!(one_layer.cells_by_role[Role::Defender as usize], 1);
        assert_eq!(one_layer.defenders_on_station, 1);
        assert_eq!(one_layer.private_coverage_bits, 1);
        assert_eq!(one_layer.explorer_coverage_bits, 0);
        assert_eq!(one_layer.maximum_private_coverage_bits, 1);
        assert_eq!(
            one_layer.outcome_successes[crate::EstimateFamily::Guard as usize],
            3
        );
        assert_eq!(
            one_layer.outcome_contentions[crate::EstimateFamily::Move as usize],
            2
        );
        assert_eq!(one_layer.guarded_loss_weighted_sum, 10);
        assert_eq!(one_layer.guarded_loss_samples, 2);
        assert_eq!(one_layer.digestion_progress_weighted_sum, 12);
        assert_eq!(one_layer.digestion_samples, 3);
        assert_eq!(one_layer.signal_retention_weighted_sum_q8, 768);
        assert_eq!(one_layer.signal_samples, 4);
        assert_eq!(one_layer.completed_wall_plan_records, 1);
        assert_eq!(one_layer.excavation_pit_pressure, 3);
        assert_eq!(one_layer.maximum_excavation_pit_pressure, 3);
        assert_eq!(one_layer.mapped_plant_records, 1);
        assert_eq!(one_layer.confirmed_mapped_plant_tiles, 1);
        assert_eq!(one_layer.plant_discovery_coverage(), 1.0);
        assert_eq!(one_layer.elevated_wall_coverage(), 1.0);
        assert_eq!(one_layer.impassable_wall_coverage(), 0.0);

        simulation
            .tile_state_mut(simulation.tile(1, 1).unwrap())
            .unwrap()
            .elevation = 2;
        let two_layers = ColonyScenarioTelemetry::capture(&simulation, 5, 5);
        assert_eq!(two_layers.impassable_ring_tiles, 1);
        assert_eq!(two_layers.impassable_wall_coverage(), 0.125);
    }

    #[test]
    fn wrapped_corner_plant_has_eight_distinct_ring_tiles() {
        let rules = ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
            metabolism_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut simulation = ReferenceSimulation::new(5, 5, rules).unwrap();
        simulation
            .tile_state_mut(simulation.tile(0, 0).unwrap())
            .unwrap()
            .plant_capacity = 1;
        assert_eq!(
            ColonyScenarioTelemetry::capture(&simulation, 5, 5).plant_ring_tiles,
            8
        );
    }

    #[test]
    fn raised_ring_remains_traversable_when_rules_never_block_it() {
        let rules = ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
            maximum_elevation_delta: i16::MAX,
            metabolism_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut simulation = ReferenceSimulation::new(5, 5, rules).unwrap();
        simulation
            .tile_state_mut(simulation.tile(2, 2).unwrap())
            .unwrap()
            .plant_capacity = 100;
        for (x, y) in [
            (1, 1),
            (2, 1),
            (3, 1),
            (3, 2),
            (3, 3),
            (2, 3),
            (1, 3),
            (1, 2),
        ] {
            simulation
                .tile_state_mut(simulation.tile(x, y).unwrap())
                .unwrap()
                .elevation = 100;
        }
        let sample = ColonyScenarioTelemetry::capture(&simulation, 5, 5);
        assert_eq!(sample.elevated_ring_tiles, 8);
        assert_eq!(sample.impassable_ring_tiles, 0);
        assert_eq!(sample.traversable_ring_tiles(), 8);
    }

    #[test]
    fn independent_founder_frames_reconstruct_one_world_plant() {
        let rules = ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
            metabolism_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut simulation = ReferenceSimulation::new(5, 5, rules).unwrap();
        simulation
            .tile_state_mut(simulation.tile(2, 2).unwrap())
            .unwrap()
            .plant_capacity = 100;
        let first = simulation
            .add_cell(
                simulation.tile(1, 1).unwrap(),
                10,
                100,
                Role::Explorer.marker(),
            )
            .unwrap();
        let second = simulation
            .add_cell(
                simulation.tile(3, 3).unwrap(),
                10,
                100,
                Role::Explorer.marker(),
            )
            .unwrap();
        let mut state = simulation.canonical_state();
        for (key, cell) in &mut state.cells {
            let mut memory = ColonyMemory::new(Role::Explorer);
            if *key == first {
                // This founder calls its current world tile (0, 0).
                memory.observe_plant(1, 1, 100, 0);
            } else if *key == second {
                // A wholly unrelated frame still maps to the same world tile:
                // world = actual - private_position + mapped_position.
                memory.x = 10;
                memory.y = -5;
                memory.observe_plant(9, -6, 100, 0);
            }
            cell.private_memory = Arc::from(memory.encode(2048).unwrap());
        }
        let simulation =
            ReferenceSimulation::from_canonical_state(5, 5, simulation.rules().clone(), state)
                .unwrap();
        let sample = ColonyScenarioTelemetry::capture(&simulation, 5, 5);
        assert_eq!(sample.cells_with_valid_memory, 2);
        assert_eq!(sample.private_coverage_bits, 2);
        assert_eq!(sample.explorer_coverage_bits, 2);
        assert_eq!(sample.mapped_plant_records, 2);
        assert_eq!(sample.mapped_world_plant_tiles, 1);
        assert_eq!(sample.confirmed_mapped_plant_tiles, 1);
        assert_eq!(sample.plant_discovery_coverage(), 1.0);
    }
}
