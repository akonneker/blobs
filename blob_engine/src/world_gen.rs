//! World generation utilities for blob_engine.
//! Lightweight versions that don't depend on external noise/probability crates.

use blob_interface::world::EnergySource;
use blob_interface::{types::Coordinate, types::TeamId};

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;

const PLANT_LAYOUT_SEED_DOMAIN: u64 = 0x504c_414e_545f_4c59;
const LOOSE_LAYOUT_SEED_DOMAIN: u64 = 0x4c4f_4f53_455f_4c59;

/// Independent spatial distribution for one initial resource family.
/// Parameters affect setup only and must be included in scenario identity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResourceLayout {
    #[default]
    Uniform,
    Patches {
        patch_count: usize,
        radius: usize,
    },
    Corridors {
        corridor_count: usize,
        half_width: usize,
    },
    Islands {
        island_count: usize,
        radius: usize,
        minimum_separation: usize,
    },
    Territories {
        radius: usize,
    },
    FavoredTerritory {
        team: usize,
        radius: usize,
    },
}

impl ResourceLayout {
    pub fn validate_for_scenario(
        &self,
        width: usize,
        height: usize,
        team_count: usize,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("resource layout requires positive world dimensions".into());
        }
        let area = width
            .checked_mul(height)
            .ok_or("resource-layout world area overflowed")?;
        match self {
            Self::Uniform => {}
            Self::Patches {
                patch_count,
                radius: _,
            } => {
                if *patch_count == 0 || *patch_count > area {
                    return Err(
                        "patch resource layout requires a feasible positive patch count".into(),
                    );
                }
            }
            Self::Corridors { corridor_count, .. } => {
                if *corridor_count == 0 || *corridor_count > width.saturating_add(height) {
                    return Err("corridor resource layout count is invalid for the world".into());
                }
            }
            Self::Islands {
                island_count,
                radius,
                minimum_separation,
            } => {
                if *island_count == 0 || *minimum_separation <= radius.saturating_mul(2) {
                    return Err(
                        "island resource layout requires positive islands and separation greater than its diameter"
                            .into(),
                    );
                }
            }
            Self::Territories { .. } => {
                if team_count == 0 {
                    return Err("territorial resource layout requires starting teams".into());
                }
            }
            Self::FavoredTerritory { team, .. } => {
                if *team >= team_count {
                    return Err("favored resource territory names no starting team".into());
                }
            }
        }
        Ok(())
    }
}

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
    let mut positions = (0..total).collect::<Vec<_>>();
    positions.shuffle(&mut rng);
    let mut positions = positions.into_iter();

    for _ in 0..num_scattered {
        let Some(idx) = positions.next() else {
            break;
        };
        energy[idx] = Some(EnergySource::Scattered(scattered_amount));
    }

    for _ in 0..num_plants {
        let Some(idx) = positions.next() else {
            break;
        };
        energy[idx] = Some(EnergySource::Plant {
            rate: plant_rate,
            current_energy: plant_max_energy / 2,
            max_energy: plant_max_energy,
        });
    }

    energy
}

/// Generate plants and loose energy from independent deterministic layouts.
/// Uniform/uniform delegates to the historical generator byte-for-byte. Every
/// other layout either places exactly the requested unique sources or fails.
#[allow(clippy::too_many_arguments)]
pub fn scatter_energy_with_layouts(
    width: usize,
    height: usize,
    num_scattered: usize,
    scattered_amount: u32,
    scattered_layout: &ResourceLayout,
    num_plants: usize,
    plant_rate: u32,
    plant_max_energy: u32,
    plant_layout: &ResourceLayout,
    territories: &[(TeamId, Vec<Coordinate>)],
    seed: u64,
) -> Result<Vec<Option<EnergySource>>, String> {
    validate_layout(scattered_layout, width, height, territories)?;
    validate_layout(plant_layout, width, height, territories)?;
    if matches!(scattered_layout, ResourceLayout::Uniform)
        && matches!(plant_layout, ResourceLayout::Uniform)
    {
        return Ok(scatter_energy(
            width,
            height,
            num_scattered,
            scattered_amount,
            num_plants,
            plant_rate,
            plant_max_energy,
            seed,
        ));
    }

    let total = width
        .checked_mul(height)
        .ok_or("resource-layout world area overflowed")?;
    if num_scattered
        .checked_add(num_plants)
        .is_none_or(|count| count > total)
    {
        return Err("initial resource sources exceed world area".into());
    }
    let mut energy = vec![None; total];
    let mut plant_rng = StdRng::seed_from_u64(seed ^ PLANT_LAYOUT_SEED_DOMAIN);
    let plants = select_layout_positions(
        plant_layout,
        num_plants,
        width,
        height,
        territories,
        &energy,
        &mut plant_rng,
    )?;
    for position in plants {
        energy[position] = Some(EnergySource::Plant {
            rate: plant_rate,
            current_energy: plant_max_energy / 2,
            max_energy: plant_max_energy,
        });
    }

    let mut loose_rng = StdRng::seed_from_u64(seed ^ LOOSE_LAYOUT_SEED_DOMAIN);
    let scattered = select_layout_positions(
        scattered_layout,
        num_scattered,
        width,
        height,
        territories,
        &energy,
        &mut loose_rng,
    )?;
    for position in scattered {
        energy[position] = Some(EnergySource::Scattered(scattered_amount));
    }
    Ok(energy)
}

fn validate_layout(
    layout: &ResourceLayout,
    width: usize,
    height: usize,
    territories: &[(TeamId, Vec<Coordinate>)],
) -> Result<(), String> {
    let team_count = territories
        .iter()
        .map(|(team, _)| team.0)
        .max()
        .map_or(0, |team| team + 1);
    layout.validate_for_scenario(width, height, team_count)?;
    match layout {
        ResourceLayout::Uniform
        | ResourceLayout::Patches { .. }
        | ResourceLayout::Corridors { .. }
        | ResourceLayout::Islands { .. } => {}
        ResourceLayout::Territories { .. } => {
            if territories.iter().any(|(_, cells)| cells.is_empty()) {
                return Err("territorial resource layout requires nonempty starting teams".into());
            }
        }
        ResourceLayout::FavoredTerritory { team, .. } => {
            if !territories
                .iter()
                .any(|(candidate, cells)| candidate.0 == *team && !cells.is_empty())
            {
                return Err("favored resource territory names no starting team".into());
            }
        }
    }
    Ok(())
}

fn select_layout_positions(
    layout: &ResourceLayout,
    count: usize,
    width: usize,
    height: usize,
    territories: &[(TeamId, Vec<Coordinate>)],
    occupied: &[Option<EnergySource>],
    rng: &mut StdRng,
) -> Result<Vec<usize>, String> {
    if count == 0 {
        return Ok(Vec::new());
    }
    match layout {
        ResourceLayout::Territories { radius } => {
            select_balanced_territories(count, *radius, width, height, territories, occupied, rng)
        }
        ResourceLayout::FavoredTerritory { team, radius } => {
            let (distance, owner) = territory_distance_map(width, height, territories);
            let mut candidates = (0..width * height)
                .filter(|position| {
                    occupied[*position].is_none()
                        && distance[*position] <= *radius
                        && owner[*position] == Some(*team)
                })
                .collect::<Vec<_>>();
            candidates.shuffle(rng);
            take_exact(candidates, count, "favored-territory")
        }
        _ => {
            let mut candidates = layout_candidates(layout, width, height, rng)?;
            candidates.retain(|position| occupied[*position].is_none());
            candidates.shuffle(rng);
            take_exact(candidates, count, layout_name(layout))
        }
    }
}

fn layout_candidates(
    layout: &ResourceLayout,
    width: usize,
    height: usize,
    rng: &mut StdRng,
) -> Result<Vec<usize>, String> {
    let total = width * height;
    match layout {
        ResourceLayout::Uniform => Ok((0..total).collect()),
        ResourceLayout::Patches {
            patch_count,
            radius,
        } => {
            let mut centers = (0..total).collect::<Vec<_>>();
            centers.shuffle(rng);
            centers.truncate(*patch_count);
            Ok((0..total)
                .filter(|position| {
                    centers
                        .iter()
                        .any(|center| tile_distance(*position, *center, width, height) <= *radius)
                })
                .collect())
        }
        ResourceLayout::Corridors {
            corridor_count,
            half_width,
        } => {
            let mut axes = (0..width)
                .map(|offset| (true, offset))
                .chain((0..height).map(|offset| (false, offset)))
                .collect::<Vec<_>>();
            axes.shuffle(rng);
            axes.truncate(*corridor_count);
            Ok((0..total)
                .filter(|position| {
                    let x = *position % width;
                    let y = *position / width;
                    axes.iter().any(|(vertical, offset)| {
                        if *vertical {
                            toroidal_distance(x, *offset, width) <= *half_width
                        } else {
                            toroidal_distance(y, *offset, height) <= *half_width
                        }
                    })
                })
                .collect())
        }
        ResourceLayout::Islands {
            island_count,
            radius,
            minimum_separation,
        } => {
            let mut positions = (0..total).collect::<Vec<_>>();
            positions.shuffle(rng);
            let mut centers = Vec::with_capacity(*island_count);
            for position in positions {
                if centers.iter().all(|center| {
                    tile_distance(position, *center, width, height) >= *minimum_separation
                }) {
                    centers.push(position);
                    if centers.len() == *island_count {
                        break;
                    }
                }
            }
            if centers.len() != *island_count {
                return Err("island resource layout cannot satisfy minimum separation".into());
            }
            Ok((0..total)
                .filter(|position| {
                    centers
                        .iter()
                        .any(|center| tile_distance(*position, *center, width, height) <= *radius)
                })
                .collect())
        }
        ResourceLayout::Territories { .. } | ResourceLayout::FavoredTerritory { .. } => {
            unreachable!("territorial layouts use their ownership map")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn select_balanced_territories(
    count: usize,
    radius: usize,
    width: usize,
    height: usize,
    territories: &[(TeamId, Vec<Coordinate>)],
    occupied: &[Option<EnergySource>],
    rng: &mut StdRng,
) -> Result<Vec<usize>, String> {
    let (distance, owner) = territory_distance_map(width, height, territories);
    let mut team_ids = territories
        .iter()
        .map(|(team, _)| team.0)
        .collect::<Vec<_>>();
    team_ids.sort_unstable();
    team_ids.dedup();
    let rotation = rng.random_range(0..team_ids.len());
    team_ids.rotate_left(rotation);
    let mut selected = Vec::with_capacity(count);
    for (index, team) in team_ids.iter().copied().enumerate() {
        let quota = count / team_ids.len() + usize::from(index < count % team_ids.len());
        let mut candidates = (0..width * height)
            .filter(|position| {
                occupied[*position].is_none()
                    && distance[*position] <= radius
                    && owner[*position] == Some(team)
            })
            .collect::<Vec<_>>();
        candidates.shuffle(rng);
        selected.extend(take_exact(candidates, quota, "balanced-territories")?);
    }
    Ok(selected)
}

fn territory_distance_map(
    width: usize,
    height: usize,
    territories: &[(TeamId, Vec<Coordinate>)],
) -> (Vec<usize>, Vec<Option<usize>>) {
    let total = width * height;
    let mut distance = vec![usize::MAX; total];
    let mut owner = vec![None; total];
    let mut queue = VecDeque::new();
    let mut ordered = territories.to_vec();
    ordered.sort_unstable_by_key(|(team, _)| team.0);
    for (team, coordinates) in ordered {
        for coordinate in coordinates {
            let position = coordinate.y * width + coordinate.x;
            if distance[position] > 0
                || (distance[position] == 0 && owner[position].is_some_and(|old| team.0 < old))
            {
                distance[position] = 0;
                owner[position] = Some(team.0);
                queue.push_back(position);
            }
        }
    }
    while let Some(position) = queue.pop_front() {
        let next_distance = distance[position] + 1;
        let source_owner = owner[position];
        let x = position % width;
        let y = position / width;
        for dy in [-1_isize, 0, 1] {
            for dx in [-1_isize, 0, 1] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let next_x = (x as isize + dx).rem_euclid(width as isize) as usize;
                let next_y = (y as isize + dy).rem_euclid(height as isize) as usize;
                let next = next_y * width + next_x;
                if next_distance < distance[next]
                    || (next_distance == distance[next] && source_owner < owner[next])
                {
                    distance[next] = next_distance;
                    owner[next] = source_owner;
                    queue.push_back(next);
                }
            }
        }
    }
    (distance, owner)
}

fn take_exact(
    mut candidates: Vec<usize>,
    count: usize,
    layout: &str,
) -> Result<Vec<usize>, String> {
    if candidates.len() < count {
        return Err(format!(
            "{layout} resource layout has {} eligible vacant tiles but {count} were requested",
            candidates.len()
        ));
    }
    candidates.truncate(count);
    Ok(candidates)
}

fn layout_name(layout: &ResourceLayout) -> &'static str {
    match layout {
        ResourceLayout::Uniform => "uniform",
        ResourceLayout::Patches { .. } => "patches",
        ResourceLayout::Corridors { .. } => "corridors",
        ResourceLayout::Islands { .. } => "islands",
        ResourceLayout::Territories { .. } => "territories",
        ResourceLayout::FavoredTerritory { .. } => "favored-territory",
    }
}

fn tile_distance(left: usize, right: usize, width: usize, height: usize) -> usize {
    let left_x = left % width;
    let left_y = left / width;
    let right_x = right % width;
    let right_y = right / width;
    toroidal_distance(left_x, right_x, width).max(toroidal_distance(left_y, right_y, height))
}

fn toroidal_distance(left: usize, right: usize, extent: usize) -> usize {
    let direct = left.abs_diff(right);
    direct.min(extent - direct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_energy_is_seeded_unique_and_exact_when_it_fits() {
        let first = scatter_energy(8, 8, 7, 3, 5, 2, 10, 41);
        let second = scatter_energy(8, 8, 7, 3, 5, 2, 10, 41);
        assert!(first == second);
        assert_eq!(
            first
                .iter()
                .filter(|source| matches!(source, Some(EnergySource::Scattered(3))))
                .count(),
            7
        );
        assert_eq!(
            first
                .iter()
                .filter(|source| matches!(source, Some(EnergySource::Plant { .. })))
                .count(),
            5
        );
        assert!(first != scatter_energy(8, 8, 7, 3, 5, 2, 10, 42));
    }

    fn source_positions(energy: &[Option<EnergySource>], plant: bool) -> Vec<usize> {
        energy
            .iter()
            .enumerate()
            .filter_map(|(position, source)| {
                let matches = if plant {
                    matches!(source, Some(EnergySource::Plant { .. }))
                } else {
                    matches!(source, Some(EnergySource::Scattered(_)))
                };
                matches.then_some(position)
            })
            .collect()
    }

    #[test]
    fn spatial_resource_layouts_are_seeded_unique_and_exact() {
        let layouts = [
            ResourceLayout::Uniform,
            ResourceLayout::Patches {
                patch_count: 4,
                radius: 5,
            },
            ResourceLayout::Corridors {
                corridor_count: 3,
                half_width: 1,
            },
            ResourceLayout::Islands {
                island_count: 4,
                radius: 3,
                minimum_separation: 8,
            },
        ];
        for layout in layouts {
            let first =
                scatter_energy_with_layouts(64, 64, 50, 3, &layout, 30, 2, 10, &layout, &[], 71)
                    .unwrap();
            let second =
                scatter_energy_with_layouts(64, 64, 50, 3, &layout, 30, 2, 10, &layout, &[], 71)
                    .unwrap();
            assert!(first == second);
            assert_eq!(source_positions(&first, false).len(), 50);
            assert_eq!(source_positions(&first, true).len(), 30);
        }
    }

    #[test]
    fn plant_layout_is_independent_of_loose_energy_layout() {
        let plants = ResourceLayout::Patches {
            patch_count: 3,
            radius: 4,
        };
        let uniform = scatter_energy_with_layouts(
            48,
            48,
            40,
            3,
            &ResourceLayout::Uniform,
            20,
            2,
            10,
            &plants,
            &[],
            72,
        )
        .unwrap();
        let corridors = scatter_energy_with_layouts(
            48,
            48,
            40,
            3,
            &ResourceLayout::Corridors {
                corridor_count: 2,
                half_width: 1,
            },
            20,
            2,
            10,
            &plants,
            &[],
            72,
        )
        .unwrap();
        assert_eq!(
            source_positions(&uniform, true),
            source_positions(&corridors, true)
        );
    }

    #[test]
    fn territorial_layouts_balance_or_explicitly_favor_starting_teams() {
        let territories = vec![
            (TeamId(0), vec![Coordinate { x: 4, y: 8 }]),
            (TeamId(1), vec![Coordinate { x: 20, y: 8 }]),
        ];
        let balanced = scatter_energy_with_layouts(
            32,
            16,
            0,
            3,
            &ResourceLayout::Uniform,
            20,
            2,
            10,
            &ResourceLayout::Territories { radius: 4 },
            &territories,
            73,
        )
        .unwrap();
        let (distance, owner) = territory_distance_map(32, 16, &territories);
        let balanced_owners = source_positions(&balanced, true)
            .into_iter()
            .map(|position| {
                assert!(distance[position] <= 4);
                owner[position].unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            balanced_owners.iter().filter(|owner| **owner == 0).count(),
            10
        );
        assert_eq!(
            balanced_owners.iter().filter(|owner| **owner == 1).count(),
            10
        );

        let favored = scatter_energy_with_layouts(
            32,
            16,
            0,
            3,
            &ResourceLayout::Uniform,
            20,
            2,
            10,
            &ResourceLayout::FavoredTerritory { team: 1, radius: 4 },
            &territories,
            73,
        )
        .unwrap();
        assert!(source_positions(&favored, true)
            .into_iter()
            .all(|position| owner[position] == Some(1) && distance[position] <= 4));
    }

    #[test]
    fn impossible_resource_geometry_fails_instead_of_spilling_elsewhere() {
        let error = scatter_energy_with_layouts(
            16,
            16,
            0,
            3,
            &ResourceLayout::Uniform,
            100,
            2,
            10,
            &ResourceLayout::Patches {
                patch_count: 1,
                radius: 1,
            },
            &[],
            74,
        );
        let error = match error {
            Ok(_) => panic!("impossible patch layout unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.contains("eligible vacant tiles"));

        assert!(scatter_energy_with_layouts(
            16,
            16,
            0,
            3,
            &ResourceLayout::Uniform,
            1,
            2,
            10,
            &ResourceLayout::Islands {
                island_count: 8,
                radius: 2,
                minimum_separation: 20,
            },
            &[],
            74,
        )
        .is_err());
    }
}
