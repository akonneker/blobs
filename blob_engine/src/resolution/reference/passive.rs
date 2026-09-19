//! Passive arithmetic and diffusion topology, independent of event orchestration.
//!
//! Scheduling, active-frontier maintenance and mutation journals remain owned by
//! the parent resolver. These kernels retain its exact checked integer rules.

use super::{
    CellState, CompiledNeighborhood, LocalSlot, ResolutionError, SimTime, SlotMask, TileIndex,
    TileState, REFERENCE_SIGNAL_CHANNELS,
};
#[cfg(not(target_arch = "wasm32"))]
use super::{ReferenceTileStore, REFERENCE_TILE_PAGE_BITMAP_WORDS, REFERENCE_TILE_PAGE_LEN};
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
pub(super) struct PassiveRateDivisor {
    denominator: u64,
    power_of_two_shift: Option<u32>,
}

impl PassiveRateDivisor {
    pub(super) fn new(denominator: u64) -> Self {
        Self {
            denominator,
            power_of_two_shift: denominator
                .is_power_of_two()
                .then(|| denominator.trailing_zeros()),
        }
    }

    pub(super) fn quotient(self, numerator: u128) -> u128 {
        self.power_of_two_shift.map_or_else(
            || numerator / u128::from(self.denominator),
            |shift| numerator >> shift,
        )
    }

    pub(super) fn remainder(self, numerator: u128) -> u128 {
        self.power_of_two_shift.map_or_else(
            || numerator % u128::from(self.denominator),
            |_| numerator & u128::from(self.denominator - 1),
        )
    }
}

#[cfg(test)]
pub(super) fn advance_plant_growth(
    tile: &mut TileState,
    elapsed: u64,
    denominator: u64,
) -> Result<(), ResolutionError> {
    advance_plant_growth_with_divisor(tile, elapsed, PassiveRateDivisor::new(denominator))
}

pub(super) fn advance_plant_growth_with_divisor(
    tile: &mut TileState,
    elapsed: u64,
    divisor: PassiveRateDivisor,
) -> Result<(), ResolutionError> {
    if tile.plant_growth_rate == 0 {
        tile.plant_growth_remainder = 0;
        return Ok(());
    }
    let available_room = tile.plant_capacity.saturating_sub(tile.plant_energy);
    if available_room == 0 || tile.diffuse_energy == 0 {
        // Growth cannot be banked while its source or sink is absent.
        tile.plant_growth_remainder = 0;
        return Ok(());
    }
    let generated = u128::from(elapsed)
        .checked_mul(u128::from(tile.plant_growth_rate))
        .and_then(|value| value.checked_add(u128::from(tile.plant_growth_remainder)))
        .ok_or(ResolutionError::ArithmeticOverflow(
            "accumulating plant growth progress",
        ))?;
    let potential = u64::try_from(divisor.quotient(generated).min(u128::from(u64::MAX)))
        .map_err(|_| ResolutionError::ArithmeticOverflow("converting plant growth"))?;
    let grown = potential.min(tile.diffuse_energy).min(available_room);
    tile.diffuse_energy -= grown;
    tile.plant_energy =
        tile.plant_energy
            .checked_add(grown)
            .ok_or(ResolutionError::ArithmeticOverflow(
                "adding grown plant energy",
            ))?;
    tile.plant_growth_remainder = if tile.diffuse_energy == 0
        || tile.plant_energy == tile.plant_capacity
    {
        0
    } else {
        u64::try_from(divisor.remainder(generated))
            .map_err(|_| ResolutionError::ArithmeticOverflow("storing plant growth remainder"))?
    };
    Ok(())
}

/// Returns exact per-channel energy deposited into the tile's diffuse
/// reservoir. Each tile is independent, so dense hosts may call this in
/// parallel without changing reduction order.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn planned_signal_frontier_changes(
    tile: &TileState,
    elapsed: u64,
    numerator: u64,
    denominator: u64,
) -> (bool, bool) {
    let mut has_signal_before = false;
    let mut has_signal_after = false;
    let mut deposits_diffuse = false;
    for channel in 0..REFERENCE_SIGNAL_CHANNELS {
        let energy = tile.signal_energy[channel];
        if energy == 0 {
            continue;
        }
        has_signal_before = true;
        // Unsigned 64-bit multiplication and addition fit in u128. Ruleset
        // validation guarantees the denominator is nonzero.
        let generated = u128::from(elapsed) * u128::from(numerator)
            + u128::from(tile.signal_decay_remainder[channel]);
        let decayed = (generated / u128::from(denominator)).min(u128::from(energy));
        has_signal_after |= decayed < u128::from(energy);
        deposits_diffuse |= decayed > 0;
    }
    (
        has_signal_before && !has_signal_after,
        tile.diffuse_energy == 0 && deposits_diffuse,
    )
}

pub(super) fn advance_signal_decay(
    tile: &mut TileState,
    elapsed: u64,
    numerator: u64,
    denominator: u64,
) -> Result<[u64; REFERENCE_SIGNAL_CHANNELS], ResolutionError> {
    let mut deposited = [0_u64; REFERENCE_SIGNAL_CHANNELS];
    for (channel, deposited_channel) in deposited.iter_mut().enumerate() {
        let energy = tile.signal_energy[channel];
        if energy == 0 {
            tile.signal_decay_remainder[channel] = 0;
            continue;
        }
        let generated = u128::from(elapsed)
            .checked_mul(u128::from(numerator))
            .and_then(|value| value.checked_add(u128::from(tile.signal_decay_remainder[channel])))
            .ok_or(ResolutionError::ArithmeticOverflow(
                "accumulating signal decay",
            ))?;
        let decayed = u64::try_from((generated / u128::from(denominator)).min(u128::from(energy)))
            .map_err(|_| ResolutionError::ArithmeticOverflow("converting signal decay"))?;
        tile.signal_energy[channel] -= decayed;
        tile.signal_decay_remainder[channel] = if tile.signal_energy[channel] == 0 {
            0
        } else {
            u64::try_from(generated % u128::from(denominator)).map_err(|_| {
                ResolutionError::ArithmeticOverflow("storing signal decay remainder")
            })?
        };
        tile.diffuse_energy =
            tile.diffuse_energy
                .checked_add(decayed)
                .ok_or(ResolutionError::ArithmeticOverflow(
                    "depositing decayed signal energy",
                ))?;
        *deposited_channel = decayed;
    }
    Ok(deposited)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct DenseSignalDecayPage {
    pub(super) decayed: [u128; REFERENCE_SIGNAL_CHANNELS],
    pub(super) active_signal: [u64; REFERENCE_TILE_PAGE_BITMAP_WORDS],
    pub(super) active_diffuse: [u64; REFERENCE_TILE_PAGE_BITMAP_WORDS],
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn dense_frontier_from_page_bitmaps<'a>(
    bitmaps: impl IntoIterator<Item = &'a [u64; REFERENCE_TILE_PAGE_BITMAP_WORDS]>,
) -> BTreeSet<TileIndex> {
    bitmaps
        .into_iter()
        .enumerate()
        .flat_map(|(page_index, bits)| {
            bits.iter().enumerate().flat_map(move |(word_index, word)| {
                let mut remaining = *word;
                std::iter::from_fn(move || {
                    if remaining == 0 {
                        return None;
                    }
                    let bit = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1;
                    Some(TileIndex(
                        page_index * REFERENCE_TILE_PAGE_LEN + word_index * 64 + bit,
                    ))
                })
            })
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy)]
pub(super) struct DenseDiffusionFlux {
    pub(super) remainder: u64,
    pub(super) share: u64,
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn plan_dense_diffusion(
    tile: &TileState,
    neighbors: &[TileIndex],
    numerator: u64,
    denominator: u64,
) -> DenseDiffusionFlux {
    let energy = tile.diffuse_energy;
    let neighbor_count = u64::try_from(neighbors.len())
        .expect("validated dense diffusion neighborhood count fits u64");
    if neighbors.is_empty() || energy < neighbor_count {
        return DenseDiffusionFlux {
            remainder: 0,
            share: 0,
        };
    }
    let generated = u128::from(energy)
        .checked_mul(u128::from(numerator))
        .and_then(|value| value.checked_add(u128::from(tile.diffusion_remainder)))
        .expect("u64 diffusion product and remainder fit u128");
    let desired = u64::try_from((generated / u128::from(denominator)).min(u128::from(energy)))
        .expect("planned dense diffusion is bounded by tile energy");
    let remainder = u64::try_from(generated % u128::from(denominator))
        .expect("validated dense diffusion remainder fits u64");
    let share = desired / neighbor_count;
    DenseDiffusionFlux { remainder, share }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn dense_diffusion_final_energy(
    destination: usize,
    tiles: &ReferenceTileStore,
    diffusion_neighbors: &[Vec<TileIndex>],
    diffusion_sources: &[Vec<TileIndex>],
    shares: &[u64],
) -> Result<u64, ResolutionError> {
    let neighbor_count = u64::try_from(diffusion_neighbors[destination].len())
        .expect("validated dense diffusion neighborhood count fits u64");
    let transported = shares[destination]
        .checked_mul(neighbor_count)
        .expect("equal dense diffusion shares cannot exceed source energy");
    let remaining_energy = tiles[destination].diffuse_energy - transported;
    let incoming = diffusion_sources[destination]
        .iter()
        .try_fold(0_u64, |total, source| {
            total
                .checked_add(shares[source.0])
                .ok_or(ResolutionError::ArithmeticOverflow(
                    "aggregating diffuse transport",
                ))
        })?;
    remaining_energy
        .checked_add(incoming)
        .ok_or(ResolutionError::ArithmeticOverflow(
            "committing diffuse transport",
        ))
}

pub(super) fn advance_cell_digestion(
    cell: &mut CellState,
    elapsed: u64,
    numerator: u64,
    denominator: u64,
) -> Result<(), ResolutionError> {
    if cell.gut_energy == 0 || numerator == 0 {
        cell.digestion_remainder = 0;
        return Ok(());
    }
    let generated = u128::from(elapsed)
        .checked_mul(u128::from(numerator))
        .and_then(|value| value.checked_add(u128::from(cell.digestion_remainder)))
        .ok_or(ResolutionError::ArithmeticOverflow(
            "accumulating digestion progress",
        ))?;
    let potential = generated / u128::from(denominator);
    let digested = u64::try_from(potential.min(u128::from(cell.gut_energy)))
        .map_err(|_| ResolutionError::ArithmeticOverflow("converting digestion amount"))?;
    cell.gut_energy -= digested;
    cell.assimilated_energy = cell.assimilated_energy.checked_add(digested).ok_or(
        ResolutionError::ArithmeticOverflow("adding digested energy"),
    )?;
    cell.digestion_remainder = if cell.gut_energy == 0 {
        0
    } else {
        u64::try_from(generated % u128::from(denominator))
            .map_err(|_| ResolutionError::ArithmeticOverflow("storing digestion remainder"))?
    };
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn dense_digestion_preflight(
    cell: &CellState,
    elapsed: u64,
    digestion_numerator: u64,
    digestion_denominator: u64,
    now: SimTime,
    metabolism_numerator: u64,
    metabolism_denominator: u64,
) -> (bool, bool) {
    let generated = u128::from(elapsed) * u128::from(digestion_numerator)
        + u128::from(cell.digestion_remainder);
    let digested = (generated / u128::from(digestion_denominator)).min(u128::from(cell.gut_energy));
    let Ok(digested) = u64::try_from(digested) else {
        return (false, false);
    };
    let Some(assimilated_energy) = cell.assimilated_energy.checked_add(digested) else {
        return (false, false);
    };
    let deadline_fits = metabolism_numerator > 0
        && metabolic_exhaustion_time_from_values(
            now,
            assimilated_energy,
            cell.metabolism_remainder,
            metabolism_numerator,
            metabolism_denominator,
        )
        .is_ok();
    (true, deadline_fits)
}

#[cfg(test)]
pub(super) fn advance_cell_metabolism(
    cell: &mut CellState,
    elapsed: u64,
    numerator: u64,
    denominator: u64,
) -> Result<u64, ResolutionError> {
    advance_cell_metabolism_with_divisor(
        cell,
        elapsed,
        numerator,
        PassiveRateDivisor::new(denominator),
    )
}

pub(super) fn advance_cell_metabolism_with_divisor(
    cell: &mut CellState,
    elapsed: u64,
    numerator: u64,
    divisor: PassiveRateDivisor,
) -> Result<u64, ResolutionError> {
    if cell.assimilated_energy == 0 || numerator == 0 {
        cell.metabolism_remainder = 0;
        return Ok(0);
    }
    let generated = u128::from(elapsed)
        .checked_mul(u128::from(numerator))
        .and_then(|value| value.checked_add(u128::from(cell.metabolism_remainder)))
        .ok_or(ResolutionError::ArithmeticOverflow(
            "accumulating metabolism progress",
        ))?;
    let potential = divisor.quotient(generated);
    let spent = u64::try_from(potential.min(u128::from(cell.assimilated_energy)))
        .map_err(|_| ResolutionError::ArithmeticOverflow("converting metabolism cost"))?;
    cell.assimilated_energy -= spent;
    cell.metabolism_remainder = if cell.assimilated_energy == 0 {
        0
    } else {
        u64::try_from(divisor.remainder(generated))
            .map_err(|_| ResolutionError::ArithmeticOverflow("storing metabolism remainder"))?
    };
    Ok(spent)
}

pub(super) fn metabolic_exhaustion_time(
    now: SimTime,
    cell: &CellState,
    numerator: u64,
    denominator: u64,
) -> Result<SimTime, ResolutionError> {
    metabolic_exhaustion_time_from_values(
        now,
        cell.assimilated_energy,
        cell.metabolism_remainder,
        numerator,
        denominator,
    )
}

fn metabolic_exhaustion_time_from_values(
    now: SimTime,
    assimilated_energy: u64,
    metabolism_remainder: u64,
    numerator: u64,
    denominator: u64,
) -> Result<SimTime, ResolutionError> {
    let required = u128::from(assimilated_energy)
        .checked_mul(u128::from(denominator))
        .and_then(|value| value.checked_sub(u128::from(metabolism_remainder)))
        .ok_or(ResolutionError::ArithmeticOverflow(
            "scheduling metabolic exhaustion",
        ))?;
    let elapsed = required.checked_add(u128::from(numerator - 1)).ok_or(
        ResolutionError::ArithmeticOverflow("rounding metabolic exhaustion time"),
    )? / u128::from(numerator);
    let elapsed = u64::try_from(elapsed)
        .map_err(|_| ResolutionError::ArithmeticOverflow("converting metabolic exhaustion time"))?;
    now.0
        .checked_add(elapsed)
        .map(SimTime)
        .ok_or(ResolutionError::ArithmeticOverflow(
            "scheduling metabolic exhaustion time",
        ))
}

pub(super) fn compile_diffusion_neighbors(
    neighborhood: &CompiledNeighborhood,
    mask: SlotMask,
) -> Vec<Vec<TileIndex>> {
    (0..neighborhood.tile_count())
        .map(|index| {
            let origin = TileIndex(index);
            let mut neighbors = std::collections::BTreeSet::new();
            for slot_index in 0..neighborhood.spec().slots.len() {
                let slot = LocalSlot(slot_index as u8);
                if !mask.contains(slot) {
                    continue;
                }
                if let Some(target) = neighborhood.target(origin, slot) {
                    if target != origin {
                        neighbors.insert(target);
                    }
                }
            }
            neighbors.into_iter().collect()
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn compile_diffusion_sources(
    neighbors: &[Vec<TileIndex>],
    tile_count: usize,
) -> Vec<Vec<TileIndex>> {
    let mut sources = vec![Vec::new(); tile_count];
    for (source, destinations) in neighbors.iter().enumerate() {
        let source = TileIndex(source);
        for destination in destinations {
            sources[destination.0].push(source);
        }
    }
    sources
}
