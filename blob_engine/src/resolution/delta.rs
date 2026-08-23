//! Canonical sparse transitions between two authoritative simulation states.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use super::neighborhood::TileIndex;
use super::reference::{CellKey, CellState, SimTime, SimulationState, TileState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileDelta {
    pub tile: TileIndex,
    pub before: TileState,
    pub after: TileState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellDelta {
    pub cell: CellKey,
    pub before: Option<CellState>,
    pub after: Option<CellState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationDelta {
    pub before_time: SimTime,
    pub after_time: SimTime,
    pub before_next_cell_key: u64,
    pub after_next_cell_key: u64,
    pub tiles: Vec<TileDelta>,
    pub cells: Vec<CellDelta>,
}

impl SimulationDelta {
    pub fn between(before: &SimulationState, after: &SimulationState) -> Self {
        Self::between_with_tiles(before, after, tile_deltas_serial(before, after))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn between_parallel(before: &SimulationState, after: &SimulationState) -> Self {
        assert_same_tile_count(before, after);
        let tiles = before
            .tiles
            .par_iter()
            .zip(&after.tiles)
            .enumerate()
            .filter_map(|(index, (before, after))| {
                (before != after).then(|| TileDelta {
                    tile: TileIndex(index),
                    before: before.clone(),
                    after: after.clone(),
                })
            })
            .collect();
        Self::between_with_tiles(before, after, tiles)
    }

    fn between_with_tiles(
        before: &SimulationState,
        after: &SimulationState,
        tiles: Vec<TileDelta>,
    ) -> Self {
        assert_same_tile_count(before, after);
        Self {
            before_time: before.now,
            after_time: after.now,
            before_next_cell_key: before.next_cell_key,
            after_next_cell_key: after.next_cell_key,
            tiles,
            cells: merge_cell_deltas(&before.cells, &after.cells),
        }
    }

    pub fn apply_forward(&self, state: &mut SimulationState) -> Result<(), DeltaError> {
        self.apply(state, Direction::Forward)
    }

    pub fn apply_backward(&self, state: &mut SimulationState) -> Result<(), DeltaError> {
        self.apply(state, Direction::Backward)
    }

    fn apply(&self, state: &mut SimulationState, direction: Direction) -> Result<(), DeltaError> {
        self.validate_shape()?;
        let (expected_time, resulting_time, expected_key, resulting_key) = match direction {
            Direction::Forward => (
                self.before_time,
                self.after_time,
                self.before_next_cell_key,
                self.after_next_cell_key,
            ),
            Direction::Backward => (
                self.after_time,
                self.before_time,
                self.after_next_cell_key,
                self.before_next_cell_key,
            ),
        };
        if state.now != expected_time {
            return Err(DeltaError::TimeMismatch {
                expected: expected_time,
                actual: state.now,
            });
        }
        if state.next_cell_key != expected_key {
            return Err(DeltaError::NextCellKeyMismatch {
                expected: expected_key,
                actual: state.next_cell_key,
            });
        }

        for delta in &self.tiles {
            let tile = state
                .tiles
                .get(delta.tile.0)
                .ok_or(DeltaError::InvalidTile(delta.tile))?;
            let expected = match direction {
                Direction::Forward => &delta.before,
                Direction::Backward => &delta.after,
            };
            if tile != expected {
                return Err(DeltaError::TileMismatch(delta.tile));
            }
        }

        let mut cells: BTreeMap<_, _> = state.cells.iter().cloned().collect();
        for delta in &self.cells {
            let expected = match direction {
                Direction::Forward => &delta.before,
                Direction::Backward => &delta.after,
            };
            if cells.get(&delta.cell) != expected.as_ref() {
                return Err(DeltaError::CellMismatch(delta.cell));
            }
        }

        for delta in &self.tiles {
            state.tiles[delta.tile.0] = match direction {
                Direction::Forward => delta.after.clone(),
                Direction::Backward => delta.before.clone(),
            };
        }
        for delta in &self.cells {
            let replacement = match direction {
                Direction::Forward => &delta.after,
                Direction::Backward => &delta.before,
            };
            match replacement {
                Some(cell) => {
                    cells.insert(delta.cell, cell.clone());
                }
                None => {
                    cells.remove(&delta.cell);
                }
            }
        }
        state.cells = cells.into_iter().collect();
        state.now = resulting_time;
        state.next_cell_key = resulting_key;
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), DeltaError> {
        if !self
            .tiles
            .windows(2)
            .all(|pair| pair[0].tile < pair[1].tile)
            || !self
                .cells
                .windows(2)
                .all(|pair| pair[0].cell < pair[1].cell)
        {
            return Err(DeltaError::NonCanonicalOrder);
        }
        if self.tiles.iter().any(|delta| delta.before == delta.after)
            || self.cells.iter().any(|delta| delta.before == delta.after)
        {
            return Err(DeltaError::UnchangedResource);
        }
        Ok(())
    }
}

fn assert_same_tile_count(before: &SimulationState, after: &SimulationState) {
    assert_eq!(
        before.tiles.len(),
        after.tiles.len(),
        "a simulation delta cannot resize the world"
    );
}

fn tile_deltas_serial(before: &SimulationState, after: &SimulationState) -> Vec<TileDelta> {
    assert_same_tile_count(before, after);
    before
        .tiles
        .iter()
        .zip(&after.tiles)
        .enumerate()
        .filter(|(_, (before, after))| before != after)
        .map(|(index, (before, after))| TileDelta {
            tile: TileIndex(index),
            before: before.clone(),
            after: after.clone(),
        })
        .collect()
}

fn merge_cell_deltas(
    before: &[(CellKey, CellState)],
    after: &[(CellKey, CellState)],
) -> Vec<CellDelta> {
    debug_assert!(before.windows(2).all(|pair| pair[0].0 < pair[1].0));
    debug_assert!(after.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let mut deltas = Vec::new();
    let (mut before_index, mut after_index) = (0, 0);
    while before_index < before.len() || after_index < after.len() {
        match (before.get(before_index), after.get(after_index)) {
            (Some((before_key, before_cell)), Some((after_key, after_cell)))
                if before_key == after_key =>
            {
                if before_cell != after_cell {
                    deltas.push(CellDelta {
                        cell: *before_key,
                        before: Some(before_cell.clone()),
                        after: Some(after_cell.clone()),
                    });
                }
                before_index += 1;
                after_index += 1;
            }
            (Some((before_key, before_cell)), Some((after_key, _))) if before_key < after_key => {
                deltas.push(CellDelta {
                    cell: *before_key,
                    before: Some(before_cell.clone()),
                    after: None,
                });
                before_index += 1;
            }
            (Some(_), Some((after_key, after_cell))) => {
                deltas.push(CellDelta {
                    cell: *after_key,
                    before: None,
                    after: Some(after_cell.clone()),
                });
                after_index += 1;
            }
            (Some((before_key, before_cell)), None) => {
                deltas.push(CellDelta {
                    cell: *before_key,
                    before: Some(before_cell.clone()),
                    after: None,
                });
                before_index += 1;
            }
            (None, Some((after_key, after_cell))) => {
                deltas.push(CellDelta {
                    cell: *after_key,
                    before: None,
                    after: Some(after_cell.clone()),
                });
                after_index += 1;
            }
            (None, None) => break,
        }
    }
    deltas
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Forward,
    Backward,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaError {
    TimeMismatch { expected: SimTime, actual: SimTime },
    NextCellKeyMismatch { expected: u64, actual: u64 },
    InvalidTile(TileIndex),
    TileMismatch(TileIndex),
    CellMismatch(CellKey),
    NonCanonicalOrder,
    UnchangedResource,
}

impl Display for DeltaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TimeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "delta expected time {expected:?}, got {actual:?}"
                )
            }
            Self::NextCellKeyMismatch { expected, actual } => write!(
                formatter,
                "delta expected next cell key {expected}, got {actual}"
            ),
            Self::InvalidTile(tile) => write!(formatter, "delta references invalid tile {tile:?}"),
            Self::TileMismatch(tile) => {
                write!(formatter, "delta source tile {tile:?} does not match")
            }
            Self::CellMismatch(cell) => {
                write!(formatter, "delta source cell {cell:?} does not match")
            }
            Self::NonCanonicalOrder => write!(formatter, "delta resources are not canonical"),
            Self::UnchangedResource => write!(formatter, "delta contains an unchanged resource"),
        }
    }
}

impl Error for DeltaError {}
