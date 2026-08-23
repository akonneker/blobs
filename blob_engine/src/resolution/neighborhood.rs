use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Maximum number of addressable local offsets in one ABI/ruleset.
pub const MAX_LOCAL_SLOTS: usize = 32;

/// Dense canonical index into the world's tile arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileIndex(pub usize);

/// Ruleset-defined local offset index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalSlot(pub u8);

/// Bit mask over [`LocalSlot`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotMask(u32);

impl SlotMask {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn first(count: usize) -> Self {
        if count >= MAX_LOCAL_SLOTS {
            Self(u32::MAX)
        } else if count == 0 {
            Self::empty()
        } else {
            Self((1_u32 << count) - 1)
        }
    }

    pub fn from_slots(slots: impl IntoIterator<Item = LocalSlot>) -> Self {
        let mut mask = Self::empty();
        for slot in slots {
            mask.insert(slot);
        }
        mask
    }

    pub fn insert(&mut self, slot: LocalSlot) {
        if usize::from(slot.0) < MAX_LOCAL_SLOTS {
            self.0 |= 1_u32 << slot.0;
        }
    }

    pub const fn contains(self, slot: LocalSlot) -> bool {
        slot.0 < MAX_LOCAL_SLOTS as u8 && (self.0 & (1_u32 << slot.0)) != 0
    }

    pub const fn bits(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalOffset {
    pub dx: i8,
    pub dy: i8,
    /// Fixed-point distance/cost multiplier where 1024 represents 1.0.
    pub distance_cost_q10: u16,
}

impl LocalOffset {
    pub const fn new(dx: i8, dy: i8, distance_cost_q10: u16) -> Self {
        Self {
            dx,
            dy,
            distance_cost_q10,
        }
    }

    pub const fn is_diagonal(self) -> bool {
        self.dx != 0 && self.dy != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryRule {
    Bounded,
    Wrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagonalCornerRule {
    Allow,
    BlockIfEitherOrthogonalOccupied,
    BlockIfBothOrthogonalsOccupied,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TargetingAction {
    Move = 0,
    Attack = 1,
    Split = 2,
    Regurgitate = 3,
}

impl TargetingAction {
    const COUNT: usize = 4;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationMasks {
    pub occupancy: SlotMask,
    pub marker: SlotMask,
    pub apparent_mass: SlotMask,
    pub activity: SlotMask,
    pub terrain: SlotMask,
    pub energy: SlotMask,
    pub signal: SlotMask,
}

impl ObservationMasks {
    pub fn all(slot_count: usize) -> Self {
        let all = SlotMask::first(slot_count);
        Self {
            occupancy: all,
            marker: all,
            apparent_mass: all,
            activity: all,
            terrain: all,
            energy: all,
            signal: all,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborhoodSpec {
    pub slots: Vec<LocalOffset>,
    pub observations: ObservationMasks,
    target_masks: [SlotMask; TargetingAction::COUNT],
    pub diagonal_corner_rule: DiagonalCornerRule,
    pub boundary_rule: BoundaryRule,
    pub max_radius: u8,
}

impl NeighborhoodSpec {
    pub fn new(
        slots: Vec<LocalOffset>,
        observations: ObservationMasks,
        target_masks: [SlotMask; TargetingAction::COUNT],
        diagonal_corner_rule: DiagonalCornerRule,
        boundary_rule: BoundaryRule,
        max_radius: u8,
    ) -> Self {
        Self {
            slots,
            observations,
            target_masks,
            diagonal_corner_rule,
            boundary_rule,
            max_radius,
        }
    }

    /// Eight surrounding square-grid cells in NW, N, NE, W, E, SW, S, SE order.
    pub fn moore_8(boundary_rule: BoundaryRule) -> Self {
        const ORTHOGONAL: u16 = 1024;
        const DIAGONAL: u16 = 1448;
        let slots = vec![
            LocalOffset::new(-1, -1, DIAGONAL),
            LocalOffset::new(0, -1, ORTHOGONAL),
            LocalOffset::new(1, -1, DIAGONAL),
            LocalOffset::new(-1, 0, ORTHOGONAL),
            LocalOffset::new(1, 0, ORTHOGONAL),
            LocalOffset::new(-1, 1, DIAGONAL),
            LocalOffset::new(0, 1, ORTHOGONAL),
            LocalOffset::new(1, 1, DIAGONAL),
        ];
        let all = SlotMask::first(slots.len());
        Self::new(
            slots,
            ObservationMasks::all(8),
            [all; TargetingAction::COUNT],
            DiagonalCornerRule::Allow,
            boundary_rule,
            1,
        )
    }

    pub fn set_target_mask(&mut self, action: TargetingAction, mask: SlotMask) {
        self.target_masks[action as usize] = mask;
    }

    pub fn target_mask(&self, action: TargetingAction) -> SlotMask {
        self.target_masks[action as usize]
    }

    pub fn compile(
        self,
        width: usize,
        height: usize,
    ) -> Result<CompiledNeighborhood, NeighborhoodError> {
        CompiledNeighborhood::compile(self, width, height)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeighborhoodError {
    EmptyWorld,
    TooManySlots { count: usize, max: usize },
    ZeroOffset { slot: usize },
    ZeroDistanceCost { slot: usize },
    OffsetOutsideRadius { slot: usize, radius: u8 },
    DuplicateOffset { dx: i8, dy: i8 },
    MaskOutsideSlots { bits: u32, slot_count: usize },
    WorldTooLarge,
}

impl Display for NeighborhoodError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyWorld => write!(f, "world dimensions must be non-zero"),
            Self::TooManySlots { count, max } => {
                write!(f, "neighborhood has {count} slots; maximum is {max}")
            }
            Self::ZeroOffset { slot } => write!(f, "slot {slot} targets the actor itself"),
            Self::ZeroDistanceCost { slot } => {
                write!(f, "slot {slot} has a zero distance cost")
            }
            Self::OffsetOutsideRadius { slot, radius } => {
                write!(f, "slot {slot} exceeds maximum radius {radius}")
            }
            Self::DuplicateOffset { dx, dy } => {
                write!(f, "duplicate neighborhood offset ({dx}, {dy})")
            }
            Self::MaskOutsideSlots { bits, slot_count } => write!(
                f,
                "slot mask {bits:#034b} references a slot outside {slot_count} slots"
            ),
            Self::WorldTooLarge => write!(f, "world tile count overflows usize"),
        }
    }
}

impl Error for NeighborhoodError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledNeighborhood {
    spec: NeighborhoodSpec,
    width: usize,
    height: usize,
    neighbors: Vec<Option<TileIndex>>,
}

impl CompiledNeighborhood {
    fn compile(
        spec: NeighborhoodSpec,
        width: usize,
        height: usize,
    ) -> Result<Self, NeighborhoodError> {
        if width == 0 || height == 0 {
            return Err(NeighborhoodError::EmptyWorld);
        }
        if spec.slots.len() > MAX_LOCAL_SLOTS {
            return Err(NeighborhoodError::TooManySlots {
                count: spec.slots.len(),
                max: MAX_LOCAL_SLOTS,
            });
        }

        let mut offsets = BTreeSet::new();
        for (slot, offset) in spec.slots.iter().enumerate() {
            if offset.dx == 0 && offset.dy == 0 {
                return Err(NeighborhoodError::ZeroOffset { slot });
            }
            if offset.distance_cost_q10 == 0 {
                return Err(NeighborhoodError::ZeroDistanceCost { slot });
            }
            let radius = offset.dx.unsigned_abs().max(offset.dy.unsigned_abs());
            if radius > spec.max_radius {
                return Err(NeighborhoodError::OffsetOutsideRadius {
                    slot,
                    radius: spec.max_radius,
                });
            }
            if !offsets.insert((offset.dx, offset.dy)) {
                return Err(NeighborhoodError::DuplicateOffset {
                    dx: offset.dx,
                    dy: offset.dy,
                });
            }
        }

        let valid_mask = SlotMask::first(spec.slots.len()).bits();
        let masks = [
            spec.observations.occupancy,
            spec.observations.marker,
            spec.observations.apparent_mass,
            spec.observations.activity,
            spec.observations.terrain,
            spec.observations.energy,
            spec.observations.signal,
            spec.target_masks[0],
            spec.target_masks[1],
            spec.target_masks[2],
            spec.target_masks[3],
        ];
        for mask in masks {
            if mask.bits() & !valid_mask != 0 {
                return Err(NeighborhoodError::MaskOutsideSlots {
                    bits: mask.bits(),
                    slot_count: spec.slots.len(),
                });
            }
        }

        let tile_count = width
            .checked_mul(height)
            .ok_or(NeighborhoodError::WorldTooLarge)?;
        let entry_count = tile_count
            .checked_mul(spec.slots.len())
            .ok_or(NeighborhoodError::WorldTooLarge)?;
        let mut neighbors = Vec::with_capacity(entry_count);
        for tile in 0..tile_count {
            let x = tile % width;
            let y = tile / width;
            for offset in &spec.slots {
                neighbors.push(Self::resolve_offset(
                    width,
                    height,
                    spec.boundary_rule,
                    x,
                    y,
                    offset.dx,
                    offset.dy,
                ));
            }
        }

        Ok(Self {
            spec,
            width,
            height,
            neighbors,
        })
    }

    fn resolve_offset(
        width: usize,
        height: usize,
        boundary: BoundaryRule,
        x: usize,
        y: usize,
        dx: i8,
        dy: i8,
    ) -> Option<TileIndex> {
        let x = i64::try_from(x).ok()?;
        let y = i64::try_from(y).ok()?;
        let width_i64 = i64::try_from(width).ok()?;
        let height_i64 = i64::try_from(height).ok()?;
        let target_x = x + i64::from(dx);
        let target_y = y + i64::from(dy);

        let (resolved_x, resolved_y) = match boundary {
            BoundaryRule::Bounded => {
                if target_x < 0 || target_y < 0 || target_x >= width_i64 || target_y >= height_i64 {
                    return None;
                }
                (target_x, target_y)
            }
            BoundaryRule::Wrap => (
                target_x.rem_euclid(width_i64),
                target_y.rem_euclid(height_i64),
            ),
        };

        let resolved_x = usize::try_from(resolved_x).ok()?;
        let resolved_y = usize::try_from(resolved_y).ok()?;
        resolved_y
            .checked_mul(width)
            .and_then(|row| row.checked_add(resolved_x))
            .map(TileIndex)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn tile_count(&self) -> usize {
        self.width * self.height
    }

    pub fn slot_count(&self) -> usize {
        self.spec.slots.len()
    }

    pub fn spec(&self) -> &NeighborhoodSpec {
        &self.spec
    }

    pub fn tile(&self, x: usize, y: usize) -> Option<TileIndex> {
        if x < self.width && y < self.height {
            Some(TileIndex(y * self.width + x))
        } else {
            None
        }
    }

    pub fn coordinate(&self, tile: TileIndex) -> Option<(usize, usize)> {
        (tile.0 < self.tile_count()).then(|| (tile.0 % self.width, tile.0 / self.width))
    }

    pub fn offset(&self, slot: LocalSlot) -> Option<LocalOffset> {
        self.spec.slots.get(usize::from(slot.0)).copied()
    }

    pub fn target(&self, origin: TileIndex, slot: LocalSlot) -> Option<TileIndex> {
        if origin.0 >= self.tile_count() || usize::from(slot.0) >= self.slot_count() {
            return None;
        }
        self.neighbors[origin.0 * self.slot_count() + usize::from(slot.0)]
    }

    /// Returns every relative target for `origin` in canonical slot order.
    /// This avoids repeating bounds checks and row-offset arithmetic while
    /// projecting a complete local observation.
    pub fn targets(&self, origin: TileIndex) -> Option<&[Option<TileIndex>]> {
        if origin.0 >= self.tile_count() {
            return None;
        }
        let start = origin.0 * self.slot_count();
        Some(&self.neighbors[start..start + self.slot_count()])
    }

    pub fn target_at_offset(&self, origin: TileIndex, dx: i8, dy: i8) -> Option<TileIndex> {
        let (x, y) = self.coordinate(origin)?;
        Self::resolve_offset(
            self.width,
            self.height,
            self.spec.boundary_rule,
            x,
            y,
            dx,
            dy,
        )
    }

    pub fn action_allows(&self, action: TargetingAction, slot: LocalSlot) -> bool {
        self.spec.target_mask(action).contains(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_and_wrapped_edges_compile_differently() {
        let bounded = NeighborhoodSpec::moore_8(BoundaryRule::Bounded)
            .compile(3, 3)
            .unwrap();
        let wrapped = NeighborhoodSpec::moore_8(BoundaryRule::Wrap)
            .compile(3, 3)
            .unwrap();
        let top_left = bounded.tile(0, 0).unwrap();

        assert_eq!(bounded.target(top_left, LocalSlot(0)), None);
        assert_eq!(wrapped.target(top_left, LocalSlot(0)), wrapped.tile(2, 2));
    }

    #[test]
    fn per_action_masks_are_enforced() {
        let mut spec = NeighborhoodSpec::moore_8(BoundaryRule::Bounded);
        spec.set_target_mask(
            TargetingAction::Attack,
            SlotMask::from_slots([LocalSlot(1), LocalSlot(3), LocalSlot(4), LocalSlot(6)]),
        );
        let compiled = spec.compile(3, 3).unwrap();

        assert!(compiled.action_allows(TargetingAction::Attack, LocalSlot(1)));
        assert!(!compiled.action_allows(TargetingAction::Attack, LocalSlot(0)));
        assert!(compiled.action_allows(TargetingAction::Move, LocalSlot(0)));
    }
}
