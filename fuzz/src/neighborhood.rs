//! Bounded topology generation with a separate wide-integer geometry oracle.

use blob_engine::resolution::{
    BoundaryRule, DiagonalCornerRule, LocalOffset, LocalSlot, NeighborhoodSpec, ObservationMasks,
    SlotMask, TargetingAction, TileIndex,
};
use std::collections::BTreeSet;

const ACTIONS: [TargetingAction; 4] = [
    TargetingAction::Move,
    TargetingAction::Attack,
    TargetingAction::Split,
    TargetingAction::Regurgitate,
];

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or(0)
}

fn mask(bits: u32) -> SlotMask {
    SlotMask::from_slots(
        (0..32)
            .filter(|slot| bits & (1 << slot) != 0)
            .map(LocalSlot),
    )
}

fn verify(spec: NeighborhoodSpec, width: usize, height: usize) {
    let count = spec.slots.len();
    let valid_bits = if count >= 32 {
        u32::MAX
    } else {
        (1_u32 << count) - 1
    };
    let masks = [
        spec.observations.occupancy,
        spec.observations.marker,
        spec.observations.apparent_mass,
        spec.observations.activity,
        spec.observations.terrain,
        spec.observations.energy,
        spec.observations.signal,
        spec.target_mask(ACTIONS[0]),
        spec.target_mask(ACTIONS[1]),
        spec.target_mask(ACTIONS[2]),
        spec.target_mask(ACTIONS[3]),
    ];
    let unique: BTreeSet<_> = spec
        .slots
        .iter()
        .map(|offset| (offset.dx, offset.dy))
        .collect();
    let valid = width > 0
        && height > 0
        && width.checked_mul(height).is_some()
        && count <= 32
        && unique.len() == count
        && spec.slots.iter().all(|offset| {
            (offset.dx != 0 || offset.dy != 0)
                && offset.distance_cost_q10 > 0
                && i16::from(offset.dx).abs() <= i16::from(spec.max_radius)
                && i16::from(offset.dy).abs() <= i16::from(spec.max_radius)
        })
        && masks.iter().all(|mask| mask.bits() & !valid_bits == 0);
    // Nonempty generated worlds have at most 64 tiles. Huge virtual worlds
    // always have zero slots, so they require no neighbor allocation.
    assert!(count == 0 || width <= 8 && height <= 8);
    let compiled = spec.clone().compile(width, height);
    assert_eq!(
        compiled.is_ok(),
        valid,
        "{width}x{height}, {spec:?}: {compiled:?}"
    );
    let Ok(compiled) = compiled else {
        return;
    };
    assert_eq!(compiled.spec(), &spec);
    assert_eq!(compiled.tile_count(), width * height);
    let area = compiled.tile_count();
    let origins: Vec<_> = if area <= 64 {
        (0..area).collect()
    } else {
        vec![0, 1, area / 2, area - 1]
    };
    for origin in origins {
        let x = origin % width;
        let y = origin / width;
        assert_eq!(compiled.coordinate(TileIndex(origin)), Some((x, y)));
        assert_eq!(compiled.tile(x, y), Some(TileIndex(origin)));
        let expected = |dx: i8, dy: i8| {
            let mut tx = x as i128 + i128::from(dx);
            let mut ty = y as i128 + i128::from(dy);
            if spec.boundary_rule == BoundaryRule::Wrap {
                tx = tx.rem_euclid(width as i128);
                ty = ty.rem_euclid(height as i128);
            }
            if tx < 0 || ty < 0 || tx >= width as i128 || ty >= height as i128 {
                None
            } else {
                Some(TileIndex(ty as usize * width + tx as usize))
            }
        };
        let targets = compiled.targets(TileIndex(origin)).unwrap();
        assert_eq!(targets.len(), count);
        for (index, offset) in spec.slots.iter().enumerate() {
            let slot = LocalSlot(index as u8);
            assert_eq!(compiled.offset(slot), Some(*offset));
            assert_eq!(targets[index], expected(offset.dx, offset.dy));
            assert_eq!(compiled.target(TileIndex(origin), slot), targets[index]);
            assert_eq!(
                compiled.target_at_offset(TileIndex(origin), offset.dx, offset.dy),
                targets[index]
            );
        }
        for (dx, dy) in [(127, 0), (-128, 0), (0, 127), (0, -128), (0, 0)] {
            assert_eq!(
                compiled.target_at_offset(TileIndex(origin), dx, dy),
                expected(dx, dy)
            );
        }
        assert_eq!(compiled.target(TileIndex(origin), LocalSlot(255)), None);
    }
    assert_eq!(compiled.coordinate(TileIndex(area)), None);
    assert_eq!(compiled.targets(TileIndex(area)), None);
    assert_eq!(compiled.tile(width, 0), None);
    assert_eq!(compiled.tile(0, height), None);
    for action in ACTIONS {
        for slot in 0..=32 {
            assert_eq!(
                compiled.action_allows(action, LocalSlot(slot)),
                slot < 32 && spec.target_mask(action).bits() & (1 << slot) != 0
            );
        }
    }
}

pub fn check(data: &[u8]) {
    if data.len() > 256 {
        return;
    }
    let flags = byte(data, 2);
    let boundary = if flags & 1 == 0 {
        BoundaryRule::Bounded
    } else {
        BoundaryRule::Wrap
    };
    let count = usize::from(byte(data, 4) % 34);
    let candidates: Vec<_> = (-3..=3_i8)
        .flat_map(|y| (-3..=3_i8).map(move |x| (x, y)))
        .filter(|offset| *offset != (0, 0))
        .collect();
    let slots = (0..count)
        .map(|index| {
            let start = 49 + index * 4;
            let (dx, dy) = if flags & 2 == 0 {
                candidates[index]
            } else {
                (byte(data, start) as i8, byte(data, start + 1) as i8)
            };
            let cost = u16::from_le_bytes([byte(data, start + 2), byte(data, start + 3)]);
            LocalOffset::new(dx, dy, if flags & 4 == 0 { cost.max(1) } else { cost })
        })
        .collect();
    let valid_bits = if count >= 32 {
        u32::MAX
    } else {
        (1_u32 << count) - 1
    };
    let masks: Vec<_> = (0..11)
        .map(|index| {
            let bits =
                u32::from_le_bytes(std::array::from_fn(|part| byte(data, 5 + index * 4 + part)));
            mask(if flags & 8 == 0 {
                bits & valid_bits
            } else {
                bits
            })
        })
        .collect();
    let spec = NeighborhoodSpec::new(
        slots,
        ObservationMasks {
            occupancy: masks[0],
            marker: masks[1],
            apparent_mass: masks[2],
            activity: masks[3],
            terrain: masks[4],
            energy: masks[5],
            signal: masks[6],
        },
        [masks[7], masks[8], masks[9], masks[10]],
        [
            DiagonalCornerRule::Allow,
            DiagonalCornerRule::BlockIfEitherOrthogonalOccupied,
            DiagonalCornerRule::BlockIfBothOrthogonalsOccupied,
        ][usize::from(flags / 16 % 3)],
        boundary,
        byte(data, 3),
    );
    verify(
        spec,
        usize::from(byte(data, 0) % 9),
        usize::from(byte(data, 1) % 9),
    );
    if flags & 128 != 0 {
        let empty = NeighborhoodSpec::new(
            vec![],
            ObservationMasks::all(0),
            [SlotMask::empty(); 4],
            DiagonalCornerRule::Allow,
            boundary,
            128,
        );
        let signed_max = (i64::MAX as u64).min(usize::MAX as u64) as usize;
        let dimensions = [
            (usize::MAX, 1),
            (1, usize::MAX),
            (signed_max, 1),
            (usize::MAX, 2),
        ];
        let (width, height) = dimensions[usize::from(byte(data, 0) % 4)];
        verify(empty, width, height);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_geometry_and_mask_contract() {
        for (radius, offsets) in [
            (3, [0, 0, 1, 0, 1, 0, 1, 0]),
            (3, [1, 0, 0, 0, 0, 1, 1, 0]),
            (3, [1, 0, 1, 0, 1, 0, 2, 0]),
            (1, [2, 0, 1, 0, 0, 1, 1, 0]),
        ] {
            let mut input = vec![0; 181];
            input[..5].copy_from_slice(&[3, 3, 6, radius, 2]);
            input[49..57].copy_from_slice(&offsets);
            check(&input);
        }
        for width in 0..=8 {
            for height in 0..=8 {
                for flags in [0, 1, 8, 9, 16, 33, 128, 129] {
                    for count in [0, 1, 8, 31, 32, 33] {
                        let mut input = vec![255; 181];
                        input[..5].copy_from_slice(&[width, height, flags, 3, count]);
                        check(&input);
                    }
                }
            }
        }
        for seed in 0..128_u64 {
            let mut random = seed + 1;
            let input: Vec<_> = (0..181)
                .map(|_| {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    random as u8
                })
                .collect();
            check(&input);
        }
    }
}
