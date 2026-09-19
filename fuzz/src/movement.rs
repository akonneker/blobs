//! One-move oracle for slot admission, occupied targets, elevation and corners.

use blob_engine::resolution::{
    ActionRequest, BoundaryRule, DiagonalCornerRule, EffortTier, LocalSlot, NeighborhoodSpec,
    OutcomeStatus, ReferenceRuleset, ReferenceSimulation, RejectReason, SlotMask, TargetingAction,
    TileIndex,
};

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or(0)
}

/// At most 25 tiles, eight slots and one actor commitment/resolution per input.
pub fn check(data: &[u8]) {
    if data.len() > 64 {
        return;
    }
    let width = 1 + usize::from(byte(data, 0) % 5);
    let height = 1 + usize::from(byte(data, 1) % 5);
    let origin = usize::from(byte(data, 3)) % (width * height);
    let wrap = byte(data, 2) & 1 != 0;
    let corner = byte(data, 2) / 2 % 3;
    let count = usize::from(byte(data, 4) % 9);
    let selected = byte(data, 5) % 11;
    let mask_bits = byte(data, 6);
    let mut spec = NeighborhoodSpec::moore_8(if wrap {
        BoundaryRule::Wrap
    } else {
        BoundaryRule::Bounded
    });
    spec.slots.truncate(count);
    spec.observations = blob_engine::resolution::ObservationMasks::all(count);
    // Truncation requires every independent action mask to be bounded too.
    for action in [
        TargetingAction::Move,
        TargetingAction::Attack,
        TargetingAction::Split,
        TargetingAction::Regurgitate,
    ] {
        spec.set_target_mask(
            action,
            SlotMask::from_slots(
                (0..count as u8)
                    .filter(|slot| mask_bits & (1 << slot) != 0)
                    .map(LocalSlot),
            ),
        );
    }
    spec.diagonal_corner_rule = [
        DiagonalCornerRule::Allow,
        DiagonalCornerRule::BlockIfEitherOrthogonalOccupied,
        DiagonalCornerRule::BlockIfBothOrthogonalsOccupied,
    ][usize::from(corner)];
    let rules = ReferenceRuleset {
        neighborhood: spec.clone(),
        diffusion_targets: SlotMask::empty(),
        metabolism_rate_numerator: 0,
        digestion_rate_numerator: 0,
        maximum_elevation_delta: i16::from(byte(data, 11) % 5),
        ..ReferenceRuleset::default()
    };
    let mut simulation = ReferenceSimulation::new(width, height, rules.clone()).unwrap();
    let actor = simulation
        .add_cell(TileIndex(origin), 20, 10_000, 1)
        .unwrap();
    let occupancy = u32::from_le_bytes(std::array::from_fn(|index| byte(data, 7 + index)));
    for index in 0..width * height {
        if index != origin && occupancy & (1 << index) != 0 {
            simulation
                .add_cell(TileIndex(index), 20, 10_000, 2)
                .unwrap();
        }
        simulation
            .tile_state_mut(TileIndex(index))
            .unwrap()
            .elevation = i16::from(byte(data, 12 + index) as i8);
    }
    let state = simulation.canonical_state();
    let total = simulation.total_energy_equivalent();
    let target_at = |dx: i8, dy: i8| {
        let x = (origin % width) as i32 + i32::from(dx);
        let y = (origin / width) as i32 + i32::from(dy);
        if wrap {
            Some(y.rem_euclid(height as i32) as usize * width + x.rem_euclid(width as i32) as usize)
        } else if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
            Some(y as usize * width + x as usize)
        } else {
            None
        }
    };
    let offset = spec.slots.get(usize::from(selected));
    let target = offset.and_then(|offset| target_at(offset.dx, offset.dy));
    let rejection = if offset.is_none() {
        Some(RejectReason::InvalidSlot)
    } else if mask_bits & (1 << selected) == 0 {
        Some(RejectReason::ActionNotAllowedInSlot)
    } else if target.is_none() {
        Some(RejectReason::TargetOutsideWorld)
    } else if target == Some(origin) {
        Some(RejectReason::TargetsSelf)
    } else {
        None
    };
    let blocked_corner = offset.is_some_and(|offset| {
        if offset.dx == 0 || offset.dy == 0 {
            return false;
        }
        let occupied = |target: Option<usize>| {
            target.is_none_or(|target| state.tiles[target].occupant.is_some())
        };
        let horizontal = occupied(target_at(offset.dx, 0));
        let vertical = occupied(target_at(0, offset.dy));
        match corner {
            0 => false,
            1 => horizontal || vertical,
            _ => horizontal && vertical,
        }
    });
    let success = rejection.is_none()
        && !blocked_corner
        && target.is_some_and(|target| {
            state.tiles[target].occupant.is_none()
                && (i32::from(state.tiles[target].elevation)
                    - i32::from(state.tiles[origin].elevation))
                .abs()
                    <= i32::from(rules.maximum_elevation_delta)
        });
    let receipt = simulation
        .commit_action(
            actor,
            ActionRequest::Move {
                target: LocalSlot(selected),
                effort: EffortTier::Standard,
            },
        )
        .unwrap();
    assert_eq!(receipt.rejection, rejection);
    assert_eq!(receipt.accepted, rejection.is_none());
    let report = simulation.resolve_next_batch().unwrap();
    assert_eq!(report.outcomes.len(), 1);
    let expected_status = rejection.map_or(
        if success {
            OutcomeStatus::Success
        } else {
            OutcomeStatus::Frustrated
        },
        OutcomeStatus::Rejected,
    );
    assert_eq!(report.outcomes[0].status, expected_status);
    assert_eq!(
        simulation.cell(actor).unwrap().position.0,
        if success { target.unwrap() } else { origin }
    );
    assert_eq!(simulation.total_energy_equivalent(), total);
    let after = simulation.canonical_state();
    assert_eq!(
        simulation.state_hash(),
        after.hash_with_compiled_ruleset(simulation.compiled_ruleset_hash())
    );
    let mut reversed = after;
    report.delta.apply_backward(&mut reversed).unwrap();
    report.delta.apply_forward(&mut reversed).unwrap();
    assert_eq!(reversed, simulation.canonical_state());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movement_matches_corner_and_slot_oracle() {
        for corner in 0..3 {
            for wrap in 0..2 {
                for occupancy in 0..512_u32 {
                    let mut data = [0; 64];
                    data[..7].copy_from_slice(&[2, 2, corner * 2 + wrap, 4, 8, 7, 255]);
                    data[7..11].copy_from_slice(&occupancy.to_le_bytes());
                    check(&data);
                }
            }
        }
        let mut random = 0x1289_aefd_3291_u64;
        for _ in 0..512 {
            let data: [u8; 64] = std::array::from_fn(|_| {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                random as u8
            });
            check(&data);
        }
    }
}
