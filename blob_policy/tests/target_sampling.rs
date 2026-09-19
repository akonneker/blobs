use blob_interface::randomness::PrivateRandom;
use blob_policy::{action::*, sampling::*};
#[path = "../examples/sampled_target_fixture/mod.rs"]
mod fixture;

fn boundary(cumulative: u64, total: u64) -> u64 {
    ((u128::from(cumulative) << 64).div_ceil(u128::from(total))) as u64
}

#[test]
fn uniform_selection_matches_exact_quantile_for_every_small_subset_and_full_capacity() {
    for count in 1..=32 {
        let masks: Vec<u32> = if count == 8 {
            (1..=255).collect()
        } else {
            vec![if count == 32 {
                u32::MAX
            } else {
                (1 << count) - 1
            }]
        };
        for mask in masks {
            let legal: Vec<_> = (0..count).map(|i| mask & (1 << i) != 0).collect();
            let candidates: Vec<_> = (0..count).filter(|i| legal[*i]).collect();
            let distribution =
                CategoricalDistribution::from_logits(&vec![1.25; count], &legal).unwrap();
            let mut words = vec![0, u64::MAX];
            for k in 1..candidates.len() {
                let point = boundary(k as u64, candidates.len() as u64);
                words.extend([point - 1, point, point + 1]);
            }
            for word in words {
                let expected =
                    candidates[((u128::from(word) * candidates.len() as u128) >> 64) as usize];
                assert_eq!(distribution.sample(word), Some(expected));
            }
        }
    }
}

#[test]
fn weighted_boundaries_extremes_and_input_errors_are_explicit() {
    let distribution = CategoricalDistribution::from_logits(&[0., -1.], &[true, true]).unwrap();
    assert_eq!(distribution.weights(), &[281474976710656, 103548857136061]);
    let point = boundary(distribution.weights()[0], distribution.total());
    assert_eq!(distribution.sample(point - 1), Some(0));
    assert_eq!(distribution.sample(point), Some(1));
    assert_eq!(distribution.sample(u64::MAX), Some(1));
    for values in [[f32::MAX, -f32::MAX], [0., -1000.]] {
        let d = CategoricalDistribution::from_logits(&values, &[true, true]).unwrap();
        assert_eq!(d.weights(), &[1 << 48, 0]);
        assert_eq!(d.sample(u64::MAX), Some(0));
    }
    assert_eq!(
        CategoricalDistribution::from_logits(&[f32::NAN], &[false])
            .unwrap()
            .sample(0),
        None
    );
    assert_eq!(
        CategoricalDistribution::from_logits(&[], &[])
            .unwrap()
            .sample(u64::MAX),
        None
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(CategoricalDistribution::from_logits(&[bad], &[true]).is_err());
    }
    assert!(CategoricalDistribution::from_logits(&[0.], &[]).is_err());
    assert!(CategoricalDistribution::from_logits(&[0.; 33], &[true; 33]).is_err());
}

#[test]
fn ordinary_input_masks_and_geometry_order_are_independent_of_slot_numbering() {
    let mut input = fixture::input(32);
    input.action_space.move_targets = 0x5555_5555;
    input.slots[2].reachable = false;
    let mut shuffled = input.clone();
    shuffled.slots.reverse();
    for (index, slot) in shuffled.slots.iter_mut().enumerate() {
        slot.slot = index as u8;
    }
    shuffled.action_space.move_targets = input.action_space.move_targets.reverse_bits();
    let left =
        TargetDistribution::from_input(&input, PolicyActionKind::Move, &fixture::logits(&input))
            .unwrap();
    let right = TargetDistribution::from_input(
        &shuffled,
        PolicyActionKind::Move,
        &fixture::logits(&shuffled),
    )
    .unwrap();
    assert_eq!(left.categorical().weights(), right.categorical().weights());
    for word in [0, 1, 1 << 63, u64::MAX] {
        let a = usize::from(left.sample(word).unwrap());
        let b = usize::from(right.sample(word).unwrap());
        assert_eq!(
            (input.slots[a].dx, input.slots[a].dy),
            (shuffled.slots[b].dx, shuffled.slots[b].dy)
        );
        assert_ne!(a, 2);
        assert_eq!(a % 2, 0);
    }
    assert!(action_is_commit_legal(
        &input,
        &fixture::decide(&input).unwrap().action,
        false
    ));
    input.action_space.move_targets = 0;
    assert_eq!(
        sample_target(&input, PolicyActionKind::Move, &[0.; 32]).unwrap(),
        None
    );
    input.action_space.move_targets = u32::MAX;
    input.self_state.assimilated_energy = 1;
    assert_eq!(
        sample_target(&input, PolicyActionKind::Move, &[0.; 32]).unwrap(),
        None
    );
}

#[test]
fn malformed_native_inputs_fail_before_masks_or_indexing() {
    let input = fixture::input(8);
    let mut bad = input.clone();
    bad.slots[0].slot = 32;
    assert!(sample_target(&bad, PolicyActionKind::Move, &[0.; 32]).is_err());
    let mut bad = input.clone();
    bad.slots[0].dx = 0;
    bad.slots[0].dy = 0;
    assert!(sample_target(&bad, PolicyActionKind::Move, &[0.; 32]).is_err());
    let mut bad = input.clone();
    bad.slots[0].dx = bad.slots[1].dx;
    bad.slots[0].dy = bad.slots[1].dy;
    assert!(sample_target(&bad, PolicyActionKind::Move, &[0.; 32]).is_err());
    let mut bad = input.clone();
    bad.action_space.move_targets = 1 << 31;
    assert!(sample_target(&bad, PolicyActionKind::Move, &[0.; 32]).is_err());
    let mut bad = input.clone();
    bad.action_space.effort_cost_denominators[0] = 0;
    assert!(sample_target(&bad, PolicyActionKind::Move, &[0.; 32]).is_err());
    assert!(sample_target(&input, PolicyActionKind::Wait, &[0.; 32]).is_err());
    assert!(sample_target(&input, PolicyActionKind::Move, &[f32::NAN; 32]).is_err());
}

#[test]
fn all_target_kinds_use_their_own_mask_and_only_word_zero_drives_sampling() {
    let mut input = fixture::input(8);
    input.action_space.move_targets = 1;
    input.action_space.attack_targets = 2;
    input.action_space.split_targets = 4;
    input.action_space.regurgitate_targets = 8;
    for (kind, expected) in [
        (PolicyActionKind::Move, 0),
        (PolicyActionKind::Attack, 1),
        (PolicyActionKind::Split, 2),
        (PolicyActionKind::Regurgitate, 3),
    ] {
        assert_eq!(
            sample_target(&input, kind, &[0.; 32]).unwrap(),
            Some(expected)
        );
    }
    input.action_space.move_targets = 255;
    let mut random = [255; 32];
    random[..8].copy_from_slice(&0_u64.to_le_bytes());
    input.randomness = PrivateRandom::from_bytes(random);
    let selected = sample_target(&input, PolicyActionKind::Move, &[0.; 32]).unwrap();
    input.randomness = PrivateRandom::ZERO;
    input.private_memory = vec![7; 31];
    assert_eq!(
        sample_target(&input, PolicyActionKind::Move, &[0.; 32]).unwrap(),
        selected
    );
    input.randomness = PrivateRandom::from_bytes([255; 32]);
    assert_ne!(
        sample_target(&input, PolicyActionKind::Move, &[0.; 32]).unwrap(),
        selected
    );
}
