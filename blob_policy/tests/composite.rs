#[path = "../examples/sampled_target_fixture/mod.rs"]
#[allow(dead_code)]
mod fixture;
use blob_interface::reference_mind::*;
use blob_policy::{action::action_is_commit_legal, composite::*, runtime::*};

fn policy(move_action: bool) -> CompositePolicy {
    let layers = layer_shapes(4, 3, 2)
        .unwrap()
        .into_iter()
        .map(|(name, i, o)| {
            let mut bias = vec![0.; o];
            if name.ends_with("action_kind_head") {
                bias[if move_action { 3 } else { 0 }] = 10.;
            }
            if name == "signal_head" {
                bias[1] = 10.;
            }
            Linear::new(i, o, vec![0.; i * o], bias).unwrap()
        })
        .collect();
    CompositePolicy {
        parent: FrozenPolicy::new(4, 3, 2, layers).unwrap(),
        utility: FrozenUtility::new(
            (0..UTILITY_PARAMETERS)
                .map(|i| ((i * 7 % 23) as f32 - 11.) * 0.002)
                .collect(),
        )
        .unwrap(),
    }
}
#[test]
fn envelope_is_explicit_bounded_and_rejects_corruption() {
    let p = policy(true);
    let bytes = p.to_bytes();
    assert_eq!(
        CompositePolicy::from_bytes(&bytes).unwrap().to_bytes(),
        bytes
    );
    assert_eq!(
        DeployedPolicy::from_bytes(&bytes)
            .unwrap()
            .execution_contract(),
        COMPOSITE_EXECUTION_CONTRACT
    );
    assert_eq!(
        DeployedPolicy::from_bytes(&p.parent.to_bytes())
            .unwrap()
            .execution_contract(),
        EXECUTION_CONTRACT
    );
    for cut in [0, 8, 11, 12, bytes.len() - 1] {
        assert!(CompositePolicy::from_bytes(&bytes[..cut]).is_err());
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(CompositePolicy::from_bytes(&bad).is_err());
    let mut bad = bytes.clone();
    bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(CompositePolicy::from_bytes(&bad).is_err());
    let mut bad = bytes;
    let n = bad.len();
    bad[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(CompositePolicy::from_bytes(&bad).is_err());
}
#[test]
fn composed_moves_preserve_parent_outputs_and_reserve_signal_cost() {
    let p = policy(true);
    for count in 1..=32 {
        let mut input = fixture::input(count);
        input.action_space.signal_enabled = true;
        input.action_space.signal_emission_cost = 2;
        input.self_state.assimilated_energy = 10;
        for slot in &mut input.slots {
            slot.distance_cost_q10 = if slot.slot == 0 { 1024 } else { 20 * 1024 };
        }
        let parent = p.parent.try_decide(&input).unwrap();
        assert!(matches!(parent.action, ReferenceMindAction::Move { .. }));
        assert!(parent.signal.is_some());
        let mut actual = p.try_decide(&input).unwrap();
        assert!(action_is_commit_legal(
            &input,
            &actual.action,
            actual.signal.is_some()
        ));
        if let (
            ReferenceMindAction::Move {
                target_slot,
                effort,
            },
            ReferenceMindAction::Move {
                target_slot: before,
                effort: before_effort,
            },
        ) = (&mut actual.action, &parent.action)
        {
            assert_eq!(effort, before_effort);
            *target_slot = *before;
        }
        assert_eq!(actual, parent);
    }
    let p = policy(false);
    let input = fixture::input(8);
    assert_eq!(
        p.try_decide(&input).unwrap(),
        p.parent.try_decide(&input).unwrap()
    );
}
#[test]
fn target_affordable_alone_is_excluded_when_parent_signal_needs_energy() {
    let p = policy(true);
    let mut input = fixture::input(2);
    input.self_state.assimilated_energy = 10;
    input.action_space.signal_emission_cost = 3;
    input.slots[0].distance_cost_q10 = 1024;
    let parent = p.parent.try_decide(&input).unwrap();
    let ReferenceMindAction::Move { effort, .. } = parent.action else {
        panic!("fixture must Move")
    };
    assert!(parent.signal.is_some());
    let alternative = ReferenceMindAction::Move {
        target_slot: 1,
        effort,
    };
    let distance = (1024..=u16::MAX)
        .find(|cost| {
            input.slots[1].distance_cost_q10 = *cost;
            action_is_commit_legal(&input, &alternative, false)
                && !action_is_commit_legal(&input, &alternative, true)
        })
        .expect("fixture has signal-only affordability boundary");
    input.slots[1].distance_cost_q10 = distance;
    input.randomness = blob_interface::randomness::PrivateRandom::from_bytes([255; 32]);
    let actual = p.try_decide(&input).unwrap();
    assert_eq!(
        actual.action,
        ReferenceMindAction::Move {
            target_slot: 0,
            effort
        }
    );
    assert_eq!(actual.signal, parent.signal);
}
#[test]
fn learned_utility_is_slot_equivariant_and_masks_competitors() {
    let p = policy(true);
    let mut local = vec![0.; 32 * 33];
    for (i, v) in local.iter_mut().enumerate() {
        *v = (i % 17) as f32 / 19.;
    }
    let mask = std::array::from_fn(|i| i < 8);
    let before = p.utility.forward(&local, &mask).unwrap();
    local[8 * 33..].fill(17.);
    assert_eq!(
        &before[..8],
        &p.utility.forward(&local, &mask).unwrap()[..8]
    );
    let reversed: Vec<_> = local.chunks_exact(33).rev().flatten().copied().collect();
    let result = p
        .utility
        .forward(&reversed, &std::array::from_fn(|i| mask[31 - i]))
        .unwrap();
    for i in 0..8 {
        assert_eq!(before[i].to_bits(), result[31 - i].to_bits());
    }
    assert!(p.utility.forward(&[], &mask).is_err());
}
