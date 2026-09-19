#[path = "../examples/sampled_target_fixture/mod.rs"]
#[allow(dead_code)]
mod fixture;
use blob_policy::{
    composite::*,
    observation::*,
    relational::*,
    runtime::{layer_shapes, FrozenPolicy, Linear, PolicyOutput},
};
fn base() -> CompositePolicy {
    let layers = layer_shapes(4, 3, 2)
        .unwrap()
        .into_iter()
        .map(|(name, i, o)| {
            let mut bias = vec![0.; o];
            if name.ends_with("action_kind_head") {
                bias[3] = 10.;
            }
            Linear::new(i, o, vec![0.01; i * o], bias).unwrap()
        })
        .collect();
    CompositePolicy {
        parent: FrozenPolicy::new(4, 3, 2, layers).unwrap(),
        utility: FrozenUtility::new(vec![0.; UTILITY_PARAMETERS]).unwrap(),
    }
}
fn bits(out: &PolicyOutput) -> Vec<u32> {
    [
        &out.kind,
        &out.target,
        &out.effort,
        &out.amount,
        &out.signal,
        &out.signal_strength,
        &out.next_memory,
    ]
    .into_iter()
    .flatten()
    .map(|x| x.to_bits())
    .collect()
}
#[test]
fn zero_migration_preserves_every_output_and_complete_move_decision() {
    let p = RelationalPolicy {
        base: base(),
        residual: FrozenResidual::new(vec![0.; PARAMETERS]).unwrap(),
    };
    for count in 1..=32 {
        let input = fixture::input(count);
        assert_eq!(
            p.try_decide(&input).unwrap(),
            p.base.try_decide(&input).unwrap()
        );
        for context in 0..3 {
            let mut ordinary = input.clone();
            if context != 0 {
                ordinary.current_tile.plant_energy = 0;
                ordinary.current_tile.plant_capacity = 0;
                ordinary.current_tile.loose_energy = 0;
            }
            if context == 1 {
                ordinary.slots[0].neighbor =
                    Some(blob_interface::reference_mind::ReferenceNeighbor {
                        marker: Some(2),
                        apparent_mass_bucket: Some(1),
                        activity: Some(blob_interface::reference_mind::ReferenceActivity::Ready),
                        progress: None,
                    });
            }
            assert_eq!(
                p.try_decide(&ordinary).unwrap(),
                p.base.try_decide(&ordinary).unwrap()
            );
            let mut obs = Observation::from_reference(&input).data;
            obs[11] = 0.;
            obs[24] = 0.;
            if context == 0 {
                obs[24] = 0.1;
            }
            if context == 1 {
                obs[70 + 13] = 1.;
            }
            let old = p.base.parent.forward(&obs, &[0.2, -0.3]).unwrap();
            let new = p.forward(&obs, &[0.2, -0.3]).unwrap();
            assert_eq!(bits(&old), bits(&new));
        }
    }
}
#[test]
fn active_residual_changes_only_interaction_kind_scores() {
    let mut v = vec![0.; PARAMETERS];
    v[PARAMETERS - 10 + 4] = 20.;
    let p = RelationalPolicy {
        base: base(),
        residual: FrozenResidual::new(v).unwrap(),
    };
    for context in 0..3 {
        let mut obs = Observation::from_reference(&fixture::input(8)).data;
        obs[11] = 0.;
        obs[24] = 0.;
        if context == 0 {
            obs[24] = 0.1;
        }
        if context == 1 {
            obs[70 + 13] = 1.;
        }
        let old = p.base.parent.forward(&obs, &[0.2, -0.3]).unwrap();
        let mut new = p.forward(&obs, &[0.2, -0.3]).unwrap();
        if context == 1 {
            assert_ne!(old.kind, new.kind);
            new.kind = old.kind.clone();
        }
        assert_eq!(bits(&old), bits(&new));
    }
    assert!(p
        .base
        .parent
        .forward_with_kind_delta(&[0.; OBS_DIM], &[0.; 2], &[f32::NAN; 10])
        .is_err());
}
#[test]
fn envelope_rejects_wrong_type_lengths_and_nonfinite_weights() {
    let p = RelationalPolicy {
        base: base(),
        residual: FrozenResidual::new(vec![0.; PARAMETERS]).unwrap(),
    };
    let b = p.to_bytes();
    assert_eq!(RelationalPolicy::from_bytes(&b).unwrap().to_bytes(), b);
    assert_eq!(
        DeployedPolicy::from_bytes(&b).unwrap().execution_contract(),
        EXECUTION_CONTRACT
    );
    assert!(CompositePolicy::from_bytes(&b).is_err());
    for cut in [0, 8, 11, 12, b.len() - 1] {
        assert!(RelationalPolicy::from_bytes(&b[..cut]).is_err());
    }
    let mut bad = b.clone();
    bad.push(0);
    assert!(RelationalPolicy::from_bytes(&bad).is_err());
    let mut bad = b.clone();
    bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(RelationalPolicy::from_bytes(&bad).is_err());
    let mut bad = b.clone();
    let n = bad.len();
    bad[n - 4..].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(RelationalPolicy::from_bytes(&bad).is_err());
    let obs = vec![f32::NAN; OBS_DIM];
    assert!(p.forward(&obs, &[0.; 2]).is_err());
}

#[test]
fn encoder_distinguishes_food_occupant_pairings_erased_by_raw_maxima() {
    let mut a = vec![0.; OBS_DIM];
    a[70] = 1.;
    a[103] = 1.;
    a[70 + 8] = 1.;
    a[103 + 13] = 1.;
    let mut b = a.clone();
    b[103 + 13] = 0.;
    b[70 + 13] = 1.;
    let maxima = |obs: &[f32]| {
        (0..33)
            .map(|i| {
                obs[70..]
                    .chunks_exact(33)
                    .map(|s| s[i])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(maxima(&a), maxima(&b));
    let mut values = vec![0.; PARAMETERS];
    values[8 * 16] = 1.;
    values[13 * 16] = 1.; // encode the joint food/occupant sum
    values[544] = 1.; // second encoder, first feature
    values[544 + 272 + 38 * 32] = 1.; // first mean feature into the hidden layer
    values[544 + 272 + 2272 + 4] = 1.; // hidden feature into Attack delta
    let residual = FrozenResidual::new(values).unwrap();
    assert_ne!(residual.forward(&a).unwrap(), residual.forward(&b).unwrap());
}
