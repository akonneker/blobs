//! Guard against batch-size-dependent approximate reciprocal gradients.
use burn::{
    backend::{Autodiff, NdArray},
    prelude::*,
};

#[test]
fn reciprocal_matches_scalar_across_vectorization_boundary() {
    for count in [1, 8, 31, 32, 33, 128, 256] {
        let values: Vec<f32> = (0..count).map(|i| (i % 8 + 1) as f32).collect();
        let actual = Tensor::<NdArray, 1>::from_data(
            TensorData::new(values.clone(), [count]),
            &Default::default(),
        )
        .recip()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        for (input, output) in values.iter().zip(actual) {
            assert!(
                (output - input.recip()).abs() < 1e-7,
                "length {count}, 1/{input}: expected {}, got {output}",
                input.recip()
            );
        }
    }
}

#[test]
fn masked_log_softmax_gradients_match_analytic_cross_entropy() {
    for count in [1, 31, 32, 33, 128] {
        for uniform in [true, false] {
            let mut values = Vec::new();
            let mut labels = Vec::new();
            let mut expected = Vec::new();
            for row in 0..count {
                let legal = row % 8 + 1;
                let scores: [f64; 8] = std::array::from_fn(|i| {
                    if i >= legal {
                        -1e9
                    } else if uniform {
                        0.
                    } else {
                        i as f64 * 0.25
                    }
                });
                let total: f64 = scores.iter().take(legal).map(|x| x.exp()).sum();
                for (slot, score) in scores.iter().enumerate() {
                    let target = if slot >= legal {
                        0.
                    } else if uniform {
                        1. / legal as f32
                    } else {
                        f32::from(slot == 0)
                    };
                    values.push(*score as f32);
                    labels.push(target);
                    expected.push(if slot < legal {
                        (score.exp() / total - f64::from(target)) / count as f64
                    } else {
                        0.
                    });
                }
            }
            let device = Default::default();
            let logits = Tensor::<Autodiff<NdArray>, 2>::from_data(
                TensorData::new(values, [count, 8]),
                &device,
            )
            .require_grad();
            let labels = Tensor::from_data(TensorData::new(labels, [count, 8]), &device);
            let loss = (burn::tensor::activation::log_softmax(logits.clone(), 1) * labels)
                .sum_dim(1)
                .mean()
                .neg();
            let actual = logits
                .grad(&loss.backward())
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (f64::from(*actual) - expected).abs() < 1e-7,
                    "batch {count}, uniform={uniform}: expected {expected}, got {actual}"
                );
            }
        }
    }
}
