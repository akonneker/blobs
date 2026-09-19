//! Supervision uses the recorded random draw; inference features never do.
use super::data::Row;
use burn::prelude::*;
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Objective {
    CrossEntropy,
    QuantileInterval,
}

pub fn loss<B: Backend>(
    logits: Tensor<B, 2>,
    rows: &[Row],
    control: bool,
    objective: Objective,
) -> Tensor<B, 1> {
    let device = logits.device();
    let labels: Vec<_> = rows
        .iter()
        .map(|r| if control { r.control } else { r.treatment })
        .collect();
    let log_probabilities = burn::tensor::activation::log_softmax(logits, 1);
    match objective {
        Objective::CrossEntropy => {
            let labels = Tensor::<B, 1, Int>::from_data(
                TensorData::new(
                    labels.iter().map(|x| *x as i32).collect::<Vec<_>>(),
                    [rows.len()],
                ),
                &device,
            );
            log_probabilities
                .gather(1, labels.unsqueeze_dim(1))
                .mean()
                .neg()
        }
        Objective::QuantileInterval => {
            let mut before = vec![0.; rows.len() * 32];
            let mut through = before.clone();
            for (index, (row, label)) in rows.iter().zip(&labels).enumerate() {
                let position = row
                    .geometry_order
                    .iter()
                    .position(|slot| slot == label)
                    .expect("label has ordinary slot geometry");
                for (rank, &slot) in row.geometry_order.iter().enumerate() {
                    before[index * 32 + slot] = f32::from(rank < position);
                    through[index * 32 + slot] = f32::from(rank <= position);
                }
            }
            let before =
                Tensor::<B, 2>::from_data(TensorData::new(before, [rows.len(), 32]), &device);
            let through =
                Tensor::<B, 2>::from_data(TensorData::new(through, [rows.len(), 32]), &device);
            let quantile = Tensor::<B, 2>::from_data(
                TensorData::new(
                    rows.iter()
                        .map(|r| (r.word as f64 / 18446744073709551616.) as f32)
                        .collect::<Vec<_>>(),
                    [rows.len(), 1],
                ),
                &device,
            );
            let probabilities = log_probabilities.exp();
            let lower = (probabilities.clone() * before).sort(1).sum_dim(1);
            let upper = (probabilities * through).sort(1).sum_dim(1);
            // Small interior margin exceeds f32 rounding at the loss boundary;
            // actual qualification still uses exact u64 / portable Q48 decoding.
            (burn::tensor::activation::relu(lower - quantile.clone() + 1e-6)
                + burn::tensor::activation::relu(quantile - upper + 1e-6))
            .mean()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    #[test]
    fn recorded_quantile_supervision_pushes_both_interval_boundaries_correctly() {
        for (target, word, expected) in [
            (0, 3_u64 << 62, [-0.25, 0.25]),
            (1, 1_u64 << 62, [0.25, -0.25]),
        ] {
            let device = Default::default();
            let mut logits = vec![-1e9; 32];
            logits[0] = 0.;
            logits[1] = 0.;
            let logits = Tensor::<Autodiff<NdArray>, 2>::from_data(
                TensorData::new(logits, [1, 32]),
                &device,
            )
            .require_grad();
            let row = Row {
                seed: 0,
                layout: 0,
                local: vec![],
                legal: std::array::from_fn(|i| i < 2),
                geometry_order: vec![0, 1],
                word,
                control: target,
                treatment: target,
                teacher: None,
                active: false,
            };
            let objective = loss(logits.clone(), &[row], false, Objective::QuantileInterval);
            let gradient = logits
                .grad(&objective.backward())
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(&gradient[..2], &expected);
            assert!(gradient[2..].iter().all(|x| *x == 0.));
        }
    }
}
