use blob_policy::observation::{HEADER_FEATURES, OBS_DIM, SLOT_FEATURES};
use burn::{nn, prelude::*};
#[derive(Module, Debug)]
pub struct Residual<B: Backend> {
    pub encoder: nn::Linear<B>,
    pub encoder2: nn::Linear<B>,
    pub fc: nn::Linear<B>,
    pub head: nn::Linear<B>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{Frozen, PARAMETERS};
    use burn::backend::NdArray;

    #[test]
    fn context_scalar_and_batched_models_agree_without_cross_row_state() {
        let device = Default::default();
        NdArray::<f32>::seed(&device, 171);
        let mut m = Residual::<NdArray>::new(&device);
        assert_eq!(m.num_params(), PARAMETERS);
        let mut a = vec![0.; OBS_DIM];
        for (i, x) in a.iter_mut().enumerate() {
            *x = (i % 19) as f32 / 19.;
        }
        for slot in a[70..].chunks_exact_mut(33) {
            slot[0] = 1.;
        }
        let c: Vec<f32> = (0..128).map(|i| (i % 13) as f32 / 13. - 0.5).collect();
        let zero = Frozen::new(m.values()).unwrap();
        assert!(zero.forward(&a, &c).unwrap().iter().all(|&x| x == 0.));
        m.head = nn::LinearConfig::new(32, 10).init(&device);
        let frozen = Frozen::new(m.values()).unwrap();
        let tensor = |v: Vec<f32>, n, width| {
            Tensor::<NdArray, 2>::from_data(TensorData::new(v, [n, width]), &device)
        };
        let single = m
            .forward(tensor(a.clone(), 1, OBS_DIM), tensor(c.clone(), 1, 128))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut b = a.clone();
        b[38..70].fill(0.);
        let slots: Vec<_> = b[70..].chunks_exact(33).rev().flatten().copied().collect();
        b[70..].copy_from_slice(&slots);
        let batch = m
            .forward(
                tensor([b.clone(), a.clone()].concat(), 2, OBS_DIM),
                tensor([vec![0.; 128], c.clone()].concat(), 2, 128),
            )
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(&batch[10..], single);
        assert_ne!(&batch[..10], single);
        assert_eq!(
            frozen.forward(&a, &c).unwrap(),
            frozen.forward(&b, &c).unwrap()
        );
        for (x, y) in single.iter().zip(frozen.forward(&a, &c).unwrap()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn pooled_encoder_gradients_match_scalar_finite_differences() {
        use burn::backend::Autodiff;
        for count in [2, 128] {
            let device = Default::default();
            NdArray::<f32>::seed(&device, 172);
            let mut m = Residual::<Autodiff<NdArray>>::new(&device);
            m.head = nn::LinearConfig::new(32, 10).init(&device);
            let observations: Vec<Vec<f32>> = (0..count)
                .map(|row| {
                    let mut obs = vec![0.; OBS_DIM];
                    for (i, x) in obs[..38].iter_mut().enumerate() {
                        *x = ((i + row * 7) % 17) as f32 / 17.;
                    }
                    for (i, slot) in obs[70..].chunks_exact_mut(33).enumerate() {
                        for (j, x) in slot.iter_mut().enumerate() {
                            *x = ((i * 7 + j * 3 + row * 5) % 31) as f32 / 31. - 0.4;
                        }
                        slot[0] = 1.;
                        slot[2] = ((i % 5) as f32 - 2.) / 127.;
                        slot[3] = ((i / 5) as f32 - 2.) / 127.;
                    }
                    obs
                })
                .collect();
            let contexts: Vec<Vec<f32>> = (0..count)
                .map(|r| {
                    (0..128)
                        .map(|i| ((i * 3 + r * 5) % 23) as f32 / 23. - 0.5)
                        .collect()
                })
                .collect();
            let obs = Tensor::from_data(
                TensorData::new(observations.concat(), [count, OBS_DIM]),
                &device,
            );
            let ctx = Tensor::from_data(TensorData::new(contexts.concat(), [count, 128]), &device);
            let labels = Tensor::<Autodiff<NdArray>, 2, Int>::from_data(
                TensorData::new(
                    (0..count).map(|i| (i % 10) as i32).collect::<Vec<_>>(),
                    [count, 1],
                ),
                &device,
            );
            let loss = burn::tensor::activation::log_softmax(m.forward(obs, ctx), 1)
                .gather(1, labels)
                .mean()
                .neg();
            let gradients = loss.backward();
            let values = m.values();
            let scalar_loss = |weights: Vec<f32>| {
                let frozen = Frozen::new(weights).unwrap();
                observations
                    .iter()
                    .zip(&contexts)
                    .enumerate()
                    .map(|(i, (obs, ctx))| {
                        let raw = frozen.forward(obs, ctx).unwrap();
                        let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                        raw.iter()
                            .map(|&x| (x as f64 - max).exp())
                            .sum::<f64>()
                            .ln()
                            + max
                            - raw[i % 10] as f64
                    })
                    .sum::<f64>()
                    / count as f64
            };
            let mut offset = 0;
            for (name, layer) in [
                ("encoder", &m.encoder),
                ("encoder2", &m.encoder2),
                ("fc", &m.fc),
                ("head", &m.head),
            ] {
                let mut grad = layer
                    .weight
                    .val()
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                grad.extend(
                    layer
                        .bias
                        .as_ref()
                        .unwrap()
                        .val()
                        .grad(&gradients)
                        .unwrap()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                );
                let norm = grad.iter().map(|&g| (g as f64).powi(2)).sum::<f64>().sqrt();
                assert!(norm > 1e-6, "{name}: no active gradient");
                let epsilon = 0.001;
                let mut plus = values.clone();
                let mut minus = values.clone();
                for (i, &g) in grad.iter().enumerate() {
                    let delta = (epsilon * g as f64 / norm) as f32;
                    plus[offset + i] += delta;
                    minus[offset + i] -= delta;
                }
                let numerical = (scalar_loss(plus) - scalar_loss(minus)) / (2. * epsilon);
                println!(
                    "batch={count} layer={name} autodiff={norm} finite_difference={numerical}"
                );
                assert!(
                    (numerical - norm).abs() < 0.0002 + norm * 0.03,
                    "{name}: autodiff {norm}, finite difference {numerical}"
                );
                offset += grad.len();
            }
            assert_eq!(offset, PARAMETERS);
        }
    }

    #[test]
    fn rejects_malformed_weights_and_private_context() {
        assert!(Frozen::new(vec![0.; PARAMETERS - 1]).is_err());
        let mut bad = vec![0.; PARAMETERS];
        bad[0] = f32::NAN;
        assert!(Frozen::new(bad).is_err());
        let f = Frozen::new(vec![0.; PARAMETERS]).unwrap();
        assert!(f.forward(&vec![0.; OBS_DIM], &[0.; 127]).is_err());
        assert!(f
            .forward(&vec![0.; OBS_DIM], &[f32::INFINITY; 128])
            .is_err());
    }
}
impl<B: Backend> Residual<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: nn::LinearConfig::new(33, 16).init(device),
            encoder2: nn::LinearConfig::new(16, 16).init(device),
            fc: nn::LinearConfig::new(198, 32).init(device),
            head: nn::LinearConfig::new(32, 10)
                .with_initializer(nn::Initializer::Zeros)
                .init(device),
        }
    }
    pub fn forward(&self, observation: Tensor<B, 2>, context: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = observation.dims()[0];
        let device = observation.device();
        let local = observation
            .clone()
            .slice([0..n, HEADER_FEATURES..OBS_DIM])
            .reshape([n, 32, SLOT_FEATURES]);
        let mask = local.clone().slice([0..n, 0..32, 0..1]);
        let scale: Vec<f32> = (0..33)
            .map(|i| if i == 2 || i == 3 { 127. } else { 1. })
            .collect();
        let local = local * Tensor::from_data(TensorData::new(scale, [1, 1, 33]), &device);
        let encoded = self
            .encoder2
            .forward(self.encoder.forward(local).tanh())
            .tanh()
            * mask;
        let mean = encoded.clone().sort(1).sum_dim(1).reshape([n, 16]) / 32.;
        let max = encoded.max_dim(1).reshape([n, 16]);
        let header = observation.slice([0..n, 0..38]);
        self.head.forward(burn::tensor::activation::relu(
            self.fc
                .forward(Tensor::cat(vec![header, mean, max, context], 1)),
        ))
    }
    pub fn values(&self) -> Vec<f32> {
        [&self.encoder, &self.encoder2, &self.fc, &self.head]
            .iter()
            .flat_map(|layer| {
                let mut v = layer.weight.val().into_data().to_vec::<f32>().unwrap();
                v.extend(
                    layer
                        .bias
                        .as_ref()
                        .unwrap()
                        .val()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                );
                v
            })
            .collect()
    }
}
