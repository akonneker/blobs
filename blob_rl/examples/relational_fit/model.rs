use blob_policy::observation::{HEADER_FEATURES, OBS_DIM, SLOT_FEATURES};
use burn::{nn, prelude::*};
#[derive(Module, Debug)]
pub struct Residual<B: Backend> {
    pub encoder: nn::Linear<B>,
    pub encoder2: nn::Linear<B>,
    pub fc: nn::Linear<B>,
    pub head: nn::Linear<B>,
}
impl<B: Backend> Residual<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: nn::LinearConfig::new(33, 16).init(device),
            encoder2: nn::LinearConfig::new(16, 16).init(device),
            fc: nn::LinearConfig::new(70, 32).init(device),
            head: nn::LinearConfig::new(32, 10)
                .with_initializer(nn::Initializer::Zeros)
                .init(device),
        }
    }
    pub fn forward(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
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
            self.fc.forward(Tensor::cat(vec![header, mean, max], 1)),
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
#[cfg(test)]
mod tests {
    use super::*;
    use blob_policy::relational::FrozenResidual;
    use burn::backend::NdArray;
    #[test]
    fn active_encoder_matches_portable_and_is_row_and_slot_independent() {
        let device = Default::default();
        NdArray::<f32>::seed(&device, 171);
        let mut model = Residual::<NdArray>::new(&device);
        model.head = nn::LinearConfig::new(32, 10).init(&device);
        let mut a = vec![0.; OBS_DIM];
        for (i, x) in a.iter_mut().enumerate() {
            *x = (i % 19) as f32 / 19.;
        }
        for s in a[70..].chunks_exact_mut(33) {
            s[0] = 1.;
        }
        let mut b = a.clone();
        b[38..70].fill(0.);
        let slots: Vec<_> = b[70..].chunks_exact(33).rev().flatten().copied().collect();
        b[70..].copy_from_slice(&slots);
        let frozen = FrozenResidual::new(model.values()).unwrap();
        assert_eq!(frozen.forward(&a).unwrap(), frozen.forward(&b).unwrap());
        let tensor = |v: Vec<f32>, n| {
            Tensor::<NdArray, 2>::from_data(TensorData::new(v, [n, OBS_DIM]), &device)
        };
        let single = model
            .forward(tensor(a.clone(), 1))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let batch = model
            .forward(tensor([b.clone(), a.clone()].concat(), 2))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(&batch[10..], single);
        assert_eq!(&batch[..10], single);
        for (x, y) in single.iter().zip(frozen.forward(&a).unwrap()) {
            assert!((x - y).abs() < 1e-5);
        }
    }
}
