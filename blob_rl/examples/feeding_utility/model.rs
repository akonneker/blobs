use super::data::Row;
use blob_rl::observation::SLOT_FEATURES;
use burn::{nn, prelude::*};
const SLOTS: usize = 32;
#[derive(Module, Debug)]
pub struct Utility<B: Backend> {
    encoder: nn::Linear<B>,
    encoder2: nn::Linear<B>,
    fc: nn::Linear<B>,
    head: nn::Linear<B>,
}
impl<B: Backend> Utility<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: nn::LinearConfig::new(SLOT_FEATURES, 16).init(device),
            encoder2: nn::LinearConfig::new(16, 16).init(device),
            fc: nn::LinearConfig::new(SLOT_FEATURES + 32, 64).init(device),
            head: nn::LinearConfig::new(64, 1)
                .with_initializer(nn::Initializer::Zeros)
                .init(device),
        }
    }
    #[cfg(test)]
    pub fn activate_for_test(&mut self, device: &B::Device) {
        self.head = nn::LinearConfig::new(64, 1).init(device);
    }
    pub fn forward(&self, local: Tensor<B, 3>, legal: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = local.dims()[0];
        let encoded = self
            .encoder2
            .forward(self.encoder.forward(local.clone()).tanh())
            .tanh();
        let summary = (encoded.clone() * legal.clone().unsqueeze_dim(2))
            .sort(1)
            .sum_dim(1)
            / SLOTS as f32;
        self.head
            .forward(burn::tensor::activation::relu(self.fc.forward(
                Tensor::cat(vec![local, encoded, summary.repeat_dim(1, SLOTS)], 2),
            )))
            .reshape([batch, SLOTS])
            + (legal - 1.) * 1e9
    }
}
pub fn batch<B: Backend>(rows: &[Row], device: &B::Device) -> (Tensor<B, 3>, Tensor<B, 2>) {
    (
        Tensor::from_data(
            TensorData::new(
                rows.iter()
                    .flat_map(|r| r.local.iter().copied())
                    .collect::<Vec<_>>(),
                [rows.len(), SLOTS, SLOT_FEATURES],
            ),
            device,
        ),
        Tensor::from_data(
            TensorData::new(
                rows.iter()
                    .flat_map(|r| r.legal.map(f32::from))
                    .collect::<Vec<_>>(),
                [rows.len(), SLOTS],
            ),
            device,
        ),
    )
}

impl Utility<burn::backend::NdArray> {
    /// Exact numeric fingerprint, excluding optimizer-specific parameter IDs.
    pub fn parameter_bits(&self) -> Vec<u32> {
        [&self.encoder, &self.encoder2, &self.fc, &self.head]
            .into_iter()
            .flat_map(|layer| {
                let mut values = layer.weight.val().into_data().to_vec::<f32>().unwrap();
                if let Some(bias) = &layer.bias {
                    values.extend(bias.val().into_data().to_vec::<f32>().unwrap());
                }
                values.into_iter().map(f32::to_bits)
            })
            .collect()
    }
}
