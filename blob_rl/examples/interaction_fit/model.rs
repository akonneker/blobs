//! The two existing deployed parameter blocks used by controlled interaction fits.
use blob_policy::{
    observation::{HEADER_FEATURES, OBS_RANDOMNESS_FEATURE_START, SLOT_FEATURES},
    runtime::layer_shapes,
};
use burn::{module::Param, nn, prelude::*};
use std::ops::Range;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Update {
    BaseHead,
    SlotAdapter,
}
impl Update {
    pub fn from_plan(value: &serde_json::Value) -> Self {
        match value.as_str() {
            None if value.is_null() => Self::BaseHead,
            Some("base_head") => Self::BaseHead,
            Some("slot_adapter") => Self::SlotAdapter,
            _ => panic!("unknown intervention"),
        }
    }
    pub fn range(self, bytes: &[u8]) -> Range<usize> {
        let dim = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        let shapes = layer_shapes(dim(8), dim(12), dim(16)).unwrap();
        let (first, last) = match self {
            Self::BaseHead => {
                assert_eq!(shapes[7], ("interaction_action_kind_head", 128, 10));
                (7, 8)
            }
            Self::SlotAdapter => {
                assert_eq!(shapes[10], ("interaction_slot_adapter_fc", 71, 16));
                assert_eq!(shapes[11], ("interaction_slot_adapter_head", 16, 10));
                (10, 12)
            }
        };
        let size = |slice: &[(&str, usize, usize)]| {
            slice.iter().map(|(_, i, o)| (i * o + o) * 4).sum::<usize>()
        };
        let start = 20 + size(&shapes[..first]);
        start..start + size(&shapes[first..last])
    }
}
pub fn local(observation: &[f32]) -> Vec<f32> {
    let mut features = observation[..OBS_RANDOMNESS_FEATURE_START].to_vec();
    features.extend((0..SLOT_FEATURES).map(|i| {
        observation[HEADER_FEATURES..]
            .chunks_exact(SLOT_FEATURES)
            .map(|s| s[i])
            .fold(f32::NEG_INFINITY, f32::max)
    }));
    features
}
#[derive(Module, Debug)]
pub struct Fit<B: Backend> {
    pub fc: Option<nn::Linear<B>>,
    pub head: nn::Linear<B>,
}
fn linear<B: Backend>(values: &[f32], i: usize, o: usize, device: &B::Device) -> nn::Linear<B> {
    assert_eq!(values.len(), i * o + o);
    nn::Linear {
        weight: Param::from_tensor(Tensor::from_data(
            TensorData::new(values[..i * o].to_vec(), [i, o]),
            device,
        )),
        bias: Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(values[i * o..].to_vec(), [o]),
            device,
        ))),
    }
}
impl<B: Backend> Fit<B> {
    pub fn new(update: Update, values: &[f32], device: &B::Device) -> Self {
        match update {
            Update::BaseHead => Self {
                fc: None,
                head: linear(values, 128, 10, device),
            },
            Update::SlotAdapter => Self {
                fc: Some(linear(&values[..1152], 71, 16, device)),
                head: linear(&values[1152..], 16, 10, device),
            },
        }
    }
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = if let Some(fc) = &self.fc {
            burn::tensor::activation::relu(fc.forward(x))
        } else {
            x
        };
        self.head.forward(x)
    }
    pub fn values(&self) -> Vec<f32> {
        self.fc
            .iter()
            .chain(std::iter::once(&self.head))
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
    #[test]
    fn local_features_ignore_private_draws_and_slot_order() {
        let mut obs = vec![0.; blob_policy::observation::OBS_DIM];
        for (i, x) in obs.iter_mut().enumerate() {
            *x = i as f32 / 100.;
        }
        let original = local(&obs);
        obs[38..70].fill(-123.);
        let slots: Vec<_> = obs[70..]
            .chunks_exact(33)
            .map(|s| s.to_vec())
            .rev()
            .flatten()
            .collect();
        obs[70..].copy_from_slice(&slots);
        assert_eq!(local(&obs), original);
    }
    #[test]
    fn adapter_roundtrips_and_zero_head_has_zero_effect() {
        let device = Default::default();
        let mut v = vec![0.125; 1152];
        v.extend(vec![0.; 170]);
        let fit = Fit::<burn::backend::NdArray>::new(Update::SlotAdapter, &v, &device);
        assert_eq!(fit.values(), v);
        let out = fit
            .forward(Tensor::ones([2, 71], &device))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(out.iter().all(|x| *x == 0.));
    }
}
