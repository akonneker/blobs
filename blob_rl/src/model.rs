//! Actor-critic neural network model using Burn.

use burn::nn;
use burn::prelude::*;

use crate::action::NUM_ACTIONS;
use crate::observation::OBS_DIM;

/// Actor-critic model with shared backbone.
///
/// Architecture: OBS_DIM → 128 (ReLU) → 64 (ReLU) → policy head + value head
#[derive(Module, Debug)]
pub struct PolicyValueNet<B: Backend> {
    shared_fc1: nn::Linear<B>,
    shared_fc2: nn::Linear<B>,
    policy_head: nn::Linear<B>,
    value_head: nn::Linear<B>,
}

/// Configuration for creating a PolicyValueNet.
#[derive(Config, Debug)]
pub struct PolicyValueNetConfig {
    #[config(default = 128)]
    pub hidden1: usize,
    #[config(default = 64)]
    pub hidden2: usize,
}

impl PolicyValueNetConfig {
    /// Initialize a new PolicyValueNet on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PolicyValueNet<B> {
        PolicyValueNet {
            shared_fc1: nn::LinearConfig::new(OBS_DIM, self.hidden1).init(device),
            shared_fc2: nn::LinearConfig::new(self.hidden1, self.hidden2).init(device),
            policy_head: nn::LinearConfig::new(self.hidden2, NUM_ACTIONS).init(device),
            value_head: nn::LinearConfig::new(self.hidden2, 1).init(device),
        }
    }
}

/// Output of a forward pass through the model.
pub struct ModelOutput<B: Backend> {
    /// Action logits [batch, NUM_ACTIONS]
    pub policy_logits: Tensor<B, 2>,
    /// State value estimates [batch, 1]
    pub values: Tensor<B, 2>,
}

impl<B: Backend> PolicyValueNet<B> {
    /// Forward pass: observations → (policy logits, value estimates)
    pub fn forward(&self, obs: Tensor<B, 2>) -> ModelOutput<B> {
        let x = self.shared_fc1.forward(obs);
        let x = burn::tensor::activation::relu(x);
        let x = self.shared_fc2.forward(x);
        let x = burn::tensor::activation::relu(x);

        let policy_logits = self.policy_head.forward(x.clone());
        let values = self.value_head.forward(x);

        ModelOutput {
            policy_logits,
            values,
        }
    }

    /// Get action probabilities via softmax.
    pub fn action_probs(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.forward(obs);
        burn::tensor::activation::softmax(output.policy_logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray as NdArrayBackend;

    type TestBackend = NdArrayBackend;

    #[test]
    fn test_model_forward_shapes() {
        let device = Default::default();
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig::new().init(&device);

        let batch_size = 8;
        let obs = Tensor::<TestBackend, 2>::zeros([batch_size, OBS_DIM], &device);
        let output = model.forward(obs);

        assert_eq!(output.policy_logits.dims(), [batch_size, NUM_ACTIONS]);
        assert_eq!(output.values.dims(), [batch_size, 1]);
    }

    #[test]
    fn test_action_probs_sum_to_one() {
        let device = Default::default();
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig::new().init(&device);

        let obs = Tensor::<TestBackend, 2>::zeros([4, OBS_DIM], &device);
        let probs = model.action_probs(obs);

        // Sum across actions dimension should be ~1.0 for each batch item
        let sums = probs.sum_dim(1);
        let sums_data: Vec<f32> = sums.into_data().to_vec().unwrap();
        for sum in sums_data {
            assert!((sum - 1.0).abs() < 1e-5, "Sum = {}, expected ~1.0", sum);
        }
    }
}
