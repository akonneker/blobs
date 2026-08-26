//! Actor-critic neural network model using Burn.

use burn::nn;
use burn::prelude::*;

use crate::action::{
    NUM_ACTIONS, NUM_AMOUNT_CHOICES, NUM_SIGNAL_CHOICES, NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::observation::OBS_DIM;

const POLICY_MEMORY_MAGIC: [u8; 4] = *b"BRM1";
const POLICY_MEMORY_HEADER_BYTES: usize = 8;

/// Actor-critic model with a cell-private recurrent state.
///
/// The recurrent state is row-separable: no operation aggregates across the
/// batch dimension. Deployment stores the state in canonical Mind private
/// memory, so it follows the same isolation and replay rules as handwritten
/// stateful Minds.
#[derive(Module, Debug)]
pub struct PolicyValueNet<B: Backend> {
    shared_fc1: nn::Linear<B>,
    recurrent: nn::Linear<B>,
    shared_fc2: nn::Linear<B>,
    policy_head: nn::Linear<B>,
    amount_head: nn::Linear<B>,
    signal_head: nn::Linear<B>,
    signal_strength_head: nn::Linear<B>,
    value_head: nn::Linear<B>,
}

/// Configuration for creating a PolicyValueNet.
#[derive(Config, Debug)]
pub struct PolicyValueNetConfig {
    #[config(default = 128)]
    pub hidden1: usize,
    #[config(default = 64)]
    pub hidden2: usize,
    #[config(default = 64)]
    pub recurrent_size: usize,
}

impl PolicyValueNetConfig {
    /// Initialize a new PolicyValueNet on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PolicyValueNet<B> {
        PolicyValueNet {
            shared_fc1: nn::LinearConfig::new(OBS_DIM, self.hidden1).init(device),
            recurrent: nn::LinearConfig::new(
                self.hidden1 + self.recurrent_size,
                self.recurrent_size,
            )
            .init(device),
            shared_fc2: nn::LinearConfig::new(self.recurrent_size, self.hidden2).init(device),
            policy_head: nn::LinearConfig::new(self.hidden2, NUM_ACTIONS).init(device),
            amount_head: nn::LinearConfig::new(self.hidden2, NUM_AMOUNT_CHOICES).init(device),
            signal_head: nn::LinearConfig::new(self.hidden2, NUM_SIGNAL_CHOICES).init(device),
            signal_strength_head: nn::LinearConfig::new(self.hidden2, NUM_SIGNAL_STRENGTH_CHOICES)
                .init(device),
            value_head: nn::LinearConfig::new(self.hidden2, 1).init(device),
        }
    }
}

/// Output of a forward pass through the model.
pub struct ModelOutput<B: Backend> {
    /// Action logits [batch, NUM_ACTIONS]
    pub policy_logits: Tensor<B, 2>,
    /// Conditional payload/amount logits [batch, NUM_AMOUNT_CHOICES].
    pub amount_logits: Tensor<B, 2>,
    /// Optional signal-selection logits [batch, NUM_SIGNAL_CHOICES].
    pub signal_logits: Tensor<B, 2>,
    /// Conditional signal-strength logits [batch, NUM_SIGNAL_STRENGTH_CHOICES].
    pub signal_strength_logits: Tensor<B, 2>,
    /// State value estimates [batch, 1]
    pub values: Tensor<B, 2>,
    /// Updated cell-private recurrent state [batch, recurrent_size].
    pub next_memory: Tensor<B, 2>,
}

impl<B: Backend> PolicyValueNet<B> {
    pub fn recurrent_size(&self) -> usize {
        self.recurrent.weight.val().dims()[1]
    }

    /// Stateless convenience pass using zero private memory.
    pub fn forward(&self, obs: Tensor<B, 2>) -> ModelOutput<B> {
        let [batch, _] = obs.dims();
        let recurrent_size = self.recurrent_size();
        let device = obs.device();
        let memory = Tensor::zeros([batch, recurrent_size], &device);
        self.forward_with_memory(obs, memory)
    }

    /// One recurrent decision step. Every output row depends only on the
    /// corresponding observation and private-memory row.
    pub fn forward_with_memory(&self, obs: Tensor<B, 2>, memory: Tensor<B, 2>) -> ModelOutput<B> {
        let x = self.shared_fc1.forward(obs);
        let x = burn::tensor::activation::relu(x);
        let next_memory =
            burn::tensor::activation::tanh(self.recurrent.forward(Tensor::cat(vec![x, memory], 1)));
        let x = self.shared_fc2.forward(next_memory.clone());
        let x = burn::tensor::activation::relu(x);

        let policy_logits = self.policy_head.forward(x.clone());
        let amount_logits = self.amount_head.forward(x.clone());
        let signal_logits = self.signal_head.forward(x.clone());
        let signal_strength_logits = self.signal_strength_head.forward(x.clone());
        let values = self.value_head.forward(x);

        ModelOutput {
            policy_logits,
            amount_logits,
            signal_logits,
            signal_strength_logits,
            values,
            next_memory,
        }
    }

    /// Get action probabilities via softmax.
    pub fn action_probs(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.forward(obs);
        burn::tensor::activation::softmax(output.policy_logits, 1)
    }
}

pub fn policy_memory_bytes(recurrent_size: usize) -> Option<usize> {
    recurrent_size
        .checked_mul(std::mem::size_of::<i16>())
        .and_then(|bytes| bytes.checked_add(POLICY_MEMORY_HEADER_BYTES))
}

/// Decode only this policy's exact versioned memory format. Empty or malformed
/// memory deterministically initializes a fresh zero state.
pub fn decode_policy_memory(bytes: &[u8], recurrent_size: usize) -> Vec<f32> {
    let Some(expected) = policy_memory_bytes(recurrent_size) else {
        return vec![0.0; recurrent_size];
    };
    if bytes.len() != expected || bytes[..4] != POLICY_MEMORY_MAGIC {
        return vec![0.0; recurrent_size];
    }
    let declared = u32::from_le_bytes(bytes[4..8].try_into().expect("fixed memory header"));
    if usize::try_from(declared).ok() != Some(recurrent_size) {
        return vec![0.0; recurrent_size];
    }
    bytes[POLICY_MEMORY_HEADER_BYTES..]
        .chunks_exact(2)
        .map(|chunk| {
            f32::from(i16::from_le_bytes(
                chunk.try_into().expect("two-byte memory element"),
            )) / f32::from(i16::MAX)
        })
        .collect()
}

pub fn encode_policy_memory(memory: &[f32]) -> Vec<u8> {
    let declared = u32::try_from(memory.len()).expect("validated recurrent state fits u32");
    let mut bytes = Vec::with_capacity(
        policy_memory_bytes(memory.len()).expect("validated recurrent state byte size"),
    );
    bytes.extend_from_slice(&POLICY_MEMORY_MAGIC);
    bytes.extend_from_slice(&declared.to_le_bytes());
    for value in memory {
        let value = if value.is_finite() { *value } else { 0.0 };
        let quantized = (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray as NdArrayBackend;

    type TestBackend = NdArrayBackend;

    #[test]
    fn test_model_forward_shapes() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig::new().init(&device);

        let batch_size = 8;
        let obs = Tensor::<TestBackend, 2>::zeros([batch_size, OBS_DIM], &device);
        let output = model.forward(obs);

        assert_eq!(output.policy_logits.dims(), [batch_size, NUM_ACTIONS]);
        assert_eq!(
            output.amount_logits.dims(),
            [batch_size, NUM_AMOUNT_CHOICES]
        );
        assert_eq!(
            output.signal_logits.dims(),
            [batch_size, NUM_SIGNAL_CHOICES]
        );
        assert_eq!(
            output.signal_strength_logits.dims(),
            [batch_size, NUM_SIGNAL_STRENGTH_CHOICES]
        );
        assert_eq!(output.values.dims(), [batch_size, 1]);
        assert_eq!(output.next_memory.dims(), [batch_size, 64]);
    }

    #[test]
    fn test_action_probs_sum_to_one() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
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

    #[test]
    fn policy_memory_is_versioned_bounded_and_fail_closed() {
        let state = vec![0.25, -0.5, 1.0];
        let encoded = encode_policy_memory(&state);
        assert_eq!(encoded.len(), 8 + state.len() * 2);
        let decoded = decode_policy_memory(&encoded, state.len());
        for (expected, actual) in state.iter().zip(decoded) {
            assert!((expected - actual).abs() <= 1.0 / f32::from(i16::MAX));
        }
        assert_eq!(
            decode_policy_memory(&encoded, state.len() + 1),
            vec![0.0; 4]
        );
        let mut malformed = encoded;
        malformed[0] ^= 1;
        assert_eq!(decode_policy_memory(&malformed, state.len()), vec![0.0; 3]);
    }

    #[test]
    fn recurrent_rows_are_isolated_and_memory_changes_the_next_state() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        TestBackend::seed(&device, 123);
        let model: PolicyValueNet<TestBackend> = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 4,
        }
        .init(&device);
        let observation = vec![0.125; OBS_DIM];
        let output = model.forward_with_memory(
            Tensor::from_data(
                TensorData::new(
                    observation
                        .iter()
                        .chain(&observation)
                        .copied()
                        .collect::<Vec<_>>(),
                    [2, OBS_DIM],
                ),
                &device,
            ),
            Tensor::from_data(
                TensorData::new(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [2, 4]),
                &device,
            ),
        );
        let memory = output.next_memory.into_data().to_vec::<f32>().unwrap();
        assert_ne!(&memory[..4], &memory[4..]);

        let scalar = model
            .forward_with_memory(
                Tensor::from_data(TensorData::new(observation, [1, OBS_DIM]), &device),
                Tensor::zeros([1, 4], &device),
            )
            .next_memory
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (batched, scalar) in memory[..4].iter().zip(scalar) {
            assert!((batched - scalar).abs() < 1.0e-6);
        }
    }
}
