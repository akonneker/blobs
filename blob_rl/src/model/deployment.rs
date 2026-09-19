//! Explicit export of the current policy's inference weights.
use super::*;
use blob_policy::runtime::{FrozenPolicy, Linear};

impl<B: Backend> PolicyValueNet<B> {
    pub fn export_frozen(&self) -> Result<FrozenPolicy, String> {
        let convert = |layer: &nn::Linear<B>| -> Result<Linear, String> {
            let [inputs, outputs] = layer.weight.val().dims();
            let weights = layer
                .weight
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|e| format!("export weights: {e}"))?;
            let bias = layer
                .bias
                .as_ref()
                .ok_or("export requires linear bias")?
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|e| format!("export bias: {e}"))?;
            Linear::new(inputs, outputs, weights, bias)
        };
        let layers = [
            &self.slot_encoder,
            &self.shared_fc1,
            &self.recurrent,
            &self.shared_fc2,
            &self.foraging_action_kind_head,
            &self.foraging_adapter_fc,
            &self.foraging_adapter_head,
            &self.interaction_action_kind_head,
            &self.interaction_adapter_fc,
            &self.interaction_adapter_head,
            &self.interaction_slot_adapter_fc,
            &self.interaction_slot_adapter_head,
            &self.exploration_action_kind_head,
            &self.exploration_adapter_fc,
            &self.exploration_adapter_head,
            &self.exploration_slot_adapter_fc,
            &self.exploration_slot_adapter_head,
            &self.exploration_guard_readiness_head,
            &self.target_query_head,
            &self.effort_head,
            &self.amount_head,
            &self.signal_head,
            &self.signal_strength_head,
            &self.target_residual.fc,
            &self.target_residual.head,
        ]
        .into_iter()
        .map(convert)
        .collect::<Result<Vec<_>, _>>()?;
        FrozenPolicy::new(
            self.shared_fc1.weight.val().dims()[1],
            self.hidden2(),
            self.recurrent_size(),
            layers,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    #[test]
    fn exported_nonzero_network_matches_training_outputs_in_all_contexts() {
        let _guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        NdArray::<f32>::seed(&device, 721);
        let mut model = PolicyValueNetConfig::new()
            .with_hidden1(16)
            .with_hidden2(8)
            .with_recurrent_size(4)
            .init::<NdArray<f32>>(&device);
        // Activate every inference adapter so export cannot silently omit one.
        for layer in [
            &mut model.foraging_action_kind_head,
            &mut model.foraging_adapter_fc,
            &mut model.foraging_adapter_head,
            &mut model.interaction_action_kind_head,
            &mut model.interaction_adapter_fc,
            &mut model.interaction_adapter_head,
            &mut model.interaction_slot_adapter_fc,
            &mut model.interaction_slot_adapter_head,
            &mut model.exploration_action_kind_head,
            &mut model.exploration_adapter_fc,
            &mut model.exploration_adapter_head,
            &mut model.exploration_slot_adapter_fc,
            &mut model.exploration_slot_adapter_head,
            &mut model.exploration_guard_readiness_head,
            &mut model.target_query_head,
            &mut model.effort_head,
            &mut model.amount_head,
            &mut model.signal_head,
            &mut model.signal_strength_head,
            &mut model.target_residual.fc,
            &mut model.target_residual.head,
        ] {
            let [i, o] = layer.weight.val().dims();
            *layer = nn::LinearConfig::new(i, o).init(&device);
        }
        let frozen = model.export_frozen().unwrap();
        for sample in 0..48 {
            let mut obs = (0..OBS_DIM)
                .map(|i| ((i * 17 + sample * 11) % 101) as f32 / 101.0)
                .collect::<Vec<_>>();
            for slot in obs[HEADER_FEATURES..].chunks_exact_mut(SLOT_FEATURES) {
                slot[SLOT_NEIGHBOR_PRESENT_FEATURE] = if sample % 3 == 1 { 1.0 } else { 0.0 };
                slot[SLOT_NEIGHBOR_ACTIVITY_FEATURE] =
                    if sample % 6 == 1 { 3.0 / 8.0 } else { 0.0 };
            }
            obs[CURRENT_TILE_PLANT_CAPACITY_FEATURE] = if sample % 3 == 0 { 0.5 } else { 0.0 };
            obs[CURRENT_TILE_LOOSE_ENERGY_FEATURE] = 0.0;
            let memory = vec![0.1, -0.2, 0.3, -0.4];
            let expected = model.forward_with_memory(
                Tensor::from_data(TensorData::new(obs.clone(), [1, OBS_DIM]), &device),
                Tensor::from_data(TensorData::new(memory.clone(), [1, 4]), &device),
            );
            let actual = frozen.forward(&obs, &memory).unwrap();
            for (label, expected, actual) in [
                ("kind", expected.action_kind_logits, actual.kind),
                ("target", expected.target_logits, actual.target),
                ("effort", expected.effort_logits, actual.effort),
                ("amount", expected.amount_logits, actual.amount),
                ("signal", expected.signal_logits, actual.signal),
                (
                    "strength",
                    expected.signal_strength_logits,
                    actual.signal_strength,
                ),
                ("memory", expected.next_memory, actual.next_memory),
            ] {
                for (a, b) in expected
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .zip(actual)
                {
                    assert!(
                        (a - b).abs() <= 2e-5 * (1.0 + a.abs()),
                        "{label} sample {sample}: {a} != {b}"
                    );
                }
            }
        }
    }
}
