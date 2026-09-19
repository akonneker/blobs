//! Query-specific learned comparisons. No rank, identity, or teacher features.
use burn::{nn, prelude::*};

#[derive(Module, Debug)]
pub(super) struct RelationalEncoder<B: Backend> {
    first: nn::Linear<B>,
    second: nn::Linear<B>,
}

impl<B: Backend> RelationalEncoder<B> {
    pub(super) fn new(device: &B::Device) -> Self {
        Self {
            first: nn::LinearConfig::new(3, 16).init(device),
            second: nn::LinearConfig::new(16, 16).init(device),
        }
    }

    pub(super) fn forward(&self, local: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, slots, _] = local.dims();
        // Other minus query: relative dx, dy, and observed food. The query's
        // own features enter the shared scorer separately. Self-comparison is
        // included, so every legal candidate contributes to the count signal.
        let differences = local.clone().unsqueeze_dim::<4>(1) - local.unsqueeze_dim::<4>(2);
        let encoded = self
            .second
            .forward(self.first.forward(differences).tanh())
            .tanh();
        let present = mask.equal_elem(0.).float().reshape([batch, 1, slots, 1]);
        // Sorting fixes addition order under competitor permutation. Dividing
        // by capacity, rather than the legal count, preserves cardinality.
        (encoded * present)
            .sort(2)
            .sum_dim(2)
            .reshape([batch, slots, 16])
            / slots as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn four_dimensional_linear_gradients_match_flat_pair_rows() {
        let device = Default::default();
        let layer = nn::LinearConfig::new(3, 16).init::<Autodiff<NdArray>>(&device);
        let input = Tensor::<Autodiff<NdArray>, 4>::from_data(
            TensorData::new(
                (0..2 * 4 * 4 * 3)
                    .map(|i| (i % 13) as f32 / 13.)
                    .collect::<Vec<_>>(),
                [2, 4, 4, 3],
            ),
            &device,
        );
        let a = layer.forward(input.clone()).powf_scalar(2.).mean();
        let b = layer.forward(input.reshape([32, 3])).powf_scalar(2.).mean();
        let ga = layer
            .weight
            .val()
            .grad(&a.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let gb = layer
            .weight
            .val()
            .grad(&b.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(ga.iter().zip(gb).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn comparisons_are_translation_invariant_and_masked_competitors_have_no_gradient() {
        let device = Default::default();
        let encoder = RelationalEncoder::<Autodiff<NdArray>>::new(&device);
        let local = Tensor::from_data(
            TensorData::new(
                vec![0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 1., 0.],
                [1, 4, 3],
            ),
            &device,
        )
        .require_grad();
        let mask = Tensor::from_data(TensorData::new(vec![0., 0., 0., -1e9], [1, 4]), &device);
        let summary = encoder.forward(local.clone(), mask.clone());
        let shifted = encoder.forward(local.clone() + 2., mask);
        assert_eq!(
            summary.clone().into_data().to_vec::<f32>().unwrap(),
            shifted.into_data().to_vec::<f32>().unwrap()
        );
        // Exclude the illegal query itself from this loss. Its location cannot
        // affect any legal query's comparison or summary.
        let loss = summary.slice([0..1, 0..3, 0..16]).sum();
        let gradients = local
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(gradients[9..12].iter().all(|g| *g == 0.));
        assert!(gradients[..9].iter().any(|g| g.abs() > 1e-6));
    }
}
