//! Independent scalar diagnostic model. Not a deployable weight format.
use blob_policy::observation::*;
pub const PARAMETERS: usize = 7514;
struct Linear {
    input: usize,
    output: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl Linear {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.input);
        (0..self.output)
            .map(|j| {
                let mut y = 0.;
                for (i, &v) in x.iter().enumerate() {
                    y += v * self.weight[i * self.output + j];
                }
                y + self.bias[j]
            })
            .collect()
    }
}
pub struct Frozen {
    layers: Vec<Linear>,
}
impl Frozen {
    pub fn new(values: Vec<f32>) -> Result<Self, String> {
        if values.len() != PARAMETERS || !values.iter().all(|v| v.is_finite()) {
            return Err("invalid diagnostic weights".into());
        }
        let mut it = values.into_iter();
        let layers = [(33, 16), (16, 16), (198, 32), (32, 10)]
            .into_iter()
            .map(|(input, output)| Linear {
                input,
                output,
                weight: it.by_ref().take(input * output).collect(),
                bias: it.by_ref().take(output).collect(),
            })
            .collect();
        Ok(Self { layers })
    }
    pub fn forward(&self, observation: &[f32], context: &[f32]) -> Result<Vec<f32>, String> {
        self.forward_with_stats(observation, context)
            .map(|(output, _)| output)
    }
    /// Counts: present encoded units, first/second tanh saturation, active/total ReLUs,
    /// then maximum first-layer absolute preactivation.
    pub fn forward_with_stats(
        &self,
        observation: &[f32],
        context: &[f32],
    ) -> Result<(Vec<f32>, [f64; 6]), String> {
        let mut stats = [0.; 6];
        if observation.len() != OBS_DIM || !observation.iter().all(|x| x.is_finite()) {
            return Err("invalid relational observation".into());
        }
        let encoded: Vec<Vec<f32>> = observation[HEADER_FEATURES..]
            .chunks_exact(SLOT_FEATURES)
            .map(|slot| {
                let mut local = slot.to_vec();
                local[2] *= 127.;
                local[3] *= 127.;
                let raw = self.layers[0].forward(&local);
                let first: Vec<_> = raw.iter().copied().map(f32::tanh).collect();
                let second: Vec<_> = self.layers[1]
                    .forward(&first)
                    .into_iter()
                    .map(f32::tanh)
                    .collect();
                if slot[0] > 0. {
                    stats[0] += 16.;
                    stats[1] += first.iter().filter(|x| x.abs() >= 0.99).count() as f64;
                    stats[2] += second.iter().filter(|x| x.abs() >= 0.99).count() as f64;
                    stats[5] = raw
                        .iter()
                        .map(|x| f64::from(x.abs()))
                        .fold(stats[5], f64::max);
                }
                second.into_iter().map(|x| x * slot[0]).collect()
            })
            .collect();
        let mut features = observation[..OBS_RANDOMNESS_FEATURE_START].to_vec();
        for i in 0..16 {
            let mut values: Vec<_> = encoded.iter().map(|s| s[i]).collect();
            values.sort_by(f32::total_cmp);
            features.push(values.into_iter().sum::<f32>() / 32.);
        }
        features.extend((0..16).map(|i| {
            encoded
                .iter()
                .map(|s| s[i])
                .fold(f32::NEG_INFINITY, f32::max)
        }));
        if context.len() != 128 || !context.iter().all(|x| x.is_finite()) {
            return Err("invalid context".into());
        }
        features.extend_from_slice(context);
        let hidden: Vec<_> = self.layers[2]
            .forward(&features)
            .into_iter()
            .map(|x| x.max(0.))
            .collect();
        stats[3] = hidden.iter().filter(|&&x| x > 0.).count() as f64;
        stats[4] = hidden.len() as f64;
        let result = self.layers[3].forward(&hidden);
        if !result.iter().all(|x| x.is_finite()) {
            return Err("nonfinite relational inference".into());
        }
        Ok((result, stats))
    }
}
