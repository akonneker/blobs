//! Frozen, training-derived context transform. Never changes parent memory.
pub struct MemoryNormalizer {
    means: Vec<f32>,
    scales: Vec<f32>,
}
impl MemoryNormalizer {
    // This shared module is also compiled into the historical hard-fixture probe.
    #[allow(dead_code)]
    pub fn apply_preserving_zero(&self, memory: &[f32]) -> Result<Vec<f32>, String> {
        if memory.len() != 128 || !memory.iter().all(|v| v.is_finite()) {
            return Err("invalid incoming memory".into());
        }
        if memory.iter().all(|&v| v == 0.) {
            Ok(vec![0.; 128])
        } else {
            self.apply(memory)
        }
    }
    pub fn new(means: Vec<f32>, scales: Vec<f32>) -> Result<Self, String> {
        if means.len() != 128
            || scales.len() != 128
            || !means.iter().all(|v| v.is_finite())
            || !scales.iter().all(|v| v.is_finite() && *v > 0.)
        {
            return Err("invalid memory normalizer".into());
        }
        Ok(Self { means, scales })
    }
    pub fn apply(&self, memory: &[f32]) -> Result<Vec<f32>, String> {
        if memory.len() != 128 || !memory.iter().all(|v| v.is_finite()) {
            return Err("invalid incoming memory".into());
        }
        let output: Vec<f32> = memory
            .iter()
            .zip(&self.means)
            .zip(&self.scales)
            .map(|((&x, &mean), &scale)| (x - mean) / scale)
            .collect();
        if !output.iter().all(|v| v.is_finite()) {
            return Err("nonfinite standardized memory".into());
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_only_the_whole_zero_state_and_validates_before_bypass() {
        let n = MemoryNormalizer::new(vec![0.25; 128], vec![0.125; 128]).unwrap();
        assert_eq!(n.apply_preserving_zero(&[0.; 128]).unwrap(), vec![0.; 128]);
        assert_eq!(n.apply_preserving_zero(&[-0.; 128]).unwrap(), vec![0.; 128]);
        assert!(n.apply_preserving_zero(&[0.; 127]).is_err());
        assert!(n.apply_preserving_zero(&[f32::NAN; 128]).is_err());
        let mut mixed = vec![0.; 128];
        mixed[0] = 0.5;
        let before = mixed.clone();
        assert_eq!(
            n.apply_preserving_zero(&mixed).unwrap(),
            n.apply(&mixed).unwrap()
        );
        assert_eq!(mixed, before);
        assert_eq!(n.apply_preserving_zero(&mixed).unwrap()[1], -2.);
    }
    #[test]
    fn standardizes_without_clipping_or_changing_input() {
        let memory = vec![0.5; 128];
        let original = memory.clone();
        let n = MemoryNormalizer::new(vec![0.25; 128], vec![0.125; 128]).unwrap();
        assert_eq!(n.apply(&memory).unwrap(), vec![2.; 128]);
        assert_eq!(memory, original);
        let constant = MemoryNormalizer::new(memory.clone(), vec![1.; 128]).unwrap();
        assert_eq!(constant.apply(&memory).unwrap(), vec![0.; 128]);
    }
    #[test]
    fn rejects_invalid_dimensions_scales_and_nonfinite_values() {
        for scales in [
            vec![1.; 127],
            vec![0.; 128],
            vec![-1.; 128],
            vec![f32::INFINITY; 128],
        ] {
            assert!(MemoryNormalizer::new(vec![0.; 128], scales).is_err());
        }
        assert!(MemoryNormalizer::new(vec![f32::NAN; 128], vec![1.; 128]).is_err());
        let n = MemoryNormalizer::new(vec![0.; 128], vec![1.; 128]).unwrap();
        assert!(n.apply(&[0.; 127]).is_err());
        assert!(n.apply(&[f32::NAN; 128]).is_err());
    }
}
