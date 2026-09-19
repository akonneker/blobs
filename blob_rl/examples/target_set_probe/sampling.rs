//! Research evaluation uses the same portable categorical arithmetic as deployment.
//! Inputs here are already in canonical geometry order.
pub(super) fn sample(logits: &[f32], legal: &[bool], word: u64) -> Option<usize> {
    blob_policy::sampling::CategoricalDistribution::from_logits(logits, legal)
        .ok()?
        .sample(word)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_mind_utils::choose_slot_by_quantile;
    #[test]
    fn uniform_logits_match_exact_u64_teacher_boundaries_for_every_subset() {
        for mask in 1..=255_u32 {
            let legal: Vec<_> = (0..8).map(|i| mask & (1 << i) != 0).collect();
            let candidates: Vec<_> = (0..8_u8).filter(|i| legal[*i as usize]).collect();
            let n = candidates.len() as u128;
            let mut words = vec![0, u64::MAX];
            for k in 1..n {
                let boundary = (k * (1_u128 << 64)).div_ceil(n) as u64;
                words.extend([boundary - 1, boundary]);
            }
            for word in words {
                assert_eq!(
                    sample(&[1.25; 8], &legal, word),
                    choose_slot_by_quantile(&candidates, word).map(usize::from)
                );
            }
        }
    }
    #[test]
    fn invalid_inputs_are_rejected_and_masked_values_cannot_be_selected() {
        assert_eq!(sample(&[], &[], 0), None);
        assert_eq!(sample(&[0.], &[], 0), None);
        assert_eq!(sample(&[0.], &[false], 0), None);
        assert_eq!(sample(&[f32::NAN], &[true], 0), None);
        assert_eq!(
            sample(&[f32::INFINITY, 0.], &[false, true], u64::MAX),
            Some(1)
        );
        assert_eq!(sample(&[0.; 33], &[true; 33], 0), None);
        for word in [0, 1 << 63, u64::MAX] {
            assert_eq!(
                sample(&[0., -1000., 1000.], &[true, true, false], word),
                Some(0)
            );
        }
    }

    #[test]
    fn uniform_logits_match_teacher_at_full_capacity_boundaries() {
        for count in 1..=32 {
            let candidates: Vec<u8> = (0..count as u8).collect();
            let mut words = vec![0, u64::MAX];
            for k in 1..count {
                let boundary = ((k as u128 * (1_u128 << 64)).div_ceil(count as u128)) as u64;
                words.extend([boundary - 1, boundary, boundary + 1]);
            }
            for word in words {
                assert_eq!(
                    sample(&vec![0.; count], &vec![true; count], word),
                    choose_slot_by_quantile(&candidates, word).map(usize::from)
                );
            }
        }
    }
}
