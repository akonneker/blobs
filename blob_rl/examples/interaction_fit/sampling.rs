//! Frozen training-only feeding strata; the uniform path preserves its RNG stream.
use rand::Rng;

pub struct FeedingPool {
    hard: Vec<usize>,
    remaining: Vec<usize>,
}

impl FeedingPool {
    pub fn new(hard: Vec<usize>, rows: usize) -> Result<Self, String> {
        if hard.is_empty()
            || hard.len() >= rows
            || hard.iter().any(|&i| i >= rows)
            || hard.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("feeding strata must be nonempty, sorted, unique and in range".into());
        }
        let remaining = (0..rows)
            .filter(|i| hard.binary_search(i).is_err())
            .collect();
        Ok(Self { hard, remaining })
    }

    pub fn sample(&self, rng: &mut impl Rng, position: usize) -> usize {
        let pool = if position < 32 {
            &self.hard
        } else {
            &self.remaining
        };
        pool[rng.random_range(0..pool.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn rejects_invalid_or_empty_strata() {
        for hard in [vec![], vec![0, 1, 2], vec![3], vec![1, 1], vec![2, 0]] {
            assert!(FeedingPool::new(hard, 3).is_err());
        }
    }

    #[test]
    fn draws_exactly_half_from_each_disjoint_stratum() {
        let pool = FeedingPool::new(vec![1, 4], 6).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(71);
        for _ in 0..100 {
            let batch: Vec<_> = (0..64).map(|i| pool.sample(&mut rng, i)).collect();
            assert!(batch[..32].iter().all(|i| [1, 4].contains(i)));
            assert!(batch[32..].iter().all(|i| [0, 2, 3, 5].contains(i)));
        }
    }
}
