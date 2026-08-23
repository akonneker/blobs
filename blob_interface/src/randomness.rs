//! Private deterministic random samples supplied to one mind invocation.

#[cfg(not(target_arch = "wasm32"))]
use ring::digest::{Context, SHA256};
#[cfg(target_arch = "wasm32")]
use sha2::{Digest, Sha256};

pub const PRIVATE_RANDOM_BYTES: usize = 32;
const MATCH_SECRET_DOMAIN: &[u8] = b"blob.match.secret.v1";
const DECISION_RANDOM_DOMAIN: &[u8] = b"blob.cell.decision.random.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrivateRandom([u8; PRIVATE_RANDOM_BYTES]);

impl PrivateRandom {
    pub const ZERO: Self = Self([0; PRIVATE_RANDOM_BYTES]);

    pub const fn from_bytes(bytes: [u8; PRIVATE_RANDOM_BYTES]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; PRIVATE_RANDOM_BYTES] {
        &self.0
    }

    /// Returns one bounded random sample. This is random output, not the match
    /// secret or the engine's derivation seed.
    pub fn sample_u64(&self, index: usize) -> u64 {
        let start = (index % 4) * 8;
        u64::from_le_bytes(self.0[start..start + 8].try_into().unwrap())
    }
}

/// Reusable decision-randomness prefix for one match. The secret and domain
/// are absorbed once; each cell derivation clones that private hash state and
/// appends only lineage and sequence. It never exposes the prefix or secret.
#[derive(Clone)]
pub struct PrivateRandomDeriver {
    #[cfg(not(target_arch = "wasm32"))]
    prefix: Context,
    #[cfg(target_arch = "wasm32")]
    prefix: Sha256,
}

impl PrivateRandomDeriver {
    pub fn new(match_secret: &[u8; 32]) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let mut prefix = Context::new(&SHA256);
        #[cfg(target_arch = "wasm32")]
        let mut prefix = Sha256::new();
        prefix.update(&(DECISION_RANDOM_DOMAIN.len() as u64).to_le_bytes());
        prefix.update(DECISION_RANDOM_DOMAIN);
        prefix.update(match_secret);
        Self { prefix }
    }

    pub fn derive(&self, cell_lineage: u64, decision_sequence: u64) -> PrivateRandom {
        let mut hasher = self.prefix.clone();
        hasher.update(&cell_lineage.to_le_bytes());
        hasher.update(&decision_sequence.to_le_bytes());
        #[cfg(not(target_arch = "wasm32"))]
        let bytes = hasher.finish().as_ref().try_into().unwrap();
        #[cfg(target_arch = "wasm32")]
        let bytes = hasher.finalize().into();
        PrivateRandom::from_bytes(bytes)
    }
}

/// Creates the private match secret used by local deterministic simulations.
/// Online servers should choose an unpredictable seed and keep it server-side.
pub fn match_secret_from_seed(seed: u64) -> [u8; 32] {
    sha256_chunks(&[
        &(MATCH_SECRET_DOMAIN.len() as u64).to_le_bytes(),
        MATCH_SECRET_DOMAIN,
        &seed.to_le_bytes(),
    ])
}

/// Derives samples for exactly one cell decision. Neither the secret nor the
/// lineage/sequence inputs are included in the returned bytes.
pub fn derive_private_random(
    match_secret: &[u8; 32],
    cell_lineage: u64,
    decision_sequence: u64,
) -> PrivateRandom {
    PrivateRandomDeriver::new(match_secret).derive(cell_lineage, decision_sequence)
}

#[cfg(not(target_arch = "wasm32"))]
fn sha256_chunks(chunks: &[&[u8]]) -> [u8; 32] {
    let mut context = Context::new(&SHA256);
    for chunk in chunks {
        context.update(chunk);
    }
    context.finish().as_ref().try_into().unwrap()
}

#[cfg(target_arch = "wasm32")]
fn sha256_chunks(chunks: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for chunk in chunks {
        hasher.update(chunk);
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_stable_private_and_cell_scoped() {
        let secret = match_secret_from_seed(42);
        assert_eq!(
            secret,
            [
                0x07, 0x87, 0xb3, 0x15, 0x91, 0x76, 0x90, 0x0c, 0x7b, 0xf5, 0x13, 0x74, 0x61, 0x90,
                0x3d, 0x1f, 0x12, 0x57, 0x42, 0x0a, 0x0e, 0x41, 0xd4, 0x2c, 0x00, 0xdc, 0xf1, 0x54,
                0x84, 0xbc, 0xb8, 0x5f,
            ]
        );
        let same = derive_private_random(&secret, 7, 3);
        assert_eq!(
            same.as_bytes(),
            &[
                0x89, 0x54, 0x84, 0xef, 0x2f, 0xbe, 0xc0, 0xca, 0xc8, 0xe2, 0xf7, 0xc8, 0x76, 0x47,
                0x87, 0xac, 0x79, 0xc2, 0x55, 0x8b, 0x8b, 0x2f, 0x9e, 0x2c, 0x4e, 0x0b, 0xa7, 0x57,
                0xcc, 0xde, 0x71, 0x1d,
            ]
        );
        assert_eq!(same, derive_private_random(&secret, 7, 3));
        assert_ne!(same, derive_private_random(&secret, 8, 3));
        assert_ne!(same, derive_private_random(&secret, 7, 4));
        assert_ne!(same.as_bytes(), &secret);
        let deriver = PrivateRandomDeriver::new(&secret);
        assert_eq!(same, deriver.derive(7, 3));
    }
}
