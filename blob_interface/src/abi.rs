//! Stable identification of the serialized Mind boundary.

use sha2::{Digest, Sha256};

pub const REFERENCE_MIND_ABI_VERSION: u16 = 6;
const REFERENCE_MIND_ABI_DOMAIN: &[u8] = b"blob.reference-mind.abi";
const REFERENCE_MIND_SCHEMA: &[u8] = include_bytes!("../interface/reference_mind.capnp");

/// Hash of the canonical Mind input and action contract.
pub fn reference_mind_abi_hash() -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((REFERENCE_MIND_ABI_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(REFERENCE_MIND_ABI_DOMAIN);
    hasher.update(REFERENCE_MIND_ABI_VERSION.to_le_bytes());
    hasher.update((REFERENCE_MIND_SCHEMA.len() as u64).to_le_bytes());
    hasher.update(REFERENCE_MIND_SCHEMA);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_abi_hash_is_stable() {
        assert_eq!(
            hex(&reference_mind_abi_hash()),
            "28edbb64da5c68f1c792e29b340508070f8a5993e298dc734a632d1171ec2c98"
        );
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
