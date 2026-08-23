//! Portable parsing of signed server match attestations.

use std::error::Error;
use std::fmt::{Display, Formatter};

use sha2::{Digest, Sha256};

use super::{
    CanonicalHash, MatchVerificationError, MatchVerificationLimits, MatchVerificationManifest,
};

pub const MATCH_ATTESTATION_FORMAT_VERSION: u16 = 1;
pub const ED25519_PUBLIC_KEY_BYTES: usize = 32;
pub const ED25519_SIGNATURE_BYTES: usize = 64;
const ATTESTATION_MAGIC: &[u8; 8] = b"BLBATT01";
const ATTESTATION_DOMAIN: &[u8] = b"blob.match.attestation";
const KEY_ID_DOMAIN: &[u8] = b"blob.server.signing-key";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchAttestationError {
    TooLarge { actual: usize, limit: usize },
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u16),
    TrailingBytes(usize),
    InvalidPublicKeyLength(usize),
    SigningKeyMismatch,
    Manifest(MatchVerificationError),
}

impl Display for MatchAttestationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { actual, limit } => {
                write!(
                    formatter,
                    "match attestation is {actual} bytes; limit is {limit}"
                )
            }
            Self::Truncated => write!(formatter, "match attestation is truncated"),
            Self::InvalidMagic => write!(formatter, "invalid match attestation magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported match attestation version {version}")
            }
            Self::TrailingBytes(count) => {
                write!(formatter, "match attestation has {count} trailing bytes")
            }
            Self::InvalidPublicKeyLength(length) => {
                write!(
                    formatter,
                    "Ed25519 public key has {length} bytes; expected 32"
                )
            }
            Self::SigningKeyMismatch => write!(formatter, "attestation signing key ID mismatch"),
            Self::Manifest(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for MatchAttestationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Manifest(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatchVerificationError> for MatchAttestationError {
    fn from(value: MatchVerificationError) -> Self {
        Self::Manifest(value)
    }
}

/// Signature envelope whose cryptographic verification can be performed by a
/// native Ed25519 implementation or browser WebCrypto.
#[derive(Debug, Clone)]
pub struct MatchAttestation {
    bytes: Vec<u8>,
    key_id: CanonicalHash,
    manifest: MatchVerificationManifest,
    signature: [u8; ED25519_SIGNATURE_BYTES],
}

impl MatchAttestation {
    pub fn from_signature(
        manifest: MatchVerificationManifest,
        key_id: CanonicalHash,
        signature: [u8; ED25519_SIGNATURE_BYTES],
    ) -> Self {
        let bytes = encode(key_id, manifest.to_bytes(), &signature);
        Self {
            bytes,
            key_id,
            manifest,
            signature,
        }
    }

    pub fn from_bytes(
        bytes: &[u8],
        max_bytes: usize,
        manifest_limits: MatchVerificationLimits,
    ) -> Result<Self, MatchAttestationError> {
        if bytes.len() > max_bytes {
            return Err(MatchAttestationError::TooLarge {
                actual: bytes.len(),
                limit: max_bytes,
            });
        }
        let fixed_prefix = ATTESTATION_MAGIC.len() + 2 + 32 + 8;
        let minimum = fixed_prefix + ED25519_SIGNATURE_BYTES;
        if bytes.len() < minimum {
            return Err(MatchAttestationError::Truncated);
        }
        let mut position = 0;
        if take(bytes, &mut position, ATTESTATION_MAGIC.len())? != ATTESTATION_MAGIC {
            return Err(MatchAttestationError::InvalidMagic);
        }
        let version = u16::from_le_bytes(
            take(bytes, &mut position, 2)?
                .try_into()
                .map_err(|_| MatchAttestationError::Truncated)?,
        );
        if version != MATCH_ATTESTATION_FORMAT_VERSION {
            return Err(MatchAttestationError::UnsupportedVersion(version));
        }
        let key_id = CanonicalHash::from_bytes(
            take(bytes, &mut position, 32)?
                .try_into()
                .map_err(|_| MatchAttestationError::Truncated)?,
        );
        let manifest_length = u64::from_le_bytes(
            take(bytes, &mut position, 8)?
                .try_into()
                .map_err(|_| MatchAttestationError::Truncated)?,
        );
        let manifest_length =
            usize::try_from(manifest_length).map_err(|_| MatchAttestationError::TooLarge {
                actual: usize::MAX,
                limit: max_bytes,
            })?;
        let manifest = MatchVerificationManifest::from_bytes_with_limits(
            take(bytes, &mut position, manifest_length)?,
            manifest_limits,
        )?;
        let signature = take(bytes, &mut position, ED25519_SIGNATURE_BYTES)?
            .try_into()
            .map_err(|_| MatchAttestationError::Truncated)?;
        if position != bytes.len() {
            return Err(MatchAttestationError::TrailingBytes(bytes.len() - position));
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            key_id,
            manifest,
            signature,
        })
    }

    pub fn verify_key_id(&self, public_key: &[u8]) -> Result<(), MatchAttestationError> {
        if public_key.len() != ED25519_PUBLIC_KEY_BYTES {
            return Err(MatchAttestationError::InvalidPublicKeyLength(
                public_key.len(),
            ));
        }
        if attestation_key_id(public_key) != self.key_id {
            return Err(MatchAttestationError::SigningKeyMismatch);
        }
        Ok(())
    }

    pub fn signing_message(&self) -> Vec<u8> {
        attestation_signing_message(self.manifest.to_bytes())
    }

    pub fn signature(&self) -> &[u8; ED25519_SIGNATURE_BYTES] {
        &self.signature
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn key_id(&self) -> CanonicalHash {
        self.key_id
    }

    pub fn manifest(&self) -> &MatchVerificationManifest {
        &self.manifest
    }
}

pub fn attestation_signing_message(manifest: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(8 + ATTESTATION_DOMAIN.len() + 2 + manifest.len());
    message.extend_from_slice(&(ATTESTATION_DOMAIN.len() as u64).to_le_bytes());
    message.extend_from_slice(ATTESTATION_DOMAIN);
    message.extend_from_slice(&MATCH_ATTESTATION_FORMAT_VERSION.to_le_bytes());
    message.extend_from_slice(manifest);
    message
}

pub fn attestation_key_id(public_key: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((KEY_ID_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(KEY_ID_DOMAIN);
    hasher.update((public_key.len() as u64).to_le_bytes());
    hasher.update(public_key);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

fn encode(
    key_id: CanonicalHash,
    manifest: &[u8],
    signature: &[u8; ED25519_SIGNATURE_BYTES],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + 2 + 32 + 8 + manifest.len() + signature.len());
    bytes.extend_from_slice(ATTESTATION_MAGIC);
    bytes.extend_from_slice(&MATCH_ATTESTATION_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(key_id.as_bytes());
    bytes.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
    bytes.extend_from_slice(manifest);
    bytes.extend_from_slice(signature);
    bytes
}

fn take<'a>(
    bytes: &'a [u8],
    position: &mut usize,
    count: usize,
) -> Result<&'a [u8], MatchAttestationError> {
    let end = position
        .checked_add(count)
        .ok_or(MatchAttestationError::Truncated)?;
    let value = bytes
        .get(*position..end)
        .ok_or(MatchAttestationError::Truncated)?;
    *position = end;
    Ok(value)
}
