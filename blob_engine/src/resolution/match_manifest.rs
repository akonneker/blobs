//! Canonical match envelope binding Minds, ABI, runtime, rules, and replay.

use std::error::Error;
use std::fmt::{Display, Formatter};

use sha2::{Digest, Sha256};

use super::hashing::CanonicalHash;
use super::reference::SimTime;
use super::replay::{Reader, ReplayChainCursor, ReplayError, Writer};
use super::replay_segment::ReplayManifest;

pub const MATCH_VERIFICATION_FORMAT_VERSION: u16 = 2;
const MATCH_MAGIC: &[u8; 8] = b"BLBVER01";
const MATCH_DOMAIN: &[u8] = b"blob.match.verification";
const ARTIFACT_DOMAIN: &[u8] = b"blob.mind.artifact";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchVerificationLimits {
    pub max_bytes: usize,
    pub max_mind_artifacts: usize,
}

impl Default for MatchVerificationLimits {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024,
            max_mind_artifacts: 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MindArtifactBinding {
    /// Server-side competition slot. This is never part of a Mind observation.
    pub slot: u64,
    pub artifact_hash: CanonicalHash,
}

/// Versioned host-capability and structural profile required of every Mind
/// artifact in a match. Runtime implementation and worker policy are excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MindRuntimeProfile {
    ExtismPdkDeterministicV1,
}

impl MindRuntimeProfile {
    const fn tag(self) -> u16 {
        match self {
            Self::ExtismPdkDeterministicV1 => 1,
        }
    }

    fn from_tag(tag: u16) -> Result<Self, MatchVerificationError> {
        match tag {
            1 => Ok(Self::ExtismPdkDeterministicV1),
            _ => Err(MatchVerificationError::UnsupportedRuntimeProfile(tag)),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::ExtismPdkDeterministicV1 => "extism_pdk_deterministic_v1",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchVerificationError {
    TooLarge { actual: usize, limit: usize },
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u16),
    UnsupportedRuntimeProfile(u16),
    HashMismatch,
    TooManyArtifacts { actual: u64, limit: usize },
    NonCanonicalArtifacts,
    TrailingBytes(usize),
    AbiMismatch,
    RuntimeProfileMismatch,
    RuntimeMismatch,
    ArtifactMismatch,
    ReplayMismatch,
    Replay(ReplayError),
}

impl Display for MatchVerificationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { actual, limit } => {
                write!(
                    formatter,
                    "match manifest is {actual} bytes; limit is {limit}"
                )
            }
            Self::Truncated => write!(formatter, "match manifest is truncated"),
            Self::InvalidMagic => write!(formatter, "invalid match manifest magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported match manifest version {version}")
            }
            Self::UnsupportedRuntimeProfile(profile) => {
                write!(formatter, "unsupported Mind runtime profile {profile}")
            }
            Self::HashMismatch => write!(formatter, "match manifest hash mismatch"),
            Self::TooManyArtifacts { actual, limit } => write!(
                formatter,
                "match manifest has {actual} Mind artifacts; limit is {limit}"
            ),
            Self::NonCanonicalArtifacts => {
                write!(formatter, "Mind artifact bindings are not strictly ordered")
            }
            Self::TrailingBytes(count) => {
                write!(formatter, "match manifest has {count} trailing bytes")
            }
            Self::AbiMismatch => write!(formatter, "Mind ABI hash mismatch"),
            Self::RuntimeProfileMismatch => write!(formatter, "Mind runtime profile mismatch"),
            Self::RuntimeMismatch => write!(formatter, "initial authoritative runtime mismatch"),
            Self::ArtifactMismatch => write!(formatter, "Mind artifact bindings mismatch"),
            Self::ReplayMismatch => write!(formatter, "replay manifest binding mismatch"),
            Self::Replay(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for MatchVerificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Replay(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ReplayError> for MatchVerificationError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

/// Hashes the exact submitted Mind bytes, independent of filenames or storage.
pub fn mind_artifact_hash(bytes: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((ARTIFACT_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(ARTIFACT_DOMAIN);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

/// Immutable inputs and outputs of one server-verifiable match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchVerificationManifest {
    bytes: Vec<u8>,
    match_id_hash: CanonicalHash,
    mind_abi_hash: CanonicalHash,
    mind_runtime_profile: MindRuntimeProfile,
    compiled_ruleset_hash: CanonicalHash,
    initial_state_hash: CanonicalHash,
    initial_runtime_hash: CanonicalHash,
    replay_manifest_hash: CanonicalHash,
    final_cursor: ReplayChainCursor,
    mind_artifacts: Vec<MindArtifactBinding>,
    manifest_hash: CanonicalHash,
}

impl MatchVerificationManifest {
    pub fn from_replay(
        match_id_hash: CanonicalHash,
        mind_abi_hash: CanonicalHash,
        mind_runtime_profile: MindRuntimeProfile,
        initial_runtime_hash: CanonicalHash,
        mind_artifacts: Vec<MindArtifactBinding>,
        replay: &ReplayManifest,
        limits: MatchVerificationLimits,
    ) -> Result<Self, MatchVerificationError> {
        validate_artifacts(&mind_artifacts, limits.max_mind_artifacts)?;
        let compiled_ruleset_hash = replay.compiled_ruleset_hash();
        let initial_state_hash = replay.initial_state_hash();
        let replay_manifest_hash = replay.manifest_hash();
        let final_cursor = replay.final_cursor();
        let mut writer = Writer::new();
        writer.raw(MATCH_MAGIC);
        writer.u16(MATCH_VERIFICATION_FORMAT_VERSION);
        writer.hash(match_id_hash);
        writer.hash(mind_abi_hash);
        writer.u16(mind_runtime_profile.tag());
        writer.hash(compiled_ruleset_hash);
        writer.hash(initial_state_hash);
        writer.hash(initial_runtime_hash);
        writer.hash(replay_manifest_hash);
        encode_cursor(&mut writer, final_cursor);
        writer.u64(mind_artifacts.len() as u64);
        for binding in &mind_artifacts {
            writer.u64(binding.slot);
            writer.hash(binding.artifact_hash);
        }
        let body = writer.finish();
        let total = body
            .len()
            .checked_add(32)
            .ok_or(MatchVerificationError::TooLarge {
                actual: usize::MAX,
                limit: limits.max_bytes,
            })?;
        if total > limits.max_bytes {
            return Err(MatchVerificationError::TooLarge {
                actual: total,
                limit: limits.max_bytes,
            });
        }
        let manifest_hash = hash_body(&body);
        let mut bytes = body;
        bytes.extend_from_slice(manifest_hash.as_bytes());
        Ok(Self {
            bytes,
            match_id_hash,
            mind_abi_hash,
            mind_runtime_profile,
            compiled_ruleset_hash,
            initial_state_hash,
            initial_runtime_hash,
            replay_manifest_hash,
            final_cursor,
            mind_artifacts,
            manifest_hash,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MatchVerificationError> {
        Self::from_bytes_with_limits(bytes, MatchVerificationLimits::default())
    }

    pub fn from_bytes_with_limits(
        bytes: &[u8],
        limits: MatchVerificationLimits,
    ) -> Result<Self, MatchVerificationError> {
        if bytes.len() > limits.max_bytes {
            return Err(MatchVerificationError::TooLarge {
                actual: bytes.len(),
                limit: limits.max_bytes,
            });
        }
        if bytes.len() < 32 {
            return Err(MatchVerificationError::Truncated);
        }
        let body_length = bytes.len() - 32;
        let body = &bytes[..body_length];
        let actual_hash = hash_body(body);
        let stored_hash = CanonicalHash::from_bytes(
            bytes[body_length..]
                .try_into()
                .map_err(|_| MatchVerificationError::Truncated)?,
        );
        if actual_hash != stored_hash {
            return Err(MatchVerificationError::HashMismatch);
        }
        let mut reader = Reader::new(body);
        if reader.take(MATCH_MAGIC.len())? != MATCH_MAGIC {
            return Err(MatchVerificationError::InvalidMagic);
        }
        let version = reader.u16()?;
        if version != MATCH_VERIFICATION_FORMAT_VERSION {
            return Err(MatchVerificationError::UnsupportedVersion(version));
        }
        let match_id_hash = reader.hash()?;
        let mind_abi_hash = reader.hash()?;
        let mind_runtime_profile = MindRuntimeProfile::from_tag(reader.u16()?)?;
        let compiled_ruleset_hash = reader.hash()?;
        let initial_state_hash = reader.hash()?;
        let initial_runtime_hash = reader.hash()?;
        let replay_manifest_hash = reader.hash()?;
        let final_cursor = decode_cursor(&mut reader)?;
        let count = reader.u64()?;
        if count > limits.max_mind_artifacts as u64 {
            return Err(MatchVerificationError::TooManyArtifacts {
                actual: count,
                limit: limits.max_mind_artifacts,
            });
        }
        let count =
            usize::try_from(count).map_err(|_| MatchVerificationError::TooManyArtifacts {
                actual: count,
                limit: limits.max_mind_artifacts,
            })?;
        let mut mind_artifacts = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            mind_artifacts.push(MindArtifactBinding {
                slot: reader.u64()?,
                artifact_hash: reader.hash()?,
            });
        }
        if reader.remaining() != 0 {
            return Err(MatchVerificationError::TrailingBytes(reader.remaining()));
        }
        validate_artifacts(&mind_artifacts, limits.max_mind_artifacts)?;
        Ok(Self {
            bytes: bytes.to_vec(),
            match_id_hash,
            mind_abi_hash,
            mind_runtime_profile,
            compiled_ruleset_hash,
            initial_state_hash,
            initial_runtime_hash,
            replay_manifest_hash,
            final_cursor,
            mind_artifacts,
            manifest_hash: stored_hash,
        })
    }

    pub fn verify_context(
        &self,
        mind_abi_hash: CanonicalHash,
        mind_runtime_profile: MindRuntimeProfile,
        initial_runtime_hash: CanonicalHash,
        mind_artifacts: &[MindArtifactBinding],
        replay: &ReplayManifest,
    ) -> Result<(), MatchVerificationError> {
        if self.mind_abi_hash != mind_abi_hash {
            return Err(MatchVerificationError::AbiMismatch);
        }
        if self.mind_runtime_profile != mind_runtime_profile {
            return Err(MatchVerificationError::RuntimeProfileMismatch);
        }
        if self.initial_runtime_hash != initial_runtime_hash {
            return Err(MatchVerificationError::RuntimeMismatch);
        }
        if self.mind_artifacts != mind_artifacts {
            return Err(MatchVerificationError::ArtifactMismatch);
        }
        if self.compiled_ruleset_hash != replay.compiled_ruleset_hash()
            || self.initial_state_hash != replay.initial_state_hash()
            || self.replay_manifest_hash != replay.manifest_hash()
            || self.final_cursor != replay.final_cursor()
        {
            return Err(MatchVerificationError::ReplayMismatch);
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn manifest_hash(&self) -> CanonicalHash {
        self.manifest_hash
    }

    pub const fn match_id_hash(&self) -> CanonicalHash {
        self.match_id_hash
    }

    pub const fn mind_abi_hash(&self) -> CanonicalHash {
        self.mind_abi_hash
    }

    pub const fn mind_runtime_profile(&self) -> MindRuntimeProfile {
        self.mind_runtime_profile
    }

    pub const fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.compiled_ruleset_hash
    }

    pub const fn initial_state_hash(&self) -> CanonicalHash {
        self.initial_state_hash
    }

    pub const fn initial_runtime_hash(&self) -> CanonicalHash {
        self.initial_runtime_hash
    }

    pub const fn replay_manifest_hash(&self) -> CanonicalHash {
        self.replay_manifest_hash
    }

    pub const fn final_cursor(&self) -> ReplayChainCursor {
        self.final_cursor
    }

    pub fn mind_artifacts(&self) -> &[MindArtifactBinding] {
        &self.mind_artifacts
    }
}

fn validate_artifacts(
    artifacts: &[MindArtifactBinding],
    limit: usize,
) -> Result<(), MatchVerificationError> {
    if artifacts.len() > limit {
        return Err(MatchVerificationError::TooManyArtifacts {
            actual: artifacts.len() as u64,
            limit,
        });
    }
    if artifacts
        .windows(2)
        .any(|pair| pair[0].slot >= pair[1].slot)
    {
        return Err(MatchVerificationError::NonCanonicalArtifacts);
    }
    Ok(())
}

fn encode_cursor(writer: &mut Writer, cursor: ReplayChainCursor) {
    writer.u64(cursor.next_sequence);
    writer.hash(cursor.previous_event_hash);
    writer.hash(cursor.previous_state_hash);
    match cursor.previous_completion {
        Some(time) => {
            writer.u8(1);
            writer.u64(time.0);
        }
        None => writer.u8(0),
    }
}

fn decode_cursor(reader: &mut Reader<'_>) -> Result<ReplayChainCursor, MatchVerificationError> {
    let next_sequence = reader.u64()?;
    let previous_event_hash = reader.hash()?;
    let previous_state_hash = reader.hash()?;
    let previous_completion = match reader.u8()? {
        0 => None,
        1 => Some(SimTime(reader.u64()?)),
        tag => {
            return Err(MatchVerificationError::Replay(ReplayError::InvalidTag {
                field: "match completion option",
                tag,
            }))
        }
    };
    Ok(ReplayChainCursor {
        next_sequence,
        previous_event_hash,
        previous_state_hash,
        previous_completion,
    })
}

fn hash_body(body: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((MATCH_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(MATCH_DOMAIN);
    hasher.update(MATCH_VERIFICATION_FORMAT_VERSION.to_le_bytes());
    hasher.update(body);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_profile_tags_are_forward_only() {
        assert_eq!(
            MindRuntimeProfile::from_tag(1),
            Ok(MindRuntimeProfile::ExtismPdkDeterministicV1)
        );
        assert_eq!(
            MindRuntimeProfile::from_tag(2),
            Err(MatchVerificationError::UnsupportedRuntimeProfile(2))
        );
    }
}
