//! Indexed replay archives paired with independently validated checkpoints.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::Range;

use sha2::{Digest, Sha256};

use super::checkpoint::{CheckpointError, CheckpointLimits, ReferenceCheckpoint};
use super::hashing::CanonicalHash;
use super::replay::{
    Reader, ReplayArchive, ReplayArchiveLimits, ReplayError, ReplayLimits, Writer,
};

pub const REPLAY_BUNDLE_FORMAT_VERSION: u16 = 1;
const REPLAY_BUNDLE_MAGIC: &[u8; 8] = b"BLBBND01";
const REPLAY_BUNDLE_DOMAIN: &[u8] = b"blob.replay.checkpoint-bundle";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayBundleLimits {
    pub max_bundle_bytes: usize,
    pub max_checkpoints: usize,
    pub archive: ReplayArchiveLimits,
    pub checkpoint: CheckpointLimits,
}

impl Default for ReplayBundleLimits {
    fn default() -> Self {
        Self {
            max_bundle_bytes: 1024 * 1024 * 1024,
            max_checkpoints: 1_000_000,
            archive: ReplayArchiveLimits::default(),
            checkpoint: CheckpointLimits::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayBundleError {
    TooLarge {
        actual: usize,
        limit: usize,
    },
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u16),
    IntegrityHashMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    TooManyCheckpoints {
        actual: u64,
        limit: usize,
    },
    MissingGenesisCheckpoint,
    NonCanonicalCheckpointOrder,
    CheckpointSequenceOutOfRange {
        sequence: u64,
        event_count: usize,
    },
    CheckpointRulesetMismatch {
        sequence: u64,
    },
    CheckpointStateMismatch {
        sequence: u64,
    },
    SeekOutOfRange {
        applied_events: usize,
        event_count: usize,
    },
    TrailingBytes(usize),
    Replay(ReplayError),
    Checkpoint(CheckpointError),
}

impl Display for ReplayBundleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { actual, limit } => {
                write!(
                    formatter,
                    "replay bundle is {actual} bytes; limit is {limit}"
                )
            }
            Self::Truncated => write!(formatter, "replay bundle is truncated"),
            Self::InvalidMagic => write!(formatter, "invalid replay bundle magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported replay bundle version {version}")
            }
            Self::IntegrityHashMismatch { expected, actual } => write!(
                formatter,
                "replay bundle integrity hash mismatch: expected {expected}, got {actual}"
            ),
            Self::TooManyCheckpoints { actual, limit } => write!(
                formatter,
                "replay bundle has {actual} checkpoints; limit is {limit}"
            ),
            Self::MissingGenesisCheckpoint => {
                write!(formatter, "replay bundle has no sequence-zero checkpoint")
            }
            Self::NonCanonicalCheckpointOrder => {
                write!(formatter, "replay checkpoints are not strictly ordered")
            }
            Self::CheckpointSequenceOutOfRange {
                sequence,
                event_count,
            } => write!(
                formatter,
                "checkpoint sequence {sequence} exceeds event count {event_count}"
            ),
            Self::CheckpointRulesetMismatch { sequence } => write!(
                formatter,
                "checkpoint at sequence {sequence} has the wrong compiled ruleset"
            ),
            Self::CheckpointStateMismatch { sequence } => write!(
                formatter,
                "checkpoint at sequence {sequence} does not match the replay state hash"
            ),
            Self::SeekOutOfRange {
                applied_events,
                event_count,
            } => write!(
                formatter,
                "cannot seek after {applied_events} events; archive contains {event_count}"
            ),
            Self::TrailingBytes(count) => {
                write!(formatter, "replay bundle has {count} trailing body bytes")
            }
            Self::Replay(error) => Display::fmt(error, formatter),
            Self::Checkpoint(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for ReplayBundleError {}

impl From<ReplayError> for ReplayBundleError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<CheckpointError> for ReplayBundleError {
    fn from(value: CheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplaySeek {
    /// Index to pass to [`ReplayBundle::checkpoint`].
    pub checkpoint_index: usize,
    /// Number of events already reflected by that checkpoint.
    pub checkpoint_applied_events: usize,
    /// Archive events the replay driver must process to reach its target.
    pub events: Range<usize>,
}

/// A canonical replay archive plus sparse checkpoints indexed by the number of
/// events already applied. Sequence zero is the replay genesis state.
#[derive(Debug, Clone)]
pub struct ReplayBundle {
    bytes: Vec<u8>,
    archive: ReplayArchive,
    checkpoints: Vec<(u64, ReferenceCheckpoint)>,
    bundle_hash: CanonicalHash,
}

impl ReplayBundle {
    pub fn from_parts(
        archive: ReplayArchive,
        checkpoints: Vec<(u64, ReferenceCheckpoint)>,
        limits: ReplayBundleLimits,
    ) -> Result<Self, ReplayBundleError> {
        validate_checkpoints(
            &archive,
            &checkpoints,
            limits.max_checkpoints,
            limits.archive.frame,
        )?;
        let mut writer = Writer::new();
        writer.raw(REPLAY_BUNDLE_MAGIC);
        writer.u16(REPLAY_BUNDLE_FORMAT_VERSION);
        writer.bytes(archive.to_bytes());
        writer.u64(checkpoints.len() as u64);
        for (sequence, checkpoint) in &checkpoints {
            writer.u64(*sequence);
            writer.bytes(&checkpoint.to_bytes());
        }
        let mut bytes = writer.finish();
        let total = bytes
            .len()
            .checked_add(32)
            .ok_or(ReplayBundleError::TooLarge {
                actual: usize::MAX,
                limit: limits.max_bundle_bytes,
            })?;
        if total > limits.max_bundle_bytes {
            return Err(ReplayBundleError::TooLarge {
                actual: total,
                limit: limits.max_bundle_bytes,
            });
        }
        let bundle_hash = hash_bundle_body(&bytes);
        bytes.extend_from_slice(bundle_hash.as_bytes());
        Ok(Self {
            bytes,
            archive,
            checkpoints,
            bundle_hash,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayBundleError> {
        Self::from_bytes_with_limits(bytes, ReplayBundleLimits::default())
    }

    pub fn from_bytes_with_limits(
        bytes: &[u8],
        limits: ReplayBundleLimits,
    ) -> Result<Self, ReplayBundleError> {
        if bytes.len() > limits.max_bundle_bytes {
            return Err(ReplayBundleError::TooLarge {
                actual: bytes.len(),
                limit: limits.max_bundle_bytes,
            });
        }
        if bytes.len() < 32 {
            return Err(ReplayBundleError::Truncated);
        }
        let body_length = bytes.len() - 32;
        let (body, submitted_hash) = bytes.split_at(body_length);
        let expected = CanonicalHash::from_bytes(
            submitted_hash
                .try_into()
                .map_err(|_| ReplayBundleError::Truncated)?,
        );
        let actual = hash_bundle_body(body);
        if actual != expected {
            return Err(ReplayBundleError::IntegrityHashMismatch { expected, actual });
        }

        let mut reader = Reader::new(body);
        if reader.take(REPLAY_BUNDLE_MAGIC.len())? != REPLAY_BUNDLE_MAGIC {
            return Err(ReplayBundleError::InvalidMagic);
        }
        let version = reader.u16()?;
        if version != REPLAY_BUNDLE_FORMAT_VERSION {
            return Err(ReplayBundleError::UnsupportedVersion(version));
        }
        let archive_bytes = reader.bytes("replay archive", limits.archive.max_archive_bytes)?;
        let archive = ReplayArchive::from_bytes_with_limits(&archive_bytes, limits.archive)?;
        let count = reader.u64()?;
        if count > limits.max_checkpoints as u64 {
            return Err(ReplayBundleError::TooManyCheckpoints {
                actual: count,
                limit: limits.max_checkpoints,
            });
        }
        let mut checkpoints = Vec::with_capacity((count as usize).min(4096));
        for _ in 0..count {
            let sequence = reader.u64()?;
            let checkpoint_bytes = reader.bytes(
                "canonical checkpoint",
                limits.checkpoint.max_checkpoint_bytes,
            )?;
            checkpoints.push((
                sequence,
                ReferenceCheckpoint::from_bytes_with_limits(&checkpoint_bytes, limits.checkpoint)?,
            ));
        }
        if reader.remaining() != 0 {
            return Err(ReplayBundleError::TrailingBytes(reader.remaining()));
        }
        validate_checkpoints(
            &archive,
            &checkpoints,
            limits.max_checkpoints,
            limits.archive.frame,
        )?;
        Ok(Self {
            bytes: bytes.to_vec(),
            archive,
            checkpoints,
            bundle_hash: expected,
        })
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn archive(&self) -> &ReplayArchive {
        &self.archive
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    pub fn checkpoint(&self, index: usize) -> Option<(u64, &ReferenceCheckpoint)> {
        self.checkpoints
            .get(index)
            .map(|(sequence, checkpoint)| (*sequence, checkpoint))
    }

    pub fn checkpoints(&self) -> impl Iterator<Item = (u64, &ReferenceCheckpoint)> {
        self.checkpoints
            .iter()
            .map(|(sequence, checkpoint)| (*sequence, checkpoint))
    }

    pub const fn bundle_hash(&self) -> CanonicalHash {
        self.bundle_hash
    }

    /// Locates the newest checkpoint at or before a target state. The target
    /// is expressed as the number of archive events already applied.
    pub fn seek(&self, applied_events: usize) -> Result<ReplaySeek, ReplayBundleError> {
        if applied_events > self.archive.event_count() {
            return Err(ReplayBundleError::SeekOutOfRange {
                applied_events,
                event_count: self.archive.event_count(),
            });
        }
        let insertion = self
            .checkpoints
            .partition_point(|(sequence, _)| *sequence <= applied_events as u64);
        let checkpoint_index = insertion
            .checked_sub(1)
            .ok_or(ReplayBundleError::MissingGenesisCheckpoint)?;
        let start = usize::try_from(self.checkpoints[checkpoint_index].0).map_err(|_| {
            ReplayBundleError::CheckpointSequenceOutOfRange {
                sequence: self.checkpoints[checkpoint_index].0,
                event_count: self.archive.event_count(),
            }
        })?;
        Ok(ReplaySeek {
            checkpoint_index,
            checkpoint_applied_events: start,
            events: start..applied_events,
        })
    }
}

fn validate_checkpoints(
    archive: &ReplayArchive,
    checkpoints: &[(u64, ReferenceCheckpoint)],
    maximum: usize,
    frame_limits: ReplayLimits,
) -> Result<(), ReplayBundleError> {
    if checkpoints.len() > maximum {
        return Err(ReplayBundleError::TooManyCheckpoints {
            actual: checkpoints.len() as u64,
            limit: maximum,
        });
    }
    if checkpoints.first().map(|entry| entry.0) != Some(0) {
        return Err(ReplayBundleError::MissingGenesisCheckpoint);
    }
    if !checkpoints.windows(2).all(|pair| pair[0].0 < pair[1].0) {
        return Err(ReplayBundleError::NonCanonicalCheckpointOrder);
    }
    for (sequence, checkpoint) in checkpoints {
        let sequence_index = usize::try_from(*sequence).map_err(|_| {
            ReplayBundleError::CheckpointSequenceOutOfRange {
                sequence: *sequence,
                event_count: archive.event_count(),
            }
        })?;
        if sequence_index > archive.event_count() {
            return Err(ReplayBundleError::CheckpointSequenceOutOfRange {
                sequence: *sequence,
                event_count: archive.event_count(),
            });
        }
        if checkpoint.compiled_ruleset_hash() != archive.compiled_ruleset_hash() {
            return Err(ReplayBundleError::CheckpointRulesetMismatch {
                sequence: *sequence,
            });
        }
        let expected_state = if sequence_index == 0 {
            archive.initial_state_hash()
        } else {
            archive
                .event(sequence_index - 1, frame_limits)?
                .post_state_hash
        };
        if checkpoint.state_hash() != expected_state {
            return Err(ReplayBundleError::CheckpointStateMismatch {
                sequence: *sequence,
            });
        }
    }
    Ok(())
}

fn hash_bundle_body(body: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((REPLAY_BUNDLE_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(REPLAY_BUNDLE_DOMAIN);
    hasher.update(REPLAY_BUNDLE_FORMAT_VERSION.to_le_bytes());
    hasher.update((body.len() as u64).to_le_bytes());
    hasher.update(body);
    CanonicalHash::from_bytes(hasher.finalize().into())
}
