//! Canonical, bounded replay frames and event-chain verification.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use blob_interface::reference_mind::{ReferenceMemoryUpdate, ReferenceSignalEmission};
use sha2::{Digest, Sha256};

use super::delta::{CellDelta, SimulationDelta, TileDelta};
use super::hashing::CanonicalHash;
use super::neighborhood::{LocalSlot, TileIndex};
use super::reference::{
    AccessMode, ActionOutcome, ActionRequest, AttackDamage, BatchReport, CellColdState, CellKey,
    CellState, CommitReceipt, EffortTier, OutcomeStatus, PendingAction, RejectReason,
    ResourceClaim, ResourceKey, SimTime, TerrainChange, TileState,
};

pub const REPLAY_FORMAT_VERSION: u16 = 10;
const REPLAY_MAGIC: &[u8; 8] = b"BLBRPL01";
const GENESIS_DOMAIN: &[u8] = b"blob.replay.genesis";
const EVENT_DOMAIN: &[u8] = b"blob.replay.batch";
const ARCHIVE_DOMAIN: &[u8] = b"blob.replay.archive";
const ARCHIVE_MAGIC: &[u8; 8] = b"BLBARC01";
const MAX_CANONICAL_TILE_INDEX: u64 = u32::MAX as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayLimits {
    pub max_frame_bytes: usize,
    pub max_commitments: usize,
    pub max_outcomes: usize,
    pub max_claims: usize,
    pub max_deaths: usize,
    pub max_births: usize,
    pub max_delta_tiles: usize,
    pub max_delta_cells: usize,
    pub max_private_memory_bytes: usize,
}

impl Default for ReplayLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 16 * 1024 * 1024,
            max_commitments: 1_000_000,
            max_outcomes: 1_000_000,
            max_claims: 4_000_000,
            max_deaths: 1_000_000,
            max_births: 1_000_000,
            max_delta_tiles: 1_000_000,
            max_delta_cells: 1_000_000,
            max_private_memory_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    FrameTooLarge {
        actual: usize,
        limit: usize,
    },
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u16),
    InvalidTag {
        field: &'static str,
        tag: u8,
    },
    CountLimit {
        field: &'static str,
        actual: u64,
        limit: usize,
    },
    TrailingBytes(usize),
    EventHashMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    RulesetHashMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    UnverifiedBatch,
    PreviousEventHashMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    PreviousStateHashMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    SequenceMismatch {
        expected: u64,
        actual: u64,
    },
    NonMonotonicTime {
        previous: SimTime,
        current: SimTime,
    },
    NonCanonicalOrder(&'static str),
    InvalidBatch(&'static str),
    SequenceOverflow,
    ArchiveTooLarge {
        actual: usize,
        limit: usize,
    },
    InvalidArchiveMagic,
    ArchiveHashMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    InvalidArchiveIndex,
    EventOutOfRange {
        index: usize,
        count: usize,
    },
}

impl Display for ReplayError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrameTooLarge { actual, limit } => {
                write!(
                    formatter,
                    "replay frame is {actual} bytes; limit is {limit}"
                )
            }
            Self::Truncated => write!(formatter, "replay frame is truncated"),
            Self::InvalidMagic => write!(formatter, "invalid replay frame magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported replay format version {version}")
            }
            Self::InvalidTag { field, tag } => {
                write!(formatter, "invalid {field} tag {tag}")
            }
            Self::CountLimit {
                field,
                actual,
                limit,
            } => write!(
                formatter,
                "replay {field} count is {actual}; limit is {limit}"
            ),
            Self::TrailingBytes(count) => {
                write!(formatter, "replay frame has {count} trailing bytes")
            }
            Self::EventHashMismatch { expected, actual } => {
                write!(
                    formatter,
                    "event hash mismatch: expected {expected}, got {actual}"
                )
            }
            Self::RulesetHashMismatch { expected, actual } => {
                write!(
                    formatter,
                    "ruleset hash mismatch: expected {expected}, got {actual}"
                )
            }
            Self::UnverifiedBatch => {
                write!(
                    formatter,
                    "an unverified batch cannot be recorded in a replay"
                )
            }
            Self::PreviousEventHashMismatch { expected, actual } => write!(
                formatter,
                "previous event hash mismatch: expected {expected}, got {actual}"
            ),
            Self::PreviousStateHashMismatch { expected, actual } => write!(
                formatter,
                "previous state hash mismatch: expected {expected}, got {actual}"
            ),
            Self::SequenceMismatch { expected, actual } => {
                write!(
                    formatter,
                    "replay sequence mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NonMonotonicTime { previous, current } => write!(
                formatter,
                "replay completion time {current:?} does not follow {previous:?}"
            ),
            Self::NonCanonicalOrder(field) => {
                write!(formatter, "replay {field} are not in canonical order")
            }
            Self::InvalidBatch(message) => write!(formatter, "invalid replay batch: {message}"),
            Self::SequenceOverflow => write!(formatter, "replay sequence overflow"),
            Self::ArchiveTooLarge { actual, limit } => {
                write!(
                    formatter,
                    "replay archive is {actual} bytes; limit is {limit}"
                )
            }
            Self::InvalidArchiveMagic => write!(formatter, "invalid replay archive magic"),
            Self::ArchiveHashMismatch { expected, actual } => write!(
                formatter,
                "archive hash mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidArchiveIndex => write!(formatter, "invalid replay archive index"),
            Self::EventOutOfRange { index, count } => {
                write!(
                    formatter,
                    "replay event {index} is outside event count {count}"
                )
            }
        }
    }
}

impl Error for ReplayError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayBatchEvent {
    pub sequence: u64,
    pub compiled_ruleset_hash: CanonicalHash,
    pub previous_event_hash: CanonicalHash,
    pub previous_state_hash: CanonicalHash,
    pub pre_state_hash: CanonicalHash,
    pub post_state_hash: CanonicalHash,
    pub completed_at: SimTime,
    pub commitments: Vec<ReplayCommitment>,
    pub outcomes: Vec<ActionOutcome>,
    pub claims: Vec<ResourceClaim>,
    pub deaths: Vec<CellKey>,
    pub births: Vec<(CellKey, CellKey)>,
    pub delta: SimulationDelta,
}

/// An action installed between two resolution completions. Recording this at
/// the decision boundary is required for authoritative re-execution of actions
/// whose completion lies in a later event or segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCommitment {
    pub actor: CellKey,
    pub request: ActionRequest,
    pub signal: Option<ReferenceSignalEmission>,
    /// Explicit private-memory update returned by the actor's pristine Mind
    /// invocation. Retention is replayed against the actor's current state.
    pub memory_update: ReferenceMemoryUpdate,
    pub started_at: SimTime,
    pub receipt: CommitReceipt,
}

impl ReplayBatchEvent {
    pub fn event_hash(&self) -> CanonicalHash {
        hash_event_body(&self.encode_body())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.encode_body();
        bytes.extend_from_slice(self.event_hash().as_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        Self::from_bytes_with_limits(bytes, ReplayLimits::default())
    }

    pub fn from_bytes_with_limits(bytes: &[u8], limits: ReplayLimits) -> Result<Self, ReplayError> {
        if bytes.len() > limits.max_frame_bytes {
            return Err(ReplayError::FrameTooLarge {
                actual: bytes.len(),
                limit: limits.max_frame_bytes,
            });
        }
        if bytes.len() < 32 {
            return Err(ReplayError::Truncated);
        }
        let body_length = bytes.len() - 32;
        let (body, submitted_hash_bytes) = bytes.split_at(body_length);
        let submitted_hash = CanonicalHash::from_bytes(
            submitted_hash_bytes
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        );
        let actual_hash = hash_event_body(body);
        if submitted_hash != actual_hash {
            return Err(ReplayError::EventHashMismatch {
                expected: submitted_hash,
                actual: actual_hash,
            });
        }

        let mut reader = Reader::new(body);
        if reader.take(REPLAY_MAGIC.len())? != REPLAY_MAGIC {
            return Err(ReplayError::InvalidMagic);
        }
        let version = reader.u16()?;
        if version != REPLAY_FORMAT_VERSION {
            return Err(ReplayError::UnsupportedVersion(version));
        }
        let event = Self {
            sequence: reader.u64()?,
            compiled_ruleset_hash: reader.hash()?,
            previous_event_hash: reader.hash()?,
            previous_state_hash: reader.hash()?,
            pre_state_hash: reader.hash()?,
            post_state_hash: reader.hash()?,
            completed_at: SimTime(reader.u64()?),
            commitments: decode_vec(
                &mut reader,
                "commitments",
                limits.max_commitments,
                |reader| decode_commitment(reader, limits.max_private_memory_bytes),
            )?,
            outcomes: decode_vec(&mut reader, "outcomes", limits.max_outcomes, |reader| {
                decode_outcome(reader, limits.max_private_memory_bytes)
            })?,
            claims: decode_vec(&mut reader, "claims", limits.max_claims, decode_claim)?,
            deaths: decode_vec(&mut reader, "deaths", limits.max_deaths, |reader| {
                Ok(CellKey(reader.u64()?))
            })?,
            births: decode_vec(&mut reader, "births", limits.max_births, |reader| {
                Ok((CellKey(reader.u64()?), CellKey(reader.u64()?)))
            })?,
            delta: decode_simulation_delta(&mut reader, limits)?,
        };
        if reader.remaining() != 0 {
            return Err(ReplayError::TrailingBytes(reader.remaining()));
        }
        validate_event_shape(&event)?;
        Ok(event)
    }

    fn encode_body(&self) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.raw(REPLAY_MAGIC);
        writer.u16(REPLAY_FORMAT_VERSION);
        writer.u64(self.sequence);
        writer.hash(self.compiled_ruleset_hash);
        writer.hash(self.previous_event_hash);
        writer.hash(self.previous_state_hash);
        writer.hash(self.pre_state_hash);
        writer.hash(self.post_state_hash);
        writer.u64(self.completed_at.0);
        writer.vec(&self.commitments, encode_commitment);
        writer.vec(&self.outcomes, encode_outcome);
        writer.vec(&self.claims, encode_claim);
        writer.vec(&self.deaths, |writer, cell| writer.u64(cell.0));
        writer.vec(&self.births, |writer, (parent, child)| {
            writer.u64(parent.0);
            writer.u64(child.0);
        });
        encode_simulation_delta(&mut writer, &self.delta);
        writer.finish()
    }
}

/// Builds an integrity chain from authoritative resolver reports.
///
/// A chain proves ordering and detects mutation; it does not prove that the
/// submitted simulation was honest. Trust still comes from server re-execution
/// reproducing the recorded ruleset and state hashes.
#[derive(Debug, Clone)]
pub struct ReplayRecorder {
    compiled_ruleset_hash: CanonicalHash,
    next_sequence: u64,
    last_event_hash: CanonicalHash,
    last_state_hash: CanonicalHash,
    last_completion: Option<SimTime>,
}

impl ReplayRecorder {
    pub fn new(compiled_ruleset_hash: CanonicalHash, initial_state_hash: CanonicalHash) -> Self {
        Self {
            compiled_ruleset_hash,
            next_sequence: 0,
            last_event_hash: genesis_hash(compiled_ruleset_hash, initial_state_hash),
            last_state_hash: initial_state_hash,
            last_completion: None,
        }
    }

    /// Resumes recording from a cursor already authenticated by a segment or
    /// manifest verifier.
    pub fn from_cursor(compiled_ruleset_hash: CanonicalHash, cursor: ReplayChainCursor) -> Self {
        Self {
            compiled_ruleset_hash,
            next_sequence: cursor.next_sequence,
            last_event_hash: cursor.previous_event_hash,
            last_state_hash: cursor.previous_state_hash,
            last_completion: cursor.previous_completion,
        }
    }

    pub fn chain_head(&self) -> CanonicalHash {
        self.last_event_hash
    }

    pub fn last_state_hash(&self) -> CanonicalHash {
        self.last_state_hash
    }

    /// Returns the exact boundary required to continue the hash chain in a
    /// separately stored replay segment.
    pub fn cursor(&self) -> ReplayChainCursor {
        ReplayChainCursor {
            next_sequence: self.next_sequence,
            previous_event_hash: self.last_event_hash,
            previous_state_hash: self.last_state_hash,
            previous_completion: self.last_completion,
        }
    }

    pub fn from_events(
        events: &[ReplayBatchEvent],
        compiled_ruleset_hash: CanonicalHash,
        initial_state_hash: CanonicalHash,
        limits: ReplayLimits,
    ) -> Result<Self, ReplayError> {
        let frames: Vec<Vec<u8>> = events.iter().map(ReplayBatchEvent::to_bytes).collect();
        let verification = verify_replay(
            frames.iter().map(Vec::as_slice),
            compiled_ruleset_hash,
            initial_state_hash,
            limits,
        )?;
        Ok(Self {
            compiled_ruleset_hash,
            next_sequence: verification.event_count,
            last_event_hash: verification.final_event_hash,
            last_state_hash: verification.final_state_hash,
            last_completion: events.last().map(|event| event.completed_at),
        })
    }

    /// Records a structural completion event without decision commitments.
    ///
    /// Such events remain useful for delta inspection, but cannot start an
    /// authoritative replay from a checkpoint that predates the actions. Live
    /// engines should use [`Self::record_with_commitments`].
    pub fn record(&mut self, report: &BatchReport) -> Result<ReplayBatchEvent, ReplayError> {
        self.record_with_commitments(report, Vec::new())
    }

    pub fn record_with_commitments(
        &mut self,
        report: &BatchReport,
        commitments: Vec<ReplayCommitment>,
    ) -> Result<ReplayBatchEvent, ReplayError> {
        if report.compiled_ruleset_hash != self.compiled_ruleset_hash {
            return Err(ReplayError::RulesetHashMismatch {
                expected: self.compiled_ruleset_hash,
                actual: report.compiled_ruleset_hash,
            });
        }
        if let Some(previous) = self.last_completion {
            if report.completed_at <= previous {
                return Err(ReplayError::NonMonotonicTime {
                    previous,
                    current: report.completed_at,
                });
            }
        }
        let (Some(pre_state_hash), Some(post_state_hash)) =
            (report.pre_state_hash(), report.state_hash())
        else {
            return Err(ReplayError::UnverifiedBatch);
        };
        let event = ReplayBatchEvent {
            sequence: self.next_sequence,
            compiled_ruleset_hash: report.compiled_ruleset_hash,
            previous_event_hash: self.last_event_hash,
            previous_state_hash: self.last_state_hash,
            pre_state_hash,
            post_state_hash,
            completed_at: report.completed_at,
            commitments,
            outcomes: report.outcomes.clone(),
            claims: report.claims.clone(),
            deaths: report.deaths.clone(),
            births: report.births.clone(),
            delta: report.delta.clone(),
        };
        validate_event_shape(&event)?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(ReplayError::SequenceOverflow)?;
        self.last_event_hash = event.event_hash();
        self.last_state_hash = event.post_state_hash;
        self.last_completion = Some(event.completed_at);
        Ok(event)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayVerification {
    pub event_count: u64,
    pub final_event_hash: CanonicalHash,
    pub final_state_hash: CanonicalHash,
}

/// The authenticated boundary between independently stored replay segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayChainCursor {
    pub next_sequence: u64,
    pub previous_event_hash: CanonicalHash,
    pub previous_state_hash: CanonicalHash,
    pub previous_completion: Option<SimTime>,
}

impl ReplayChainCursor {
    pub fn genesis(
        compiled_ruleset_hash: CanonicalHash,
        initial_state_hash: CanonicalHash,
    ) -> Self {
        Self {
            next_sequence: 0,
            previous_event_hash: genesis_hash(compiled_ruleset_hash, initial_state_hash),
            previous_state_hash: initial_state_hash,
            previous_completion: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayArchiveLimits {
    pub max_archive_bytes: usize,
    pub max_events: usize,
    pub frame: ReplayLimits,
}

impl Default for ReplayArchiveLimits {
    fn default() -> Self {
        Self {
            max_archive_bytes: 512 * 1024 * 1024,
            max_events: 10_000_000,
            frame: ReplayLimits::default(),
        }
    }
}

/// A verified replay stream with a canonical O(1) event-offset index.
#[derive(Debug, Clone)]
pub struct ReplayArchive {
    bytes: Vec<u8>,
    compiled_ruleset_hash: CanonicalHash,
    initial_state_hash: CanonicalHash,
    event_index: Vec<(usize, usize)>,
    archive_hash: CanonicalHash,
}

impl ReplayArchive {
    pub fn from_events(
        events: &[ReplayBatchEvent],
        compiled_ruleset_hash: CanonicalHash,
        initial_state_hash: CanonicalHash,
        limits: ReplayArchiveLimits,
    ) -> Result<Self, ReplayError> {
        let frames: Vec<Vec<u8>> = events.iter().map(ReplayBatchEvent::to_bytes).collect();
        Self::from_frames(&frames, compiled_ruleset_hash, initial_state_hash, limits)
    }

    pub fn from_frames<B: AsRef<[u8]>>(
        frames: &[B],
        compiled_ruleset_hash: CanonicalHash,
        initial_state_hash: CanonicalHash,
        limits: ReplayArchiveLimits,
    ) -> Result<Self, ReplayError> {
        if frames.len() > limits.max_events {
            return Err(ReplayError::CountLimit {
                field: "archive events",
                actual: frames.len() as u64,
                limit: limits.max_events,
            });
        }
        verify_replay(
            frames.iter().map(AsRef::as_ref),
            compiled_ruleset_hash,
            initial_state_hash,
            limits.frame,
        )?;

        let mut payload_length = 0_usize;
        let mut relative_index = Vec::with_capacity(frames.len());
        for frame in frames {
            let length = frame.as_ref().len();
            relative_index.push((payload_length, length));
            payload_length =
                payload_length
                    .checked_add(length)
                    .ok_or(ReplayError::ArchiveTooLarge {
                        actual: usize::MAX,
                        limit: limits.max_archive_bytes,
                    })?;
        }

        let index_length = frames
            .len()
            .checked_mul(16)
            .ok_or(ReplayError::ArchiveTooLarge {
                actual: usize::MAX,
                limit: limits.max_archive_bytes,
            })?;
        let body_capacity = 8_usize
            .checked_add(2)
            .and_then(|value| value.checked_add(32 + 32 + 8))
            .and_then(|value| value.checked_add(index_length))
            .and_then(|value| value.checked_add(payload_length))
            .ok_or(ReplayError::ArchiveTooLarge {
                actual: usize::MAX,
                limit: limits.max_archive_bytes,
            })?;
        let total_length = body_capacity
            .checked_add(32)
            .ok_or(ReplayError::ArchiveTooLarge {
                actual: usize::MAX,
                limit: limits.max_archive_bytes,
            })?;
        if total_length > limits.max_archive_bytes {
            return Err(ReplayError::ArchiveTooLarge {
                actual: total_length,
                limit: limits.max_archive_bytes,
            });
        }

        let mut writer = Writer(Vec::with_capacity(body_capacity));
        writer.raw(ARCHIVE_MAGIC);
        writer.u16(REPLAY_FORMAT_VERSION);
        writer.hash(compiled_ruleset_hash);
        writer.hash(initial_state_hash);
        writer.u64(frames.len() as u64);
        for (offset, length) in &relative_index {
            writer.u64(*offset as u64);
            writer.u64(*length as u64);
        }
        for frame in frames {
            writer.raw(frame.as_ref());
        }
        let mut bytes = writer.finish();
        let archive_hash = hash_archive_body(&bytes);
        bytes.extend_from_slice(archive_hash.as_bytes());
        Self::from_bytes_with_limits(&bytes, limits)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        Self::from_bytes_with_limits(bytes, ReplayArchiveLimits::default())
    }

    pub fn from_bytes_with_limits(
        bytes: &[u8],
        limits: ReplayArchiveLimits,
    ) -> Result<Self, ReplayError> {
        if bytes.len() > limits.max_archive_bytes {
            return Err(ReplayError::ArchiveTooLarge {
                actual: bytes.len(),
                limit: limits.max_archive_bytes,
            });
        }
        if bytes.len() < 32 {
            return Err(ReplayError::Truncated);
        }
        let body_length = bytes.len() - 32;
        let (body, submitted_hash_bytes) = bytes.split_at(body_length);
        let submitted_hash = CanonicalHash::from_bytes(
            submitted_hash_bytes
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        );
        let actual_hash = hash_archive_body(body);
        if submitted_hash != actual_hash {
            return Err(ReplayError::ArchiveHashMismatch {
                expected: submitted_hash,
                actual: actual_hash,
            });
        }

        let mut reader = Reader::new(body);
        if reader.take(ARCHIVE_MAGIC.len())? != ARCHIVE_MAGIC {
            return Err(ReplayError::InvalidArchiveMagic);
        }
        let version = reader.u16()?;
        if version != REPLAY_FORMAT_VERSION {
            return Err(ReplayError::UnsupportedVersion(version));
        }
        let compiled_ruleset_hash = reader.hash()?;
        let initial_state_hash = reader.hash()?;
        let event_count = reader.u64()?;
        if event_count > limits.max_events as u64 {
            return Err(ReplayError::CountLimit {
                field: "archive events",
                actual: event_count,
                limit: limits.max_events,
            });
        }
        let event_count = usize::try_from(event_count).map_err(|_| ReplayError::CountLimit {
            field: "archive events",
            actual: event_count,
            limit: limits.max_events,
        })?;
        let mut relative_index = Vec::with_capacity(event_count.min(4096));
        let mut expected_offset = 0_usize;
        for _ in 0..event_count {
            let offset =
                usize::try_from(reader.u64()?).map_err(|_| ReplayError::InvalidArchiveIndex)?;
            let length =
                usize::try_from(reader.u64()?).map_err(|_| ReplayError::InvalidArchiveIndex)?;
            if offset != expected_offset || length > limits.frame.max_frame_bytes {
                return Err(ReplayError::InvalidArchiveIndex);
            }
            expected_offset = expected_offset
                .checked_add(length)
                .ok_or(ReplayError::InvalidArchiveIndex)?;
            relative_index.push((offset, length));
        }
        let payload_start = reader.position;
        if expected_offset != reader.remaining() {
            return Err(ReplayError::InvalidArchiveIndex);
        }
        let event_index: Vec<_> = relative_index
            .into_iter()
            .map(|(offset, length)| (payload_start + offset, length))
            .collect();
        let frames = event_index
            .iter()
            .map(|(offset, length)| &body[*offset..*offset + *length]);
        verify_replay(
            frames,
            compiled_ruleset_hash,
            initial_state_hash,
            limits.frame,
        )?;

        Ok(Self {
            bytes: bytes.to_vec(),
            compiled_ruleset_hash,
            initial_state_hash,
            event_index,
            archive_hash: submitted_hash,
        })
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn event_count(&self) -> usize {
        self.event_index.len()
    }

    pub fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.compiled_ruleset_hash
    }

    pub fn initial_state_hash(&self) -> CanonicalHash {
        self.initial_state_hash
    }

    pub fn archive_hash(&self) -> CanonicalHash {
        self.archive_hash
    }

    pub fn event_bytes(&self, index: usize) -> Result<&[u8], ReplayError> {
        let (offset, length) =
            *self
                .event_index
                .get(index)
                .ok_or(ReplayError::EventOutOfRange {
                    index,
                    count: self.event_index.len(),
                })?;
        Ok(&self.bytes[offset..offset + length])
    }

    pub fn event(
        &self,
        index: usize,
        limits: ReplayLimits,
    ) -> Result<ReplayBatchEvent, ReplayError> {
        ReplayBatchEvent::from_bytes_with_limits(self.event_bytes(index)?, limits)
    }
}

/// Performs bounded structural decoding and verifies the replay hash chain.
/// This is intentionally distinct from authoritative simulation re-execution.
pub fn verify_replay<I, B>(
    frames: I,
    compiled_ruleset_hash: CanonicalHash,
    initial_state_hash: CanonicalHash,
    limits: ReplayLimits,
) -> Result<ReplayVerification, ReplayError>
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
{
    let cursor = verify_replay_from_cursor(
        frames,
        compiled_ruleset_hash,
        ReplayChainCursor::genesis(compiled_ruleset_hash, initial_state_hash),
        limits,
    )?;

    Ok(ReplayVerification {
        event_count: cursor.next_sequence,
        final_event_hash: cursor.previous_event_hash,
        final_state_hash: cursor.previous_state_hash,
    })
}

/// Verifies a replay suffix from an authenticated segment boundary.
pub fn verify_replay_from_cursor<I, B>(
    frames: I,
    compiled_ruleset_hash: CanonicalHash,
    mut cursor: ReplayChainCursor,
    limits: ReplayLimits,
) -> Result<ReplayChainCursor, ReplayError>
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
{
    let mut expected_sequence = cursor.next_sequence;
    let mut previous_event_hash = cursor.previous_event_hash;
    let mut previous_state_hash = cursor.previous_state_hash;
    let mut previous_completion = cursor.previous_completion;

    for frame in frames {
        let event = ReplayBatchEvent::from_bytes_with_limits(frame.as_ref(), limits)?;
        if event.sequence != expected_sequence {
            return Err(ReplayError::SequenceMismatch {
                expected: expected_sequence,
                actual: event.sequence,
            });
        }
        if event.compiled_ruleset_hash != compiled_ruleset_hash {
            return Err(ReplayError::RulesetHashMismatch {
                expected: compiled_ruleset_hash,
                actual: event.compiled_ruleset_hash,
            });
        }
        if event.previous_event_hash != previous_event_hash {
            return Err(ReplayError::PreviousEventHashMismatch {
                expected: previous_event_hash,
                actual: event.previous_event_hash,
            });
        }
        if event.previous_state_hash != previous_state_hash {
            return Err(ReplayError::PreviousStateHashMismatch {
                expected: previous_state_hash,
                actual: event.previous_state_hash,
            });
        }
        if let Some(previous) = previous_completion {
            if event.completed_at <= previous {
                return Err(ReplayError::NonMonotonicTime {
                    previous,
                    current: event.completed_at,
                });
            }
        }
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or(ReplayError::SequenceOverflow)?;
        previous_event_hash = event.event_hash();
        previous_state_hash = event.post_state_hash;
        previous_completion = Some(event.completed_at);
    }

    cursor.next_sequence = expected_sequence;
    cursor.previous_event_hash = previous_event_hash;
    cursor.previous_state_hash = previous_state_hash;
    cursor.previous_completion = previous_completion;
    Ok(cursor)
}

pub fn genesis_hash(
    compiled_ruleset_hash: CanonicalHash,
    initial_state_hash: CanonicalHash,
) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((GENESIS_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(GENESIS_DOMAIN);
    hasher.update(REPLAY_FORMAT_VERSION.to_le_bytes());
    hasher.update(compiled_ruleset_hash.as_bytes());
    hasher.update(initial_state_hash.as_bytes());
    CanonicalHash::from_bytes(hasher.finalize().into())
}

fn hash_event_body(body: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((EVENT_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(EVENT_DOMAIN);
    hasher.update(REPLAY_FORMAT_VERSION.to_le_bytes());
    hasher.update(body);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

fn hash_archive_body(body: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((ARCHIVE_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(ARCHIVE_DOMAIN);
    hasher.update(REPLAY_FORMAT_VERSION.to_le_bytes());
    hasher.update(body);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

fn validate_event_shape(event: &ReplayBatchEvent) -> Result<(), ReplayError> {
    if event.commitments.iter().any(|commitment| {
        commitment.receipt.actor != commitment.actor
            || commitment.receipt.action != commitment.request.kind()
            || commitment.receipt.accepted != commitment.receipt.rejection.is_none()
            || commitment.receipt.completes_at <= commitment.started_at
            || commitment.started_at >= event.completed_at
    }) {
        return Err(ReplayError::InvalidBatch(
            "commitment receipt or timing metadata is inconsistent",
        ));
    }
    if !is_sorted_by(&event.commitments, |left, right| {
        (left.started_at, left.actor) < (right.started_at, right.actor)
    }) {
        return Err(ReplayError::NonCanonicalOrder("commitments"));
    }
    if event.outcomes.iter().any(|outcome| {
        outcome.completed_at != event.completed_at
            || outcome.action != outcome.request.kind()
            || !outcome_effects_are_consistent(outcome)
    }) {
        return Err(ReplayError::InvalidBatch(
            "outcome action, effects, or completion metadata is inconsistent",
        ));
    }
    if !is_sorted_by(&event.outcomes, |left, right| {
        (left.actor, left.started_at, left.action) < (right.actor, right.started_at, right.action)
    }) {
        return Err(ReplayError::NonCanonicalOrder("outcomes"));
    }
    if !is_sorted_by(&event.claims, |left, right| {
        (left.key, left.mode, left.actor) < (right.key, right.mode, right.actor)
    }) {
        return Err(ReplayError::NonCanonicalOrder("claims"));
    }
    if !event.deaths.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(ReplayError::NonCanonicalOrder("deaths"));
    }
    if !event.births.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(ReplayError::NonCanonicalOrder("births"));
    }
    if event.delta.before_time != event.completed_at || event.delta.after_time != event.completed_at
    {
        return Err(ReplayError::InvalidBatch(
            "delta time does not match batch completion time",
        ));
    }
    if !event
        .delta
        .tiles
        .windows(2)
        .all(|pair| pair[0].tile < pair[1].tile)
    {
        return Err(ReplayError::NonCanonicalOrder("delta tiles"));
    }
    if event
        .delta
        .tiles
        .iter()
        .any(|delta| delta.before == delta.after)
    {
        return Err(ReplayError::InvalidBatch(
            "delta contains an unchanged tile",
        ));
    }
    if !event
        .delta
        .cells
        .windows(2)
        .all(|pair| pair[0].cell < pair[1].cell)
    {
        return Err(ReplayError::NonCanonicalOrder("delta cells"));
    }
    if event
        .delta
        .cells
        .iter()
        .any(|delta| delta.before == delta.after)
    {
        return Err(ReplayError::InvalidBatch(
            "delta contains an unchanged cell",
        ));
    }
    Ok(())
}

fn outcome_effects_are_consistent(outcome: &ActionOutcome) -> bool {
    let damage_consistent = match (&outcome.request, outcome.status, &outcome.attack_damage) {
        (ActionRequest::Attack { .. }, OutcomeStatus::Success, Some(damage)) => {
            damage.victim != outcome.actor
                && (!damage.target_was_guarded || damage.mitigated <= damage.raw)
                && (damage.target_was_guarded || damage.mitigated == 0)
                && damage
                    .mitigated
                    .checked_add(damage.applied)
                    .and_then(|value| value.checked_add(damage.overkill))
                    == Some(damage.raw)
        }
        (ActionRequest::Attack { .. }, OutcomeStatus::Success, None) => false,
        (ActionRequest::Attack { .. }, _, None) => true,
        (ActionRequest::Attack { .. }, _, Some(_)) | (_, _, Some(_)) => false,
        (_, _, None) => true,
    };
    if !damage_consistent {
        return false;
    }

    match (&outcome.request, outcome.status, &outcome.terrain_change) {
        (ActionRequest::Excavate, OutcomeStatus::Success, Some(change)) => {
            change.tile == outcome.origin
                && change.material_mass > 0
                && change.elevation_before.checked_sub(1) == Some(change.elevation_after)
        }
        (ActionRequest::DepositTerrain, OutcomeStatus::Success, Some(change)) => {
            change.tile == outcome.origin
                && change.material_mass > 0
                && change.elevation_before.checked_add(1) == Some(change.elevation_after)
        }
        (ActionRequest::Excavate | ActionRequest::DepositTerrain, OutcomeStatus::Success, None) => {
            false
        }
        (ActionRequest::Excavate | ActionRequest::DepositTerrain, _, None) => true,
        (ActionRequest::Excavate | ActionRequest::DepositTerrain, _, Some(_)) | (_, _, Some(_)) => {
            false
        }
        (_, _, None) => true,
    }
}

fn is_sorted_by<T>(values: &[T], ordered: impl Fn(&T, &T) -> bool) -> bool {
    values.windows(2).all(|pair| ordered(&pair[0], &pair[1]))
}

pub(super) struct Writer(Vec<u8>);

impl Writer {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }

    pub(super) fn finish(self) -> Vec<u8> {
        self.0
    }

    pub(super) fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    pub(super) fn u8(&mut self, value: u8) {
        self.raw(&[value]);
    }

    pub(super) fn i8(&mut self, value: i8) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn i16(&mut self, value: i16) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn hash(&mut self, hash: CanonicalHash) {
        self.raw(hash.as_bytes());
    }

    pub(super) fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.raw(bytes);
    }

    pub(super) fn vec<T>(&mut self, values: &[T], encode: impl Fn(&mut Self, &T)) {
        self.u64(values.len() as u64);
        for value in values {
            encode(self, value);
        }
    }
}

pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(super) fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    pub(super) fn position(&self) -> usize {
        self.position
    }

    pub(super) fn take(&mut self, count: usize) -> Result<&'a [u8], ReplayError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(ReplayError::Truncated)?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(ReplayError::Truncated)?;
        self.position = end;
        Ok(value)
    }

    pub(super) fn u8(&mut self) -> Result<u8, ReplayError> {
        Ok(self.take(1)?[0])
    }

    pub(super) fn i8(&mut self) -> Result<i8, ReplayError> {
        Ok(i8::from_le_bytes(
            self.take(1)?
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        ))
    }

    pub(super) fn u16(&mut self) -> Result<u16, ReplayError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        ))
    }

    pub(super) fn i16(&mut self) -> Result<i16, ReplayError> {
        Ok(i16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        ))
    }

    pub(super) fn u32(&mut self) -> Result<u32, ReplayError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        ))
    }

    pub(super) fn u64(&mut self) -> Result<u64, ReplayError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        ))
    }

    pub(super) fn hash(&mut self) -> Result<CanonicalHash, ReplayError> {
        Ok(CanonicalHash::from_bytes(
            self.take(32)?
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        ))
    }

    pub(super) fn bytes(
        &mut self,
        field: &'static str,
        limit: usize,
    ) -> Result<Vec<u8>, ReplayError> {
        let length = self.u64()?;
        if length > limit as u64 {
            return Err(ReplayError::CountLimit {
                field,
                actual: length,
                limit,
            });
        }
        let length = usize::try_from(length).map_err(|_| ReplayError::CountLimit {
            field,
            actual: length,
            limit,
        })?;
        Ok(self.take(length)?.to_vec())
    }
}

pub(super) fn decode_vec<T>(
    reader: &mut Reader<'_>,
    field: &'static str,
    limit: usize,
    mut decode: impl FnMut(&mut Reader<'_>) -> Result<T, ReplayError>,
) -> Result<Vec<T>, ReplayError> {
    let count = reader.u64()?;
    if count > limit as u64 {
        return Err(ReplayError::CountLimit {
            field,
            actual: count,
            limit,
        });
    }
    let count = usize::try_from(count).map_err(|_| ReplayError::CountLimit {
        field,
        actual: count,
        limit,
    })?;
    // Do not reserve from an untrusted count before proving the frame contains
    // the corresponding entries. This keeps tiny truncated frames cheap.
    let mut values = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        values.push(decode(reader)?);
    }
    Ok(values)
}

fn encode_simulation_delta(writer: &mut Writer, delta: &SimulationDelta) {
    writer.u64(delta.before_time.0);
    writer.u64(delta.after_time.0);
    writer.u64(delta.before_next_cell_key);
    writer.u64(delta.after_next_cell_key);
    writer.vec(&delta.tiles, |writer, delta| {
        writer.u64(delta.tile.0 as u64);
        encode_tile_state(writer, &delta.before);
        encode_tile_state(writer, &delta.after);
    });
    writer.vec(&delta.cells, |writer, delta| {
        writer.u64(delta.cell.0);
        encode_cell_state_option(writer, delta.before.as_ref());
        encode_cell_state_option(writer, delta.after.as_ref());
    });
}

fn decode_simulation_delta(
    reader: &mut Reader<'_>,
    limits: ReplayLimits,
) -> Result<SimulationDelta, ReplayError> {
    Ok(SimulationDelta {
        before_time: SimTime(reader.u64()?),
        after_time: SimTime(reader.u64()?),
        before_next_cell_key: reader.u64()?,
        after_next_cell_key: reader.u64()?,
        tiles: decode_vec(reader, "delta tiles", limits.max_delta_tiles, |reader| {
            Ok(TileDelta {
                tile: TileIndex(read_usize(reader, "tile index")?),
                before: decode_tile_state(reader)?,
                after: decode_tile_state(reader)?,
            })
        })?,
        cells: decode_vec(reader, "delta cells", limits.max_delta_cells, |reader| {
            Ok(CellDelta {
                cell: CellKey(reader.u64()?),
                before: decode_cell_state_option(reader, limits.max_private_memory_bytes)?,
                after: decode_cell_state_option(reader, limits.max_private_memory_bytes)?,
            })
        })?,
    })
}

pub(super) fn encode_tile_state(writer: &mut Writer, tile: &TileState) {
    writer.raw(&tile.elevation.to_le_bytes());
    encode_cell_key_option(writer, tile.occupant);
    writer.u64(tile.plant_energy);
    writer.u64(tile.plant_capacity);
    writer.u64(tile.plant_growth_rate);
    writer.u64(tile.plant_growth_remainder);
    writer.u64(tile.loose_energy);
    writer.u64(tile.diffuse_energy);
    writer.u64(tile.diffusion_remainder);
    for energy in tile.signal_energy {
        writer.u64(energy);
    }
    for remainder in tile.signal_decay_remainder {
        writer.u64(remainder);
    }
}

pub(super) fn decode_tile_state(reader: &mut Reader<'_>) -> Result<TileState, ReplayError> {
    Ok(TileState {
        elevation: i16::from_le_bytes(
            reader
                .take(2)?
                .try_into()
                .map_err(|_| ReplayError::Truncated)?,
        ),
        occupant: decode_cell_key_option(reader)?,
        plant_energy: reader.u64()?,
        plant_capacity: reader.u64()?,
        plant_growth_rate: reader.u64()?,
        plant_growth_remainder: reader.u64()?,
        loose_energy: reader.u64()?,
        diffuse_energy: reader.u64()?,
        diffusion_remainder: reader.u64()?,
        signal_energy: decode_u64_array(reader)?,
        signal_decay_remainder: decode_u64_array(reader)?,
    })
}

fn decode_u64_array<const N: usize>(reader: &mut Reader<'_>) -> Result<[u64; N], ReplayError> {
    let mut values = [0; N];
    for value in &mut values {
        *value = reader.u64()?;
    }
    Ok(values)
}

fn encode_cell_state_option(writer: &mut Writer, cell: Option<&CellState>) {
    match cell {
        Some(cell) => {
            writer.u8(1);
            encode_cell_state(writer, cell);
        }
        None => writer.u8(0),
    }
}

fn decode_cell_state_option(
    reader: &mut Reader<'_>,
    private_memory_limit: usize,
) -> Result<Option<CellState>, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(decode_cell_state(reader, private_memory_limit)?)),
        _ => Err(ReplayError::InvalidTag {
            field: "cell state option",
            tag,
        }),
    }
}

pub(super) fn encode_cell_state(writer: &mut Writer, cell: &CellState) {
    writer.u64(cell.position.0 as u64);
    writer.u64(cell.core_mass);
    writer.u64(cell.assimilated_energy);
    writer.u64(cell.gut_energy);
    writer.u64(cell.digestion_remainder);
    writer.u64(cell.metabolism_remainder);
    writer.u64(cell.carried_material_mass);
    writer.u32(cell.marker);
    writer.u8(u8::from(cell.guarded));
    writer.bytes(&cell.private_memory);
    writer.u64(cell.ready_at.0);
    match &cell.pending_action {
        Some(pending) => {
            writer.u8(1);
            encode_pending_action(writer, pending);
        }
        None => writer.u8(0),
    }
    encode_outcome_status_option(writer, cell.last_outcome);
}

pub(super) fn decode_cell_state(
    reader: &mut Reader<'_>,
    private_memory_limit: usize,
) -> Result<CellState, ReplayError> {
    let position = TileIndex(read_usize(reader, "tile index")?);
    let core_mass = reader.u64()?;
    let assimilated_energy = reader.u64()?;
    let gut_energy = reader.u64()?;
    let digestion_remainder = reader.u64()?;
    let metabolism_remainder = reader.u64()?;
    let carried_material_mass = reader.u64()?;
    let marker = reader.u32()?;
    let guarded = decode_bool(reader, "guarded")?;
    let private_memory = reader.bytes("private memory", private_memory_limit)?;
    let ready_at = SimTime(reader.u64()?);
    let pending_action = match reader.u8()? {
        0 => None,
        1 => Some(decode_pending_action(reader, private_memory_limit)?),
        tag => {
            return Err(ReplayError::InvalidTag {
                field: "pending action option",
                tag,
            });
        }
    };
    let last_outcome = decode_outcome_status_option(reader)?;
    Ok(CellState {
        position,
        core_mass,
        assimilated_energy,
        gut_energy,
        digestion_remainder,
        metabolism_remainder,
        carried_material_mass,
        marker,
        guarded,
        ready_at,
        pending_action: pending_action.map(Arc::new),
        last_outcome,
        cold: Arc::new(CellColdState {
            private_memory: private_memory.into(),
        }),
    })
}

fn encode_pending_action(writer: &mut Writer, pending: &PendingAction) {
    encode_action_request(writer, &pending.request);
    writer.u64(pending.origin.0 as u64);
    encode_tile_option(writer, pending.target);
    writer.u64(pending.started_at.0);
    writer.u64(pending.completes_at.0);
    writer.u64(pending.effort_spent);
    writer.u64(pending.payload_escrow);
    encode_reject_reason_option(writer, pending.rejection);
}

fn decode_pending_action(
    reader: &mut Reader<'_>,
    private_memory_limit: usize,
) -> Result<PendingAction, ReplayError> {
    Ok(PendingAction {
        request: decode_action_request(reader, private_memory_limit)?,
        origin: TileIndex(read_usize(reader, "tile index")?),
        target: decode_tile_option(reader)?,
        started_at: SimTime(reader.u64()?),
        completes_at: SimTime(reader.u64()?),
        effort_spent: reader.u64()?,
        payload_escrow: reader.u64()?,
        rejection: decode_reject_reason_option(reader)?,
    })
}

fn encode_cell_key_option(writer: &mut Writer, cell: Option<CellKey>) {
    match cell {
        Some(cell) => {
            writer.u8(1);
            writer.u64(cell.0);
        }
        None => writer.u8(0),
    }
}

fn decode_cell_key_option(reader: &mut Reader<'_>) -> Result<Option<CellKey>, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(CellKey(reader.u64()?))),
        _ => Err(ReplayError::InvalidTag {
            field: "cell key option",
            tag,
        }),
    }
}

fn encode_outcome_status_option(writer: &mut Writer, status: Option<OutcomeStatus>) {
    match status {
        Some(status) => {
            writer.u8(1);
            encode_outcome_status(writer, status);
        }
        None => writer.u8(0),
    }
}

fn decode_outcome_status_option(
    reader: &mut Reader<'_>,
) -> Result<Option<OutcomeStatus>, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(decode_outcome_status(reader)?)),
        _ => Err(ReplayError::InvalidTag {
            field: "outcome option",
            tag,
        }),
    }
}

fn encode_reject_reason_option(writer: &mut Writer, reason: Option<RejectReason>) {
    match reason {
        Some(reason) => {
            writer.u8(1);
            encode_reject_reason(writer, reason);
        }
        None => writer.u8(0),
    }
}

fn decode_reject_reason_option(
    reader: &mut Reader<'_>,
) -> Result<Option<RejectReason>, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(decode_reject_reason(reader)?)),
        _ => Err(ReplayError::InvalidTag {
            field: "rejection option",
            tag,
        }),
    }
}

fn decode_bool(reader: &mut Reader<'_>, field: &'static str) -> Result<bool, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(ReplayError::InvalidTag { field, tag }),
    }
}

fn encode_commitment(writer: &mut Writer, commitment: &ReplayCommitment) {
    writer.u64(commitment.actor.0);
    encode_action_request(writer, &commitment.request);
    match commitment.signal {
        Some(signal) => {
            writer.u8(1);
            writer.u8(signal.channel);
            writer.u64(signal.amount);
        }
        None => writer.u8(0),
    }
    match &commitment.memory_update {
        ReferenceMemoryUpdate::Retain => writer.u8(0),
        ReferenceMemoryUpdate::Replace(bytes) => {
            writer.u8(1);
            writer.bytes(bytes);
        }
    }
    writer.u64(commitment.started_at.0);
    writer.u64(commitment.receipt.completes_at.0);
    writer.u64(commitment.receipt.effort_spent);
    writer.u64(commitment.receipt.payload_escrow);
    encode_reject_reason_option(writer, commitment.receipt.rejection);
}

fn decode_commitment(
    reader: &mut Reader<'_>,
    private_memory_limit: usize,
) -> Result<ReplayCommitment, ReplayError> {
    let actor = CellKey(reader.u64()?);
    let request = decode_action_request(reader, private_memory_limit)?;
    let signal = match reader.u8()? {
        0 => None,
        1 => Some(ReferenceSignalEmission {
            channel: reader.u8()?,
            amount: reader.u64()?,
        }),
        tag => {
            return Err(ReplayError::InvalidTag {
                field: "signal option",
                tag,
            });
        }
    };
    let memory_update = match reader.u8()? {
        0 => ReferenceMemoryUpdate::Retain,
        1 => ReferenceMemoryUpdate::Replace(
            reader.bytes("next private memory", private_memory_limit)?,
        ),
        tag => {
            return Err(ReplayError::InvalidTag {
                field: "private memory update",
                tag,
            });
        }
    };
    let started_at = SimTime(reader.u64()?);
    let completes_at = SimTime(reader.u64()?);
    let effort_spent = reader.u64()?;
    let payload_escrow = reader.u64()?;
    let rejection = decode_reject_reason_option(reader)?;
    Ok(ReplayCommitment {
        actor,
        request: request.clone(),
        signal,
        memory_update,
        started_at,
        receipt: CommitReceipt {
            actor,
            action: request.kind(),
            accepted: rejection.is_none(),
            rejection,
            completes_at,
            effort_spent,
            payload_escrow,
        },
    })
}

fn encode_outcome(writer: &mut Writer, outcome: &ActionOutcome) {
    writer.u64(outcome.actor.0);
    encode_action_request(writer, &outcome.request);
    writer.u64(outcome.origin.0 as u64);
    encode_outcome_status(writer, outcome.status);
    writer.u64(outcome.started_at.0);
    writer.u64(outcome.completed_at.0);
    encode_tile_option(writer, outcome.target);
    writer.u64(outcome.effort_spent);
    writer.u64(outcome.payload);
    encode_attack_damage_option(writer, outcome.attack_damage.as_ref());
    encode_terrain_change_option(writer, outcome.terrain_change.as_ref());
}

fn decode_outcome(
    reader: &mut Reader<'_>,
    private_memory_limit: usize,
) -> Result<ActionOutcome, ReplayError> {
    let actor = CellKey(reader.u64()?);
    let request = decode_action_request(reader, private_memory_limit)?;
    Ok(ActionOutcome {
        actor,
        action: request.kind(),
        request,
        origin: TileIndex(read_usize(reader, "tile index")?),
        status: decode_outcome_status(reader)?,
        started_at: SimTime(reader.u64()?),
        completed_at: SimTime(reader.u64()?),
        target: decode_tile_option(reader)?,
        effort_spent: reader.u64()?,
        payload: reader.u64()?,
        attack_damage: decode_attack_damage_option(reader)?,
        terrain_change: decode_terrain_change_option(reader)?,
    })
}

fn encode_attack_damage_option(writer: &mut Writer, damage: Option<&AttackDamage>) {
    let Some(damage) = damage else {
        writer.u8(0);
        return;
    };
    writer.u8(1);
    writer.u64(damage.victim.0);
    writer.u8(u8::from(damage.target_was_guarded));
    writer.u64(damage.raw);
    writer.u64(damage.mitigated);
    writer.u64(damage.applied);
    writer.u64(damage.overkill);
}

fn decode_attack_damage_option(
    reader: &mut Reader<'_>,
) -> Result<Option<AttackDamage>, ReplayError> {
    match reader.u8()? {
        0 => Ok(None),
        1 => Ok(Some(AttackDamage {
            victim: CellKey(reader.u64()?),
            target_was_guarded: decode_bool(reader, "attack target guarded")?,
            raw: reader.u64()?,
            mitigated: reader.u64()?,
            applied: reader.u64()?,
            overkill: reader.u64()?,
        })),
        tag => Err(ReplayError::InvalidTag {
            field: "attack damage option",
            tag,
        }),
    }
}

fn encode_terrain_change_option(writer: &mut Writer, change: Option<&TerrainChange>) {
    let Some(change) = change else {
        writer.u8(0);
        return;
    };
    writer.u8(1);
    writer.u64(change.tile.0 as u64);
    writer.i16(change.elevation_before);
    writer.i16(change.elevation_after);
    writer.u64(change.material_mass);
}

fn decode_terrain_change_option(
    reader: &mut Reader<'_>,
) -> Result<Option<TerrainChange>, ReplayError> {
    match reader.u8()? {
        0 => Ok(None),
        1 => Ok(Some(TerrainChange {
            tile: TileIndex(read_usize(reader, "terrain change tile")?),
            elevation_before: reader.i16()?,
            elevation_after: reader.i16()?,
            material_mass: reader.u64()?,
        })),
        tag => Err(ReplayError::InvalidTag {
            field: "terrain change option",
            tag,
        }),
    }
}

fn encode_action_request(writer: &mut Writer, request: &ActionRequest) {
    match request {
        ActionRequest::Wait => writer.u8(0),
        ActionRequest::Move { target, effort } => {
            writer.u8(1);
            writer.u8(target.0);
            encode_effort(writer, *effort);
        }
        ActionRequest::Attack {
            target,
            effort,
            payload,
        } => {
            writer.u8(2);
            writer.u8(target.0);
            encode_effort(writer, *effort);
            writer.u64(*payload);
        }
        ActionRequest::Guard { effort } => {
            writer.u8(3);
            encode_effort(writer, *effort);
        }
        ActionRequest::Consume { amount } => {
            writer.u8(4);
            writer.u64(*amount);
        }
        ActionRequest::Split {
            target,
            child_allocation,
            marker,
            private_memory,
        } => {
            writer.u8(5);
            writer.u8(target.0);
            writer.u64(*child_allocation);
            writer.u32(*marker);
            writer.bytes(private_memory);
        }
        ActionRequest::Regurgitate { target, amount } => {
            writer.u8(6);
            writer.u8(target.0);
            writer.u64(*amount);
        }
        ActionRequest::Signal { amounts } => {
            writer.u8(9);
            for amount in amounts {
                writer.u64(*amount);
            }
        }
        ActionRequest::Excavate => writer.u8(7),
        ActionRequest::DepositTerrain => writer.u8(8),
    }
}

fn decode_action_request(
    reader: &mut Reader<'_>,
    private_memory_limit: usize,
) -> Result<ActionRequest, ReplayError> {
    let tag = reader.u8()?;
    Ok(match tag {
        0 => ActionRequest::Wait,
        1 => ActionRequest::Move {
            target: LocalSlot(reader.u8()?),
            effort: decode_effort(reader)?,
        },
        2 => ActionRequest::Attack {
            target: LocalSlot(reader.u8()?),
            effort: decode_effort(reader)?,
            payload: reader.u64()?,
        },
        3 => ActionRequest::Guard {
            effort: decode_effort(reader)?,
        },
        4 => ActionRequest::Consume {
            amount: reader.u64()?,
        },
        5 => ActionRequest::Split {
            target: LocalSlot(reader.u8()?),
            child_allocation: reader.u64()?,
            marker: reader.u32()?,
            private_memory: reader.bytes("private memory", private_memory_limit)?,
        },
        6 => ActionRequest::Regurgitate {
            target: LocalSlot(reader.u8()?),
            amount: reader.u64()?,
        },
        7 => ActionRequest::Excavate,
        8 => ActionRequest::DepositTerrain,
        9 => ActionRequest::Signal {
            amounts: [reader.u64()?, reader.u64()?, reader.u64()?, reader.u64()?],
        },
        _ => {
            return Err(ReplayError::InvalidTag {
                field: "action",
                tag,
            });
        }
    })
}

fn encode_effort(writer: &mut Writer, effort: EffortTier) {
    writer.u8(match effort {
        EffortTier::Low => 0,
        EffortTier::Standard => 1,
        EffortTier::High => 2,
    });
}

fn decode_effort(reader: &mut Reader<'_>) -> Result<EffortTier, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(EffortTier::Low),
        1 => Ok(EffortTier::Standard),
        2 => Ok(EffortTier::High),
        _ => Err(ReplayError::InvalidTag {
            field: "effort tier",
            tag,
        }),
    }
}

fn encode_outcome_status(writer: &mut Writer, status: OutcomeStatus) {
    match status {
        OutcomeStatus::Rejected(reason) => {
            writer.u8(0);
            encode_reject_reason(writer, reason);
        }
        OutcomeStatus::Success => writer.u8(1),
        OutcomeStatus::Frustrated => writer.u8(2),
        OutcomeStatus::Contested => writer.u8(3),
        OutcomeStatus::Interrupted => writer.u8(4),
    }
}

fn decode_outcome_status(reader: &mut Reader<'_>) -> Result<OutcomeStatus, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(OutcomeStatus::Rejected(decode_reject_reason(reader)?)),
        1 => Ok(OutcomeStatus::Success),
        2 => Ok(OutcomeStatus::Frustrated),
        3 => Ok(OutcomeStatus::Contested),
        4 => Ok(OutcomeStatus::Interrupted),
        _ => Err(ReplayError::InvalidTag {
            field: "outcome",
            tag,
        }),
    }
}

fn encode_reject_reason(writer: &mut Writer, reason: RejectReason) {
    writer.u8(match reason {
        RejectReason::InvalidSlot => 0,
        RejectReason::ActionNotAllowedInSlot => 1,
        RejectReason::TargetOutsideWorld => 2,
        RejectReason::TargetsSelf => 3,
        RejectReason::ZeroPayload => 4,
        RejectReason::InsufficientGutEnergy => 5,
        RejectReason::ChildAllocationTooSmall => 6,
        RejectReason::PrivateMemoryTooLarge => 7,
        RejectReason::InsufficientEnergy => 8,
        RejectReason::InsufficientMaterial => 9,
        RejectReason::TerrainLimit => 10,
        RejectReason::ArithmeticOverflow => 11,
    });
}

fn decode_reject_reason(reader: &mut Reader<'_>) -> Result<RejectReason, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(RejectReason::InvalidSlot),
        1 => Ok(RejectReason::ActionNotAllowedInSlot),
        2 => Ok(RejectReason::TargetOutsideWorld),
        3 => Ok(RejectReason::TargetsSelf),
        4 => Ok(RejectReason::ZeroPayload),
        5 => Ok(RejectReason::InsufficientGutEnergy),
        6 => Ok(RejectReason::ChildAllocationTooSmall),
        7 => Ok(RejectReason::PrivateMemoryTooLarge),
        8 => Ok(RejectReason::InsufficientEnergy),
        9 => Ok(RejectReason::InsufficientMaterial),
        10 => Ok(RejectReason::TerrainLimit),
        11 => Ok(RejectReason::ArithmeticOverflow),
        _ => Err(ReplayError::InvalidTag {
            field: "rejection reason",
            tag,
        }),
    }
}

fn encode_claim(writer: &mut Writer, claim: &ResourceClaim) {
    writer.u64(claim.actor.0);
    match claim.key {
        ResourceKey::Occupancy(tile) => {
            writer.u8(0);
            writer.u64(tile.0 as u64);
        }
        ResourceKey::CellState(cell) => {
            writer.u8(1);
            writer.u64(cell.0);
        }
        ResourceKey::PlantEnergy(tile) => {
            writer.u8(2);
            writer.u64(tile.0 as u64);
        }
        ResourceKey::LooseEnergy(tile) => {
            writer.u8(3);
            writer.u64(tile.0 as u64);
        }
        ResourceKey::DiffuseEnergy(tile) => {
            writer.u8(4);
            writer.u64(tile.0 as u64);
        }
        ResourceKey::Terrain(tile) => {
            writer.u8(5);
            writer.u64(tile.0 as u64);
        }
        ResourceKey::Signal(tile) => {
            writer.u8(6);
            writer.u64(tile.0 as u64);
        }
    }
    writer.u8(match claim.mode {
        AccessMode::ReadSnapshot => 0,
        AccessMode::Add => 1,
        AccessMode::BoundedTake => 2,
        AccessMode::Exclusive => 3,
    });
}

fn decode_claim(reader: &mut Reader<'_>) -> Result<ResourceClaim, ReplayError> {
    let actor = CellKey(reader.u64()?);
    let tag = reader.u8()?;
    let value = reader.u64()?;
    let key = match tag {
        0 => ResourceKey::Occupancy(TileIndex(u64_to_usize(value, "tile index")?)),
        1 => ResourceKey::CellState(CellKey(value)),
        2 => ResourceKey::PlantEnergy(TileIndex(u64_to_usize(value, "tile index")?)),
        3 => ResourceKey::LooseEnergy(TileIndex(u64_to_usize(value, "tile index")?)),
        4 => ResourceKey::DiffuseEnergy(TileIndex(u64_to_usize(value, "tile index")?)),
        5 => ResourceKey::Terrain(TileIndex(u64_to_usize(value, "tile index")?)),
        6 => ResourceKey::Signal(TileIndex(u64_to_usize(value, "tile index")?)),
        _ => {
            return Err(ReplayError::InvalidTag {
                field: "resource key",
                tag,
            });
        }
    };
    let tag = reader.u8()?;
    let mode = match tag {
        0 => AccessMode::ReadSnapshot,
        1 => AccessMode::Add,
        2 => AccessMode::BoundedTake,
        3 => AccessMode::Exclusive,
        _ => {
            return Err(ReplayError::InvalidTag {
                field: "access mode",
                tag,
            });
        }
    };
    Ok(ResourceClaim { actor, key, mode })
}

fn encode_tile_option(writer: &mut Writer, tile: Option<TileIndex>) {
    match tile {
        Some(tile) => {
            writer.u8(1);
            writer.u64(tile.0 as u64);
        }
        None => writer.u8(0),
    }
}

fn decode_tile_option(reader: &mut Reader<'_>) -> Result<Option<TileIndex>, ReplayError> {
    let tag = reader.u8()?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(TileIndex(read_usize(reader, "tile index")?))),
        _ => Err(ReplayError::InvalidTag {
            field: "tile option",
            tag,
        }),
    }
}

fn read_usize(reader: &mut Reader<'_>, field: &'static str) -> Result<usize, ReplayError> {
    u64_to_usize(reader.u64()?, field)
}

fn u64_to_usize(value: u64, field: &'static str) -> Result<usize, ReplayError> {
    if value > MAX_CANONICAL_TILE_INDEX {
        return Err(ReplayError::CountLimit {
            field,
            actual: value,
            limit: u32::MAX as usize,
        });
    }
    usize::try_from(value).map_err(|_| ReplayError::CountLimit {
        field,
        actual: value,
        limit: usize::MAX,
    })
}
