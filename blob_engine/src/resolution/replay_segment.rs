//! Content-addressed replay segments and their canonical chain manifest.

use std::error::Error;
use std::fmt::{Display, Formatter};

use sha2::{Digest, Sha256};

use super::checkpoint::{CheckpointError, CheckpointLimits, ReferenceCheckpoint};
use super::hashing::CanonicalHash;
use super::reference::SimTime;
use super::replay::{
    verify_replay_from_cursor, Reader, ReplayBatchEvent, ReplayChainCursor, ReplayError,
    ReplayLimits, Writer,
};

pub const REPLAY_SEGMENT_FORMAT_VERSION: u16 = 1;
pub const REPLAY_MANIFEST_FORMAT_VERSION: u16 = 1;
const SEGMENT_MAGIC: &[u8; 8] = b"BLBSEG01";
const MANIFEST_MAGIC: &[u8; 8] = b"BLBMAN01";
const SEGMENT_DOMAIN: &[u8] = b"blob.replay.segment";
const MANIFEST_DOMAIN: &[u8] = b"blob.replay.manifest";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplaySegmentLimits {
    pub max_segment_bytes: usize,
    pub max_events: usize,
    pub frame: ReplayLimits,
    pub checkpoint: CheckpointLimits,
}

impl Default for ReplaySegmentLimits {
    fn default() -> Self {
        Self {
            max_segment_bytes: 96 * 1024 * 1024,
            max_events: 65_536,
            frame: ReplayLimits::default(),
            checkpoint: CheckpointLimits::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayManifestLimits {
    pub max_manifest_bytes: usize,
    pub max_segments: usize,
}

impl Default for ReplayManifestLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 16 * 1024 * 1024,
            max_segments: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayStreamError {
    TooLarge {
        kind: &'static str,
        actual: usize,
        limit: usize,
    },
    Truncated,
    InvalidMagic(&'static str),
    UnsupportedVersion {
        kind: &'static str,
        version: u16,
    },
    IntegrityHashMismatch {
        kind: &'static str,
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    CountLimit {
        kind: &'static str,
        actual: u64,
        limit: usize,
    },
    EmptySegment,
    InvalidSegmentIndex,
    InvalidCompletionTag(u8),
    CheckpointRulesetMismatch,
    CheckpointStateMismatch,
    ManifestChain(&'static str),
    SegmentDescriptorMismatch,
    EventOutOfRange {
        sequence: u64,
        event_count: u64,
    },
    TrailingBytes(usize),
    Replay(ReplayError),
    Checkpoint(CheckpointError),
}

impl Display for ReplayStreamError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge {
                kind,
                actual,
                limit,
            } => write!(formatter, "{kind} is {actual} bytes; limit is {limit}"),
            Self::Truncated => write!(formatter, "replay stream object is truncated"),
            Self::InvalidMagic(kind) => write!(formatter, "invalid {kind} magic"),
            Self::UnsupportedVersion { kind, version } => {
                write!(formatter, "unsupported {kind} version {version}")
            }
            Self::IntegrityHashMismatch {
                kind,
                expected,
                actual,
            } => write!(
                formatter,
                "{kind} integrity hash mismatch: expected {expected}, got {actual}"
            ),
            Self::CountLimit {
                kind,
                actual,
                limit,
            } => write!(formatter, "{kind} count is {actual}; limit is {limit}"),
            Self::EmptySegment => write!(formatter, "replay segments must contain an event"),
            Self::InvalidSegmentIndex => write!(formatter, "invalid replay segment index"),
            Self::InvalidCompletionTag(tag) => {
                write!(formatter, "invalid replay completion tag {tag}")
            }
            Self::CheckpointRulesetMismatch => {
                write!(
                    formatter,
                    "segment checkpoint has the wrong compiled ruleset"
                )
            }
            Self::CheckpointStateMismatch => {
                write!(
                    formatter,
                    "segment checkpoint does not match its chain boundary"
                )
            }
            Self::ManifestChain(message) => write!(formatter, "invalid replay manifest: {message}"),
            Self::SegmentDescriptorMismatch => {
                write!(formatter, "segment does not match its manifest descriptor")
            }
            Self::EventOutOfRange {
                sequence,
                event_count,
            } => write!(
                formatter,
                "replay event {sequence} is outside manifest event count {event_count}"
            ),
            Self::TrailingBytes(count) => {
                write!(formatter, "replay stream object has {count} trailing bytes")
            }
            Self::Replay(error) => Display::fmt(error, formatter),
            Self::Checkpoint(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for ReplayStreamError {}

impl From<ReplayError> for ReplayStreamError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<CheckpointError> for ReplayStreamError {
    fn from(value: CheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}

/// A self-contained replay suffix rooted at an authoritative checkpoint.
#[derive(Debug, Clone)]
pub struct ReplaySegment {
    bytes: Vec<u8>,
    compiled_ruleset_hash: CanonicalHash,
    start_cursor: ReplayChainCursor,
    end_cursor: ReplayChainCursor,
    checkpoint: ReferenceCheckpoint,
    event_index: Vec<(usize, usize)>,
    segment_hash: CanonicalHash,
}

impl ReplaySegment {
    pub fn from_events(
        compiled_ruleset_hash: CanonicalHash,
        start_cursor: ReplayChainCursor,
        checkpoint: ReferenceCheckpoint,
        events: &[ReplayBatchEvent],
        limits: ReplaySegmentLimits,
    ) -> Result<Self, ReplayStreamError> {
        if events.is_empty() {
            return Err(ReplayStreamError::EmptySegment);
        }
        if events.len() > limits.max_events {
            return Err(ReplayStreamError::CountLimit {
                kind: "replay segment events",
                actual: events.len() as u64,
                limit: limits.max_events,
            });
        }
        validate_checkpoint(compiled_ruleset_hash, start_cursor, &checkpoint)?;

        let frames: Vec<Vec<u8>> = events.iter().map(ReplayBatchEvent::to_bytes).collect();
        verify_replay_from_cursor(
            frames.iter().map(Vec::as_slice),
            compiled_ruleset_hash,
            start_cursor,
            limits.frame,
        )?;

        let mut payload_length = 0_usize;
        let mut relative_index = Vec::with_capacity(frames.len());
        for frame in &frames {
            relative_index.push((payload_length, frame.len()));
            payload_length =
                payload_length
                    .checked_add(frame.len())
                    .ok_or(ReplayStreamError::TooLarge {
                        kind: "replay segment",
                        actual: usize::MAX,
                        limit: limits.max_segment_bytes,
                    })?;
        }

        let mut writer = Writer::new();
        writer.raw(SEGMENT_MAGIC);
        writer.u16(REPLAY_SEGMENT_FORMAT_VERSION);
        writer.hash(compiled_ruleset_hash);
        encode_cursor(&mut writer, start_cursor);
        writer.bytes(&checkpoint.to_bytes());
        writer.u64(frames.len() as u64);
        for (offset, length) in relative_index {
            writer.u64(offset as u64);
            writer.u64(length as u64);
        }
        for frame in frames {
            writer.raw(&frame);
        }
        let mut bytes = writer.finish();
        let total = bytes
            .len()
            .checked_add(32)
            .ok_or(ReplayStreamError::TooLarge {
                kind: "replay segment",
                actual: usize::MAX,
                limit: limits.max_segment_bytes,
            })?;
        if total > limits.max_segment_bytes {
            return Err(ReplayStreamError::TooLarge {
                kind: "replay segment",
                actual: total,
                limit: limits.max_segment_bytes,
            });
        }
        let hash = hash_body(SEGMENT_DOMAIN, REPLAY_SEGMENT_FORMAT_VERSION, &bytes);
        bytes.extend_from_slice(hash.as_bytes());
        Self::from_bytes_with_limits(&bytes, limits)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayStreamError> {
        Self::from_bytes_with_limits(bytes, ReplaySegmentLimits::default())
    }

    pub fn from_bytes_with_limits(
        bytes: &[u8],
        limits: ReplaySegmentLimits,
    ) -> Result<Self, ReplayStreamError> {
        let (body, segment_hash) = checked_body(
            bytes,
            limits.max_segment_bytes,
            "replay segment",
            SEGMENT_DOMAIN,
            REPLAY_SEGMENT_FORMAT_VERSION,
        )?;
        let mut reader = Reader::new(body);
        if reader.take(SEGMENT_MAGIC.len())? != SEGMENT_MAGIC {
            return Err(ReplayStreamError::InvalidMagic("replay segment"));
        }
        let version = reader.u16()?;
        if version != REPLAY_SEGMENT_FORMAT_VERSION {
            return Err(ReplayStreamError::UnsupportedVersion {
                kind: "replay segment",
                version,
            });
        }
        let compiled_ruleset_hash = reader.hash()?;
        let start_cursor = decode_cursor(&mut reader)?;
        let checkpoint_bytes =
            reader.bytes("segment checkpoint", limits.checkpoint.max_checkpoint_bytes)?;
        let checkpoint =
            ReferenceCheckpoint::from_bytes_with_limits(&checkpoint_bytes, limits.checkpoint)?;
        validate_checkpoint(compiled_ruleset_hash, start_cursor, &checkpoint)?;

        let count = reader.u64()?;
        if count == 0 {
            return Err(ReplayStreamError::EmptySegment);
        }
        if count > limits.max_events as u64 {
            return Err(ReplayStreamError::CountLimit {
                kind: "replay segment events",
                actual: count,
                limit: limits.max_events,
            });
        }
        let count = usize::try_from(count).map_err(|_| ReplayStreamError::CountLimit {
            kind: "replay segment events",
            actual: count,
            limit: limits.max_events,
        })?;
        let mut relative_index = Vec::with_capacity(count.min(4096));
        let mut expected_offset = 0_usize;
        for _ in 0..count {
            let offset = usize::try_from(reader.u64()?)
                .map_err(|_| ReplayStreamError::InvalidSegmentIndex)?;
            let length = usize::try_from(reader.u64()?)
                .map_err(|_| ReplayStreamError::InvalidSegmentIndex)?;
            if offset != expected_offset || length > limits.frame.max_frame_bytes {
                return Err(ReplayStreamError::InvalidSegmentIndex);
            }
            expected_offset = expected_offset
                .checked_add(length)
                .ok_or(ReplayStreamError::InvalidSegmentIndex)?;
            relative_index.push((offset, length));
        }
        let payload_start = reader.position();
        if expected_offset != reader.remaining() {
            return Err(ReplayStreamError::InvalidSegmentIndex);
        }
        let event_index: Vec<_> = relative_index
            .into_iter()
            .map(|(offset, length)| (payload_start + offset, length))
            .collect();
        let frames = event_index
            .iter()
            .map(|(offset, length)| &body[*offset..*offset + *length]);
        let end_cursor =
            verify_replay_from_cursor(frames, compiled_ruleset_hash, start_cursor, limits.frame)?;

        Ok(Self {
            bytes: bytes.to_vec(),
            compiled_ruleset_hash,
            start_cursor,
            end_cursor,
            checkpoint,
            event_index,
            segment_hash,
        })
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.compiled_ruleset_hash
    }

    pub const fn start_cursor(&self) -> ReplayChainCursor {
        self.start_cursor
    }

    pub const fn end_cursor(&self) -> ReplayChainCursor {
        self.end_cursor
    }

    pub fn checkpoint(&self) -> &ReferenceCheckpoint {
        &self.checkpoint
    }

    pub fn event_count(&self) -> usize {
        self.event_index.len()
    }

    pub const fn segment_hash(&self) -> CanonicalHash {
        self.segment_hash
    }

    pub fn event_bytes(&self, index: usize) -> Result<&[u8], ReplayStreamError> {
        let (offset, length) =
            *self
                .event_index
                .get(index)
                .ok_or(ReplayStreamError::EventOutOfRange {
                    sequence: self.start_cursor.next_sequence.saturating_add(index as u64),
                    event_count: self.end_cursor.next_sequence,
                })?;
        Ok(&self.bytes[offset..offset + length])
    }

    pub fn event(
        &self,
        index: usize,
        limits: ReplayLimits,
    ) -> Result<ReplayBatchEvent, ReplayStreamError> {
        Ok(ReplayBatchEvent::from_bytes_with_limits(
            self.event_bytes(index)?,
            limits,
        )?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplaySegmentDescriptor {
    pub start_cursor: ReplayChainCursor,
    pub end_cursor: ReplayChainCursor,
    pub checkpoint_hash: CanonicalHash,
    pub segment_hash: CanonicalHash,
    pub byte_length: u64,
}

impl ReplaySegmentDescriptor {
    pub fn from_segment(segment: &ReplaySegment) -> Self {
        Self {
            start_cursor: segment.start_cursor,
            end_cursor: segment.end_cursor,
            checkpoint_hash: segment.checkpoint.checkpoint_hash(),
            segment_hash: segment.segment_hash,
            byte_length: segment.bytes.len() as u64,
        }
    }

    pub fn event_count(&self) -> u64 {
        self.end_cursor
            .next_sequence
            .saturating_sub(self.start_cursor.next_sequence)
    }
}

/// A compact content-addressed index. Segment bytes can live in object storage
/// and be fetched only when a browser or verifier needs them.
#[derive(Debug, Clone)]
pub struct ReplayManifest {
    bytes: Vec<u8>,
    compiled_ruleset_hash: CanonicalHash,
    initial_state_hash: CanonicalHash,
    segments: Vec<ReplaySegmentDescriptor>,
    manifest_hash: CanonicalHash,
}

impl ReplayManifest {
    pub fn from_segments(
        compiled_ruleset_hash: CanonicalHash,
        initial_state_hash: CanonicalHash,
        segments: Vec<ReplaySegmentDescriptor>,
        limits: ReplayManifestLimits,
    ) -> Result<Self, ReplayStreamError> {
        validate_manifest(
            compiled_ruleset_hash,
            initial_state_hash,
            &segments,
            limits.max_segments,
        )?;
        let mut writer = Writer::new();
        writer.raw(MANIFEST_MAGIC);
        writer.u16(REPLAY_MANIFEST_FORMAT_VERSION);
        writer.hash(compiled_ruleset_hash);
        writer.hash(initial_state_hash);
        writer.u64(segments.len() as u64);
        for segment in &segments {
            encode_cursor(&mut writer, segment.start_cursor);
            encode_cursor(&mut writer, segment.end_cursor);
            writer.hash(segment.checkpoint_hash);
            writer.hash(segment.segment_hash);
            writer.u64(segment.byte_length);
        }
        let mut bytes = writer.finish();
        let total = bytes
            .len()
            .checked_add(32)
            .ok_or(ReplayStreamError::TooLarge {
                kind: "replay manifest",
                actual: usize::MAX,
                limit: limits.max_manifest_bytes,
            })?;
        if total > limits.max_manifest_bytes {
            return Err(ReplayStreamError::TooLarge {
                kind: "replay manifest",
                actual: total,
                limit: limits.max_manifest_bytes,
            });
        }
        let manifest_hash = hash_body(MANIFEST_DOMAIN, REPLAY_MANIFEST_FORMAT_VERSION, &bytes);
        bytes.extend_from_slice(manifest_hash.as_bytes());
        Ok(Self {
            bytes,
            compiled_ruleset_hash,
            initial_state_hash,
            segments,
            manifest_hash,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayStreamError> {
        Self::from_bytes_with_limits(bytes, ReplayManifestLimits::default())
    }

    pub fn from_bytes_with_limits(
        bytes: &[u8],
        limits: ReplayManifestLimits,
    ) -> Result<Self, ReplayStreamError> {
        let (body, manifest_hash) = checked_body(
            bytes,
            limits.max_manifest_bytes,
            "replay manifest",
            MANIFEST_DOMAIN,
            REPLAY_MANIFEST_FORMAT_VERSION,
        )?;
        let mut reader = Reader::new(body);
        if reader.take(MANIFEST_MAGIC.len())? != MANIFEST_MAGIC {
            return Err(ReplayStreamError::InvalidMagic("replay manifest"));
        }
        let version = reader.u16()?;
        if version != REPLAY_MANIFEST_FORMAT_VERSION {
            return Err(ReplayStreamError::UnsupportedVersion {
                kind: "replay manifest",
                version,
            });
        }
        let compiled_ruleset_hash = reader.hash()?;
        let initial_state_hash = reader.hash()?;
        let count = reader.u64()?;
        if count > limits.max_segments as u64 {
            return Err(ReplayStreamError::CountLimit {
                kind: "replay manifest segments",
                actual: count,
                limit: limits.max_segments,
            });
        }
        let count = usize::try_from(count).map_err(|_| ReplayStreamError::CountLimit {
            kind: "replay manifest segments",
            actual: count,
            limit: limits.max_segments,
        })?;
        let mut segments = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            segments.push(ReplaySegmentDescriptor {
                start_cursor: decode_cursor(&mut reader)?,
                end_cursor: decode_cursor(&mut reader)?,
                checkpoint_hash: reader.hash()?,
                segment_hash: reader.hash()?,
                byte_length: reader.u64()?,
            });
        }
        if reader.remaining() != 0 {
            return Err(ReplayStreamError::TrailingBytes(reader.remaining()));
        }
        validate_manifest(
            compiled_ruleset_hash,
            initial_state_hash,
            &segments,
            limits.max_segments,
        )?;
        Ok(Self {
            bytes: bytes.to_vec(),
            compiled_ruleset_hash,
            initial_state_hash,
            segments,
            manifest_hash,
        })
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.compiled_ruleset_hash
    }

    pub const fn initial_state_hash(&self) -> CanonicalHash {
        self.initial_state_hash
    }

    pub const fn manifest_hash(&self) -> CanonicalHash {
        self.manifest_hash
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn segment(&self, index: usize) -> Option<&ReplaySegmentDescriptor> {
        self.segments.get(index)
    }

    pub fn segments(&self) -> impl Iterator<Item = &ReplaySegmentDescriptor> {
        self.segments.iter()
    }

    pub fn event_count(&self) -> u64 {
        self.segments
            .last()
            .map_or(0, |segment| segment.end_cursor.next_sequence)
    }

    pub fn final_cursor(&self) -> ReplayChainCursor {
        self.segments.last().map_or_else(
            || ReplayChainCursor::genesis(self.compiled_ruleset_hash, self.initial_state_hash),
            |segment| segment.end_cursor,
        )
    }

    pub fn verify_segment(
        &self,
        index: usize,
        segment: &ReplaySegment,
    ) -> Result<(), ReplayStreamError> {
        let expected = self
            .segments
            .get(index)
            .ok_or(ReplayStreamError::InvalidSegmentIndex)?;
        if segment.compiled_ruleset_hash != self.compiled_ruleset_hash
            || ReplaySegmentDescriptor::from_segment(segment) != *expected
        {
            return Err(ReplayStreamError::SegmentDescriptorMismatch);
        }
        Ok(())
    }

    /// Returns the segment containing a global zero-based event sequence.
    pub fn segment_for_event(&self, sequence: u64) -> Result<usize, ReplayStreamError> {
        let event_count = self.event_count();
        if sequence >= event_count {
            return Err(ReplayStreamError::EventOutOfRange {
                sequence,
                event_count,
            });
        }
        Ok(self
            .segments
            .partition_point(|segment| segment.end_cursor.next_sequence <= sequence))
    }
}

fn validate_checkpoint(
    compiled_ruleset_hash: CanonicalHash,
    cursor: ReplayChainCursor,
    checkpoint: &ReferenceCheckpoint,
) -> Result<(), ReplayStreamError> {
    if checkpoint.compiled_ruleset_hash() != compiled_ruleset_hash {
        return Err(ReplayStreamError::CheckpointRulesetMismatch);
    }
    if checkpoint.state_hash() != cursor.previous_state_hash {
        return Err(ReplayStreamError::CheckpointStateMismatch);
    }
    Ok(())
}

fn validate_manifest(
    compiled_ruleset_hash: CanonicalHash,
    initial_state_hash: CanonicalHash,
    segments: &[ReplaySegmentDescriptor],
    maximum: usize,
) -> Result<(), ReplayStreamError> {
    if segments.len() > maximum {
        return Err(ReplayStreamError::CountLimit {
            kind: "replay manifest segments",
            actual: segments.len() as u64,
            limit: maximum,
        });
    }
    let mut expected = ReplayChainCursor::genesis(compiled_ruleset_hash, initial_state_hash);
    for segment in segments {
        if segment.start_cursor != expected {
            return Err(ReplayStreamError::ManifestChain(
                "segment cursor does not continue the preceding segment",
            ));
        }
        if segment.end_cursor.next_sequence <= segment.start_cursor.next_sequence {
            return Err(ReplayStreamError::ManifestChain(
                "segments must contain at least one event",
            ));
        }
        let Some(end_completion) = segment.end_cursor.previous_completion else {
            return Err(ReplayStreamError::ManifestChain(
                "non-empty segment has no completion time",
            ));
        };
        if segment
            .start_cursor
            .previous_completion
            .is_some_and(|start| end_completion <= start)
        {
            return Err(ReplayStreamError::ManifestChain(
                "segment completion time does not advance",
            ));
        }
        if segment.byte_length == 0 {
            return Err(ReplayStreamError::ManifestChain(
                "segment byte length must be positive",
            ));
        }
        expected = segment.end_cursor;
    }
    Ok(())
}

fn encode_cursor(writer: &mut Writer, cursor: ReplayChainCursor) {
    writer.u64(cursor.next_sequence);
    writer.hash(cursor.previous_event_hash);
    writer.hash(cursor.previous_state_hash);
    match cursor.previous_completion {
        None => writer.u8(0),
        Some(completion) => {
            writer.u8(1);
            writer.u64(completion.0);
        }
    }
}

fn decode_cursor(reader: &mut Reader<'_>) -> Result<ReplayChainCursor, ReplayStreamError> {
    let next_sequence = reader.u64()?;
    let previous_event_hash = reader.hash()?;
    let previous_state_hash = reader.hash()?;
    let previous_completion = match reader.u8()? {
        0 => None,
        1 => Some(SimTime(reader.u64()?)),
        tag => return Err(ReplayStreamError::InvalidCompletionTag(tag)),
    };
    Ok(ReplayChainCursor {
        next_sequence,
        previous_event_hash,
        previous_state_hash,
        previous_completion,
    })
}

fn checked_body<'a>(
    bytes: &'a [u8],
    limit: usize,
    kind: &'static str,
    domain: &[u8],
    version: u16,
) -> Result<(&'a [u8], CanonicalHash), ReplayStreamError> {
    if bytes.len() > limit {
        return Err(ReplayStreamError::TooLarge {
            kind,
            actual: bytes.len(),
            limit,
        });
    }
    if bytes.len() < 32 {
        return Err(ReplayStreamError::Truncated);
    }
    let body_length = bytes.len() - 32;
    let (body, encoded_hash) = bytes.split_at(body_length);
    let expected = CanonicalHash::from_bytes(
        encoded_hash
            .try_into()
            .map_err(|_| ReplayStreamError::Truncated)?,
    );
    let actual = hash_body(domain, version, body);
    if actual != expected {
        return Err(ReplayStreamError::IntegrityHashMismatch {
            kind,
            expected,
            actual,
        });
    }
    Ok((body, expected))
}

fn hash_body(domain: &[u8], version: u16, body: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update(version.to_le_bytes());
    hasher.update((body.len() as u64).to_le_bytes());
    hasher.update(body);
    CanonicalHash::from_bytes(hasher.finalize().into())
}
