//! Authoritative replay by deterministic re-execution from canonical checkpoints.

use std::error::Error;
use std::fmt::{Display, Formatter};

use super::{
    verify_replay_from_cursor, BatchReport, CanonicalHash, CellKey, CommitReceipt,
    ReferenceCheckpoint, ReferenceSimulation, ReplayBatchEvent, ReplayChainCursor, ReplayError,
    ReplayLimits, ReplaySegment, ReplayStreamError, ResolutionError,
};

#[derive(Debug)]
pub enum ReplayDriverError {
    Replay(ReplayError),
    ReplayStream(ReplayStreamError),
    Resolution(ResolutionError),
    Checkpoint(String),
    RulesetMismatch,
    SegmentStartMismatch,
    SegmentEndMismatch,
    StateMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    CommitmentTimeMismatch {
        actor: CellKey,
    },
    CommitmentFailed {
        actor: CellKey,
        message: String,
    },
    ReceiptMismatch {
        actor: CellKey,
        expected: CommitReceipt,
        actual: CommitReceipt,
    },
    ReportMismatch(&'static str),
    PrefixOutOfRange {
        requested: usize,
        available: usize,
    },
}

impl Display for ReplayDriverError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Replay(error) => Display::fmt(error, formatter),
            Self::ReplayStream(error) => Display::fmt(error, formatter),
            Self::Resolution(error) => Display::fmt(error, formatter),
            Self::Checkpoint(message) => write!(formatter, "invalid replay checkpoint: {message}"),
            Self::RulesetMismatch => write!(formatter, "replay driver ruleset mismatch"),
            Self::SegmentStartMismatch => {
                write!(formatter, "replay segment does not start at the driver cursor")
            }
            Self::SegmentEndMismatch => {
                write!(formatter, "replay segment does not end at the computed cursor")
            }
            Self::StateMismatch { expected, actual } => write!(
                formatter,
                "replay driver state mismatch: expected {expected}, got {actual}"
            ),
            Self::CommitmentTimeMismatch { actor } => write!(
                formatter,
                "replay commitment for {actor:?} was not made at the current decision boundary"
            ),
            Self::CommitmentFailed { actor, message } => {
                write!(formatter, "replay commitment for {actor:?} failed: {message}")
            }
            Self::ReceiptMismatch {
                actor,
                expected,
                actual,
            } => write!(
                formatter,
                "replay commitment receipt mismatch for {actor:?}: expected {expected:?}, got {actual:?}"
            ),
            Self::ReportMismatch(field) => {
                write!(formatter, "re-executed replay {field} does not match the recording")
            }
            Self::PrefixOutOfRange {
                requested,
                available,
            } => write!(
                formatter,
                "replay prefix requests {requested} events; segment contains {available}"
            ),
        }
    }
}

impl Error for ReplayDriverError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Replay(error) => Some(error),
            Self::ReplayStream(error) => Some(error),
            Self::Resolution(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ReplayError> for ReplayDriverError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<ReplayStreamError> for ReplayDriverError {
    fn from(value: ReplayStreamError) -> Self {
        Self::ReplayStream(value)
    }
}

impl From<ResolutionError> for ReplayDriverError {
    fn from(value: ResolutionError) -> Self {
        Self::Resolution(value)
    }
}

/// Deterministic replay state shared by native verification and browser WASM.
#[derive(Debug, Clone)]
pub struct ReferenceReplayDriver {
    simulation: ReferenceSimulation,
    compiled_ruleset_hash: CanonicalHash,
    cursor: ReplayChainCursor,
}

impl ReferenceReplayDriver {
    pub fn from_segment(segment: &ReplaySegment) -> Result<Self, ReplayDriverError> {
        Self::from_checkpoint(segment.checkpoint().clone(), segment.start_cursor())
    }

    pub fn from_checkpoint(
        checkpoint: ReferenceCheckpoint,
        cursor: ReplayChainCursor,
    ) -> Result<Self, ReplayDriverError> {
        if checkpoint.state_hash() != cursor.previous_state_hash {
            return Err(ReplayDriverError::StateMismatch {
                expected: cursor.previous_state_hash,
                actual: checkpoint.state_hash(),
            });
        }
        let compiled_ruleset_hash = checkpoint.compiled_ruleset_hash();
        let simulation = checkpoint
            .into_simulation()
            .map_err(|error| ReplayDriverError::Checkpoint(error.to_string()))?;
        if cursor
            .previous_completion
            .is_some_and(|completion| completion != simulation.now())
        {
            return Err(ReplayDriverError::SegmentStartMismatch);
        }
        Ok(Self {
            simulation,
            compiled_ruleset_hash,
            cursor,
        })
    }

    pub fn simulation(&self) -> &ReferenceSimulation {
        &self.simulation
    }

    pub const fn cursor(&self) -> ReplayChainCursor {
        self.cursor
    }

    pub fn apply_event(
        &mut self,
        event: &ReplayBatchEvent,
        limits: ReplayLimits,
    ) -> Result<BatchReport, ReplayDriverError> {
        if event.compiled_ruleset_hash != self.compiled_ruleset_hash {
            return Err(ReplayDriverError::RulesetMismatch);
        }
        let encoded = event.to_bytes();
        let verified_cursor = verify_replay_from_cursor(
            [encoded.as_slice()],
            self.compiled_ruleset_hash,
            self.cursor,
            limits,
        )?;
        let actual_state = self.simulation.state_hash();
        if actual_state != self.cursor.previous_state_hash {
            return Err(ReplayDriverError::StateMismatch {
                expected: self.cursor.previous_state_hash,
                actual: actual_state,
            });
        }

        for commitment in &event.commitments {
            if commitment.started_at != self.simulation.now() {
                return Err(ReplayDriverError::CommitmentTimeMismatch {
                    actor: commitment.actor,
                });
            }
            let actual = self
                .simulation
                .commit_memory_update_with_signal(
                    commitment.actor,
                    commitment.request.clone(),
                    commitment.signal,
                    commitment.memory_update.clone(),
                )
                .map_err(|error| ReplayDriverError::CommitmentFailed {
                    actor: commitment.actor,
                    message: error.to_string(),
                })?;
            if actual != commitment.receipt {
                return Err(ReplayDriverError::ReceiptMismatch {
                    actor: commitment.actor,
                    expected: commitment.receipt.clone(),
                    actual,
                });
            }
        }

        let report = self.simulation.resolve_next_batch()?;
        compare_report(&report, event)?;
        self.cursor = verified_cursor;
        Ok(report)
    }

    pub fn apply_segment(
        &mut self,
        segment: &ReplaySegment,
        limits: ReplayLimits,
    ) -> Result<usize, ReplayDriverError> {
        self.apply_segment_prefix(segment, segment.event_count(), limits)
    }

    pub fn apply_segment_prefix(
        &mut self,
        segment: &ReplaySegment,
        event_count: usize,
        limits: ReplayLimits,
    ) -> Result<usize, ReplayDriverError> {
        if event_count > segment.event_count() {
            return Err(ReplayDriverError::PrefixOutOfRange {
                requested: event_count,
                available: segment.event_count(),
            });
        }
        if segment.compiled_ruleset_hash() != self.compiled_ruleset_hash {
            return Err(ReplayDriverError::RulesetMismatch);
        }
        if segment.start_cursor() != self.cursor
            || segment.checkpoint().state_hash() != self.simulation.state_hash()
        {
            return Err(ReplayDriverError::SegmentStartMismatch);
        }
        for index in 0..event_count {
            let event = segment.event(index, limits)?;
            self.apply_event(&event, limits)?;
        }
        if event_count == segment.event_count() && self.cursor != segment.end_cursor() {
            return Err(ReplayDriverError::SegmentEndMismatch);
        }
        Ok(event_count)
    }
}

fn compare_report(
    actual: &BatchReport,
    expected: &ReplayBatchEvent,
) -> Result<(), ReplayDriverError> {
    let matches = [
        (
            actual.completed_at == expected.completed_at,
            "completion time",
        ),
        (actual.outcomes == expected.outcomes, "outcomes"),
        (actual.claims == expected.claims, "claims"),
        (actual.deaths == expected.deaths, "deaths"),
        (actual.births == expected.births, "births"),
        (
            actual.compiled_ruleset_hash == expected.compiled_ruleset_hash,
            "compiled ruleset hash",
        ),
        (
            actual.pre_state_hash() == Some(expected.pre_state_hash),
            "pre-state hash",
        ),
        (
            actual.state_hash() == Some(expected.post_state_hash),
            "post-state hash",
        ),
        (actual.delta == expected.delta, "state delta"),
    ];
    for (matches, field) in matches {
        if !matches {
            return Err(ReplayDriverError::ReportMismatch(field));
        }
    }
    Ok(())
}
