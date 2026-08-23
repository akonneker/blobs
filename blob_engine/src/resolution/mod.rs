//! Deterministic reference action resolution.
//!
//! The serial path remains the semantic oracle. Native hosts may use the
//! ordered parallel claim/component path; browser WASM uses the same canonical
//! phases serially.

mod attestation;
mod checkpoint;
mod delta;
mod hashing;
mod match_manifest;
mod neighborhood;
mod reference;
mod replay;
mod replay_bundle;
mod replay_driver;
mod replay_segment;

pub use attestation::{
    attestation_key_id, attestation_signing_message, MatchAttestation, MatchAttestationError,
    ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES, MATCH_ATTESTATION_FORMAT_VERSION,
};
pub use checkpoint::{
    CheckpointError, CheckpointLimits, ReferenceCheckpoint, CHECKPOINT_FORMAT_VERSION,
};
pub use delta::{CellDelta, DeltaError, SimulationDelta, TileDelta};
pub use hashing::{
    CanonicalHash, CANONICAL_HASH_ALGORITHM, CANONICAL_HASH_FORMAT_VERSION,
    REFERENCE_SEMANTIC_KERNEL_VERSION,
};
pub use match_manifest::{
    mind_artifact_hash, MatchVerificationError, MatchVerificationLimits, MatchVerificationManifest,
    MindArtifactBinding, MindRuntimeProfile, MATCH_VERIFICATION_FORMAT_VERSION,
};
pub use neighborhood::{
    BoundaryRule, CompiledNeighborhood, DiagonalCornerRule, LocalOffset, LocalSlot,
    NeighborhoodError, NeighborhoodSpec, ObservationMasks, SlotMask, TargetingAction, TileIndex,
    MAX_LOCAL_SLOTS,
};
pub use reference::{
    AccessMode, ActionKind, ActionOutcome, ActionRequest, ActivityCue, BatchIntegrity, BatchReport,
    CellColdState, CellKey, CellState, CellStore, CommitError, CommitReceipt, DecisionCommitment,
    DurationRule, EffortProfile, EffortTier, IntegrityMode, NeighborCue, OutcomeStatus,
    PendingAction, ProgressBucket, ReferenceObservationBatch, ReferenceRuleset,
    ReferenceSimulation, RejectReason, ResolutionError, ResolutionMetrics, ResolutionPhaseTimings,
    ResourceClaim, ResourceKey, SimTime, SimulationState, TileState, TimeConfig, SIGNAL_CHANNELS,
};
pub use replay::{
    genesis_hash, verify_replay, verify_replay_from_cursor, ReplayArchive, ReplayArchiveLimits,
    ReplayBatchEvent, ReplayChainCursor, ReplayCommitment, ReplayError, ReplayLimits,
    ReplayRecorder, ReplayVerification, REPLAY_FORMAT_VERSION,
};
pub use replay_bundle::{
    ReplayBundle, ReplayBundleError, ReplayBundleLimits, ReplaySeek, REPLAY_BUNDLE_FORMAT_VERSION,
};
pub use replay_driver::{ReferenceReplayDriver, ReplayDriverError};
pub use replay_segment::{
    ReplayManifest, ReplayManifestLimits, ReplaySegment, ReplaySegmentDescriptor,
    ReplaySegmentLimits, ReplayStreamError, REPLAY_MANIFEST_FORMAT_VERSION,
    REPLAY_SEGMENT_FORMAT_VERSION,
};
