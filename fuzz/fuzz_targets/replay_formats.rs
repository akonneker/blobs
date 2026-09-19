#![no_main]

use blob_engine::resolution::{
    CHECKPOINT_FORMAT_VERSION, CheckpointLimits, MATCH_ATTESTATION_FORMAT_VERSION,
    MATCH_VERIFICATION_FORMAT_VERSION, MatchAttestation, MatchVerificationLimits,
    MatchVerificationManifest, REPLAY_BUNDLE_FORMAT_VERSION, REPLAY_FORMAT_VERSION,
    REPLAY_MANIFEST_FORMAT_VERSION, REPLAY_SEGMENT_FORMAT_VERSION, ReferenceCheckpoint,
    ReplayArchive, ReplayArchiveLimits, ReplayBatchEvent, ReplayBundle, ReplayBundleLimits,
    ReplayLimits, ReplayManifest, ReplayManifestLimits, ReplaySegment, ReplaySegmentLimits,
};
use libfuzzer_sys::fuzz_target;
use sha2::{Digest, Sha256};

const MAX_BYTES: usize = 8 * 1024;
const MAX_ITEMS: usize = 64;

macro_rules! assert_canonical {
    ($result:expr, $bytes:expr) => {
        if let Ok(value) = $result {
            assert_eq!(&*value.to_bytes(), $bytes);
        }
    };
}

fn prefixed(magic: &[u8; 8], version: u16, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(magic.len() + 2 + payload.len());
    body.extend_from_slice(magic);
    body.extend_from_slice(&version.to_le_bytes());
    body.extend_from_slice(payload);
    body
}

fn authenticated(
    magic: &[u8; 8],
    version: u16,
    domain: &[u8],
    payload: &[u8],
    include_body_length: bool,
) -> Vec<u8> {
    let mut body = prefixed(magic, version, payload);
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update(version.to_le_bytes());
    if include_body_length {
        hasher.update((body.len() as u64).to_le_bytes());
    }
    hasher.update(&body);
    body.extend_from_slice(&hasher.finalize());
    body
}

fn replay_limits() -> ReplayLimits {
    ReplayLimits {
        max_frame_bytes: MAX_BYTES,
        max_commitments: MAX_ITEMS,
        max_outcomes: MAX_ITEMS,
        max_claims: MAX_ITEMS,
        max_deaths: MAX_ITEMS,
        max_births: MAX_ITEMS,
        max_delta_tiles: MAX_ITEMS,
        max_delta_cells: MAX_ITEMS,
        max_private_memory_bytes: 512,
    }
}

fn checkpoint_limits() -> CheckpointLimits {
    CheckpointLimits {
        max_checkpoint_bytes: MAX_BYTES,
        max_tiles: MAX_ITEMS,
        max_cells: MAX_ITEMS,
        max_cell_slots: MAX_ITEMS * 4,
        max_private_memory_bytes: 512,
    }
}

fuzz_target!(|bytes: &[u8]| {
    let frame = replay_limits();
    let checkpoint = checkpoint_limits();
    let archive = ReplayArchiveLimits {
        max_archive_bytes: MAX_BYTES,
        max_events: MAX_ITEMS,
        frame,
    };

    assert_canonical!(
        ReplayBatchEvent::from_bytes_with_limits(bytes, frame),
        bytes
    );
    assert_canonical!(ReplayArchive::from_bytes_with_limits(bytes, archive), bytes);
    assert_canonical!(
        ReferenceCheckpoint::from_bytes_with_limits(bytes, checkpoint),
        bytes
    );
    assert_canonical!(
        ReplayBundle::from_bytes_with_limits(
            bytes,
            ReplayBundleLimits {
                max_bundle_bytes: MAX_BYTES,
                max_checkpoints: MAX_ITEMS,
                archive,
                checkpoint,
            },
        ),
        bytes
    );
    assert_canonical!(
        ReplaySegment::from_bytes_with_limits(
            bytes,
            ReplaySegmentLimits {
                max_segment_bytes: MAX_BYTES,
                max_events: MAX_ITEMS,
                frame,
                checkpoint,
            },
        ),
        bytes
    );
    assert_canonical!(
        ReplayManifest::from_bytes_with_limits(
            bytes,
            ReplayManifestLimits {
                max_manifest_bytes: MAX_BYTES,
                max_segments: MAX_ITEMS,
            },
        ),
        bytes
    );
    let manifest_limits = MatchVerificationLimits {
        max_bytes: MAX_BYTES,
        max_mind_artifacts: MAX_ITEMS,
    };
    assert_canonical!(
        MatchVerificationManifest::from_bytes_with_limits(bytes, manifest_limits),
        bytes
    );
    assert_canonical!(
        MatchAttestation::from_bytes(bytes, MAX_BYTES, manifest_limits),
        bytes
    );

    // Raw mutation almost always stops at an integrity hash. These normalized
    // envelopes preserve arbitrary bodies while repairing the outer magic,
    // version, and hash so coverage reaches bounded nested decoders.
    let event = authenticated(
        b"BLBRPL01",
        REPLAY_FORMAT_VERSION,
        b"blob.replay.batch",
        bytes,
        false,
    );
    let replay_archive = authenticated(
        b"BLBARC01",
        REPLAY_FORMAT_VERSION,
        b"blob.replay.archive",
        bytes,
        false,
    );
    let checkpoint_bytes = authenticated(
        b"BLBCHK06",
        CHECKPOINT_FORMAT_VERSION,
        b"blob.simulation.checkpoint",
        bytes,
        true,
    );
    let bundle = authenticated(
        b"BLBBND01",
        REPLAY_BUNDLE_FORMAT_VERSION,
        b"blob.replay.checkpoint-bundle",
        bytes,
        true,
    );
    let segment = authenticated(
        b"BLBSEG01",
        REPLAY_SEGMENT_FORMAT_VERSION,
        b"blob.replay.segment",
        bytes,
        true,
    );
    let replay_manifest = authenticated(
        b"BLBMAN01",
        REPLAY_MANIFEST_FORMAT_VERSION,
        b"blob.replay.manifest",
        bytes,
        true,
    );
    let match_manifest = authenticated(
        b"BLBVER01",
        MATCH_VERIFICATION_FORMAT_VERSION,
        b"blob.match.verification",
        bytes,
        false,
    );
    let attestation = prefixed(b"BLBATT01", MATCH_ATTESTATION_FORMAT_VERSION, bytes);

    assert_canonical!(
        ReplayBatchEvent::from_bytes_with_limits(&event, frame),
        &event
    );
    assert_canonical!(
        ReplayArchive::from_bytes_with_limits(&replay_archive, archive),
        &replay_archive
    );
    assert_canonical!(
        ReferenceCheckpoint::from_bytes_with_limits(&checkpoint_bytes, checkpoint),
        &checkpoint_bytes
    );
    assert_canonical!(
        ReplayBundle::from_bytes_with_limits(
            &bundle,
            ReplayBundleLimits {
                max_bundle_bytes: MAX_BYTES,
                max_checkpoints: MAX_ITEMS,
                archive,
                checkpoint,
            },
        ),
        &bundle
    );
    assert_canonical!(
        ReplaySegment::from_bytes_with_limits(
            &segment,
            ReplaySegmentLimits {
                max_segment_bytes: MAX_BYTES,
                max_events: MAX_ITEMS,
                frame,
                checkpoint,
            },
        ),
        &segment
    );
    assert_canonical!(
        ReplayManifest::from_bytes_with_limits(
            &replay_manifest,
            ReplayManifestLimits {
                max_manifest_bytes: MAX_BYTES,
                max_segments: MAX_ITEMS,
            },
        ),
        &replay_manifest
    );
    assert_canonical!(
        MatchVerificationManifest::from_bytes_with_limits(&match_manifest, manifest_limits),
        &match_manifest
    );
    assert_canonical!(
        MatchAttestation::from_bytes(&attestation, MAX_BYTES, manifest_limits),
        &attestation
    );
});
