//! Native server verification and public Ed25519 match attestations.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};

use crate::engine::TickEvents;
use crate::resolution::{
    attestation_key_id, attestation_signing_message, BatchReport, CanonicalHash, MatchAttestation,
    MatchAttestationError, MatchVerificationError, MatchVerificationLimits,
    MatchVerificationManifest, ReplayCommitment, ReplayError, ReplayLimits, ReplayRecorder,
    ReplaySegment, ReplayStreamError, ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES,
};

#[derive(Debug)]
pub enum ServerVerificationError {
    SigningKey(String),
    Attestation(MatchAttestationError),
    InvalidPublicKey,
    SigningKeyMismatch,
    InvalidSignature,
    Manifest(MatchVerificationError),
    Replay(ReplayError),
    ReplayStream(ReplayStreamError),
    RulesetMismatch,
    StartStateMismatch,
    SegmentStartMismatch,
    EmptyReplay,
    UnexpectedSegment { index: usize, count: usize },
    SegmentExhausted,
    SegmentIncomplete { verified: usize, expected: usize },
    MissingReferenceBatch,
    CommitmentMismatch { sequence: u64 },
    BatchMismatch { sequence: u64 },
}

impl Display for ServerVerificationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SigningKey(message) => write!(formatter, "invalid server signing key: {message}"),
            Self::Attestation(error) => Display::fmt(error, formatter),
            Self::InvalidPublicKey => write!(formatter, "invalid Ed25519 public key"),
            Self::SigningKeyMismatch => write!(formatter, "attestation signing key ID mismatch"),
            Self::InvalidSignature => write!(formatter, "invalid match attestation signature"),
            Self::Manifest(error) => Display::fmt(error, formatter),
            Self::Replay(error) => Display::fmt(error, formatter),
            Self::ReplayStream(error) => Display::fmt(error, formatter),
            Self::RulesetMismatch => write!(formatter, "server verifier ruleset mismatch"),
            Self::StartStateMismatch => write!(formatter, "server verifier start-state mismatch"),
            Self::SegmentStartMismatch => write!(formatter, "server replay segment chain mismatch"),
            Self::EmptyReplay => write!(formatter, "server cannot attest an empty replay"),
            Self::UnexpectedSegment { index, count } => write!(
                formatter,
                "server received replay segment {index}; manifest contains {count} segments"
            ),
            Self::SegmentExhausted => write!(formatter, "server replay segment is exhausted"),
            Self::SegmentIncomplete { verified, expected } => write!(
                formatter,
                "server verified {verified} batches; segment contains {expected}"
            ),
            Self::MissingReferenceBatch => {
                write!(formatter, "server tick did not produce a reference batch")
            }
            Self::CommitmentMismatch { sequence } => write!(
                formatter,
                "server Mind commitments diverged at replay sequence {sequence}"
            ),
            Self::BatchMismatch { sequence } => write!(
                formatter,
                "server resolution diverged at replay sequence {sequence}"
            ),
        }
    }
}

impl Error for ServerVerificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Attestation(error) => Some(error),
            Self::Manifest(error) => Some(error),
            Self::Replay(error) => Some(error),
            Self::ReplayStream(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatchAttestationError> for ServerVerificationError {
    fn from(value: MatchAttestationError) -> Self {
        Self::Attestation(value)
    }
}

impl From<MatchVerificationError> for ServerVerificationError {
    fn from(value: MatchVerificationError) -> Self {
        Self::Manifest(value)
    }
}

impl From<ReplayError> for ServerVerificationError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<ReplayStreamError> for ServerVerificationError {
    fn from(value: ReplayStreamError) -> Self {
        Self::ReplayStream(value)
    }
}

/// Ed25519 server key kept outside replay and Mind state.
pub struct ServerSigningKey(Ed25519KeyPair);

impl ServerSigningKey {
    pub fn from_seed(seed: [u8; 32]) -> Result<Self, ServerVerificationError> {
        Ed25519KeyPair::from_seed_unchecked(&seed)
            .map(Self)
            .map_err(|error| ServerVerificationError::SigningKey(error.to_string()))
    }

    pub fn public_key(&self) -> [u8; ED25519_PUBLIC_KEY_BYTES] {
        self.0
            .public_key()
            .as_ref()
            .try_into()
            .expect("ring Ed25519 public keys are 32 bytes")
    }

    pub fn key_id(&self) -> CanonicalHash {
        attestation_key_id(&self.public_key())
    }

    pub(crate) fn sign_message(&self, message: &[u8]) -> [u8; ED25519_SIGNATURE_BYTES] {
        self.0
            .sign(message)
            .as_ref()
            .try_into()
            .expect("ring Ed25519 signatures are 64 bytes")
    }
}

#[derive(Debug, Clone)]
pub struct AttestedMatchManifest {
    inner: MatchAttestation,
}

impl AttestedMatchManifest {
    pub fn sign(
        verified: VerifiedMatchManifest,
        key: &ServerSigningKey,
    ) -> Result<Self, ServerVerificationError> {
        let manifest = verified.0;
        let key_id = key.key_id();
        let message = attestation_signing_message(manifest.to_bytes());
        let signature = key.sign_message(&message);
        Ok(Self {
            inner: MatchAttestation::from_signature(manifest, key_id, signature),
        })
    }

    pub fn from_bytes(
        bytes: &[u8],
        max_bytes: usize,
        manifest_limits: MatchVerificationLimits,
    ) -> Result<Self, ServerVerificationError> {
        Ok(Self {
            inner: MatchAttestation::from_bytes(bytes, max_bytes, manifest_limits)?,
        })
    }

    pub fn verify(&self, public_key: &[u8]) -> Result<(), ServerVerificationError> {
        if public_key.len() != ED25519_PUBLIC_KEY_BYTES {
            return Err(ServerVerificationError::InvalidPublicKey);
        }
        self.inner
            .verify_key_id(public_key)
            .map_err(|error| match error {
                MatchAttestationError::SigningKeyMismatch => {
                    ServerVerificationError::SigningKeyMismatch
                }
                MatchAttestationError::InvalidPublicKeyLength(_) => {
                    ServerVerificationError::InvalidPublicKey
                }
                other => ServerVerificationError::Attestation(other),
            })?;
        UnparsedPublicKey::new(&ED25519, public_key)
            .verify(&self.inner.signing_message(), self.inner.signature())
            .map_err(|_| ServerVerificationError::InvalidSignature)
    }

    pub fn to_bytes(&self) -> &[u8] {
        self.inner.to_bytes()
    }

    pub const fn key_id(&self) -> CanonicalHash {
        self.inner.key_id()
    }

    pub fn manifest(&self) -> &MatchVerificationManifest {
        self.inner.manifest()
    }
}

/// Type-state proof that every replay segment was compared with server Mind
/// invocations under the manifest's bound context.
pub struct VerifiedMatchManifest(MatchVerificationManifest);

impl VerifiedMatchManifest {
    pub fn manifest(&self) -> &MatchVerificationManifest {
        &self.0
    }
}

/// Coordinates segment-at-a-time verification without buffering a whole match.
pub struct ServerMatchVerifier {
    verification_manifest: MatchVerificationManifest,
    replay_manifest: crate::resolution::ReplayManifest,
    current: ServerSegmentVerifier,
    segment_index: usize,
}

impl ServerMatchVerifier {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        verification_manifest: MatchVerificationManifest,
        replay_manifest: crate::resolution::ReplayManifest,
        mind_abi_hash: CanonicalHash,
        mind_runtime_profile: crate::resolution::MindRuntimeProfile,
        initial_runtime_hash: CanonicalHash,
        mind_artifacts: &[crate::resolution::MindArtifactBinding],
        first_segment: ReplaySegment,
        actual_compiled_ruleset_hash: CanonicalHash,
        actual_state_hash: CanonicalHash,
    ) -> Result<Self, ServerVerificationError> {
        verification_manifest.verify_context(
            mind_abi_hash,
            mind_runtime_profile,
            initial_runtime_hash,
            mind_artifacts,
            &replay_manifest,
        )?;
        if replay_manifest.segment_count() == 0 {
            return Err(ServerVerificationError::EmptyReplay);
        }
        replay_manifest.verify_segment(0, &first_segment)?;
        let current = ServerSegmentVerifier::from_segment(
            first_segment,
            actual_compiled_ruleset_hash,
            actual_state_hash,
        )?;
        Ok(Self {
            verification_manifest,
            replay_manifest,
            current,
            segment_index: 0,
        })
    }

    pub fn verify_tick(
        &mut self,
        tick: &TickEvents,
        limits: ReplayLimits,
    ) -> Result<(), ServerVerificationError> {
        self.current.verify_tick(tick, limits)
    }

    pub fn advance_segment(
        &mut self,
        segment: ReplaySegment,
        actual_compiled_ruleset_hash: CanonicalHash,
        actual_state_hash: CanonicalHash,
    ) -> Result<(), ServerVerificationError> {
        self.current.finish()?;
        let next_index = self.segment_index + 1;
        if next_index >= self.replay_manifest.segment_count() {
            return Err(ServerVerificationError::UnexpectedSegment {
                index: next_index,
                count: self.replay_manifest.segment_count(),
            });
        }
        self.replay_manifest.verify_segment(next_index, &segment)?;
        if self.current.cursor() != segment.start_cursor() {
            return Err(ServerVerificationError::SegmentStartMismatch);
        }
        self.current = ServerSegmentVerifier::from_segment(
            segment,
            actual_compiled_ruleset_hash,
            actual_state_hash,
        )?;
        self.segment_index = next_index;
        Ok(())
    }

    pub fn finish(self) -> Result<VerifiedMatchManifest, ServerVerificationError> {
        self.current.finish()?;
        if self.segment_index + 1 != self.replay_manifest.segment_count() {
            return Err(ServerVerificationError::SegmentIncomplete {
                verified: self.segment_index + 1,
                expected: self.replay_manifest.segment_count(),
            });
        }
        if self.current.cursor() != self.replay_manifest.final_cursor() {
            return Err(ServerVerificationError::SegmentStartMismatch);
        }
        Ok(VerifiedMatchManifest(self.verification_manifest))
    }
}

/// Compares batches produced by a trusted server Mind invocation against one
/// untrusted replay segment. The server Engine remains the source of actions.
pub struct ServerSegmentVerifier {
    segment: ReplaySegment,
    recorder: ReplayRecorder,
    next_event: usize,
}

impl ServerSegmentVerifier {
    pub fn from_segment(
        segment: ReplaySegment,
        actual_compiled_ruleset_hash: CanonicalHash,
        actual_state_hash: CanonicalHash,
    ) -> Result<Self, ServerVerificationError> {
        if segment.compiled_ruleset_hash() != actual_compiled_ruleset_hash {
            return Err(ServerVerificationError::RulesetMismatch);
        }
        if segment.start_cursor().previous_state_hash != actual_state_hash
            || segment.checkpoint().state_hash() != actual_state_hash
        {
            return Err(ServerVerificationError::StartStateMismatch);
        }
        Ok(Self {
            recorder: ReplayRecorder::from_cursor(
                actual_compiled_ruleset_hash,
                segment.start_cursor(),
            ),
            segment,
            next_event: 0,
        })
    }

    pub fn verify_tick(
        &mut self,
        tick: &TickEvents,
        limits: ReplayLimits,
    ) -> Result<(), ServerVerificationError> {
        let report = tick
            .reference_batch
            .as_ref()
            .ok_or(ServerVerificationError::MissingReferenceBatch)?;
        self.verify_batch(report, &tick.reference_commitments, limits)
    }

    pub fn verify_batch(
        &mut self,
        report: &BatchReport,
        commitments: &[ReplayCommitment],
        limits: ReplayLimits,
    ) -> Result<(), ServerVerificationError> {
        if self.next_event >= self.segment.event_count() {
            return Err(ServerVerificationError::SegmentExhausted);
        }
        let expected = self.segment.event(self.next_event, limits)?;
        let mut candidate_recorder = self.recorder.clone();
        let actual = candidate_recorder.record_with_commitments(report, commitments.to_vec())?;
        if actual.commitments != expected.commitments {
            return Err(ServerVerificationError::CommitmentMismatch {
                sequence: expected.sequence,
            });
        }
        if actual != expected {
            return Err(ServerVerificationError::BatchMismatch {
                sequence: expected.sequence,
            });
        }
        self.recorder = candidate_recorder;
        self.next_event += 1;
        Ok(())
    }

    pub fn finish(&self) -> Result<(), ServerVerificationError> {
        if self.next_event != self.segment.event_count() {
            return Err(ServerVerificationError::SegmentIncomplete {
                verified: self.next_event,
                expected: self.segment.event_count(),
            });
        }
        if self.recorder.cursor() != self.segment.end_cursor() {
            return Err(ServerVerificationError::SegmentStartMismatch);
        }
        Ok(())
    }

    pub fn cursor(&self) -> crate::resolution::ReplayChainCursor {
        self.recorder.cursor()
    }
}
