//! Native service boundary for admitted artifacts, streamed verification,
//! server-derived scoring, and atomic leaderboard publication.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ring::signature::{UnparsedPublicKey, ED25519};
use sha2::{Digest, Sha256};

use crate::engine::TickEvents;
use crate::mind_runtime::inspect_mind_artifact;
use crate::replay_store::{ReplayArtifactStore, ReplayStoreError};
use crate::resolution::{
    attestation_key_id, mind_artifact_hash, CanonicalHash, MatchAttestation,
    MatchVerificationLimits, MatchVerificationManifest, MindArtifactBinding, MindRuntimeProfile,
    ReferenceSimulation, ReplayLimits, ReplayManifest, ReplayManifestLimits,
    ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES,
};
use crate::server_verification::{
    AttestedMatchManifest, ServerMatchVerifier, ServerSigningKey, ServerVerificationError,
    VerifiedMatchManifest,
};

const PUBLICATION_MAGIC: &[u8; 8] = b"BLBPUB01";
const PUBLICATION_VERSION: u16 = 1;
const PUBLICATION_SIGNATURE_DOMAIN: &[u8] = b"blob.leaderboard.publication.signature";
const PUBLICATION_HASH_DOMAIN: &[u8] = b"blob.leaderboard.publication";
const MATCH_ID_DOMAIN: &[u8] = b"blob.online.match-id";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
pub struct OnlineServiceLimits {
    pub max_artifacts: usize,
    pub max_artifact_bytes: usize,
    pub max_total_artifact_bytes: usize,
    pub max_match_id_bytes: usize,
    pub verification_manifest: MatchVerificationLimits,
    pub replay_manifest: ReplayManifestLimits,
}

impl Default for OnlineServiceLimits {
    fn default() -> Self {
        Self {
            max_artifacts: 64,
            max_artifact_bytes: 4 * 1024 * 1024,
            max_total_artifact_bytes: 16 * 1024 * 1024,
            max_match_id_bytes: 128,
            verification_manifest: MatchVerificationLimits::default(),
            replay_manifest: ReplayManifestLimits::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubmittedMindArtifact {
    pub slot: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct AcceptedMindArtifact {
    slot: u64,
    artifact_hash: CanonicalHash,
    bytes: Vec<u8>,
}

impl AcceptedMindArtifact {
    pub const fn slot(&self) -> u64 {
        self.slot
    }

    pub const fn artifact_hash(&self) -> CanonicalHash {
        self.artifact_hash
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug, Clone)]
pub struct AcceptedMatchSubmission {
    verification_manifest: MatchVerificationManifest,
    replay_manifest: ReplayManifest,
    artifacts: Vec<AcceptedMindArtifact>,
}

impl AcceptedMatchSubmission {
    pub fn verification_manifest(&self) -> &MatchVerificationManifest {
        &self.verification_manifest
    }

    pub fn replay_manifest(&self) -> &ReplayManifest {
        &self.replay_manifest
    }

    pub fn artifacts(&self) -> &[AcceptedMindArtifact] {
        &self.artifacts
    }

    fn artifact_bindings(&self) -> Vec<MindArtifactBinding> {
        self.artifacts
            .iter()
            .map(|artifact| MindArtifactBinding {
                slot: artifact.slot,
                artifact_hash: artifact.artifact_hash,
            })
            .collect()
    }
}

#[derive(Debug)]
pub enum OnlineServiceError {
    TooManyArtifacts {
        actual: usize,
        limit: usize,
    },
    ArtifactTooLarge {
        slot: u64,
        actual: usize,
        limit: usize,
    },
    TotalArtifactsTooLarge {
        actual: usize,
        limit: usize,
    },
    ArtifactSlotsNotStrictlyOrdered,
    ArtifactRuntimeProfile {
        slot: u64,
        message: String,
    },
    InvalidMatchId,
    MatchIdMismatch,
    TerminalRulesetMismatch,
    TerminalStateMismatch,
    ScorePolicy(String),
    InvalidPublication(&'static str),
    PublicationTooLarge {
        actual: usize,
        limit: usize,
    },
    PublicationConflict,
    TooManyLeaderboardEntries {
        actual: usize,
        limit: usize,
    },
    Io(std::io::Error),
    MatchManifest(crate::resolution::MatchVerificationError),
    ReplayManifest(crate::resolution::ReplayStreamError),
    ReplayStore(ReplayStoreError),
    Verification(ServerVerificationError),
    Attestation(crate::resolution::MatchAttestationError),
}

impl Display for OnlineServiceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyArtifacts { actual, limit } => {
                write!(
                    formatter,
                    "submission has {actual} artifacts; limit is {limit}"
                )
            }
            Self::ArtifactTooLarge {
                slot,
                actual,
                limit,
            } => write!(
                formatter,
                "artifact in slot {slot} is {actual} bytes; limit is {limit}"
            ),
            Self::TotalArtifactsTooLarge { actual, limit } => write!(
                formatter,
                "submission artifacts total {actual} bytes; limit is {limit}"
            ),
            Self::ArtifactSlotsNotStrictlyOrdered => {
                write!(formatter, "artifact slots are not strictly ordered")
            }
            Self::ArtifactRuntimeProfile { slot, message } => {
                write!(
                    formatter,
                    "artifact in slot {slot} violates its runtime profile: {message}"
                )
            }
            Self::InvalidMatchId => write!(formatter, "invalid service match ID"),
            Self::MatchIdMismatch => write!(formatter, "submission match ID hash mismatch"),
            Self::TerminalRulesetMismatch => write!(formatter, "terminal ruleset hash mismatch"),
            Self::TerminalStateMismatch => write!(formatter, "terminal state hash mismatch"),
            Self::ScorePolicy(message) => write!(formatter, "score policy failed: {message}"),
            Self::InvalidPublication(message) => {
                write!(formatter, "invalid leaderboard publication: {message}")
            }
            Self::PublicationTooLarge { actual, limit } => write!(
                formatter,
                "leaderboard publication is {actual} bytes; limit is {limit}"
            ),
            Self::PublicationConflict => {
                write!(
                    formatter,
                    "a different leaderboard publication already exists"
                )
            }
            Self::TooManyLeaderboardEntries { actual, limit } => write!(
                formatter,
                "leaderboard contains at least {actual} entries; scan limit is {limit}"
            ),
            Self::Io(error) => Display::fmt(error, formatter),
            Self::MatchManifest(error) => Display::fmt(error, formatter),
            Self::ReplayManifest(error) => Display::fmt(error, formatter),
            Self::ReplayStore(error) => Display::fmt(error, formatter),
            Self::Verification(error) => Display::fmt(error, formatter),
            Self::Attestation(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for OnlineServiceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::MatchManifest(error) => Some(error),
            Self::ReplayManifest(error) => Some(error),
            Self::ReplayStore(error) => Some(error),
            Self::Verification(error) => Some(error),
            Self::Attestation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for OnlineServiceError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<crate::resolution::MatchVerificationError> for OnlineServiceError {
    fn from(value: crate::resolution::MatchVerificationError) -> Self {
        Self::MatchManifest(value)
    }
}

impl From<crate::resolution::ReplayStreamError> for OnlineServiceError {
    fn from(value: crate::resolution::ReplayStreamError) -> Self {
        Self::ReplayManifest(value)
    }
}

impl From<ReplayStoreError> for OnlineServiceError {
    fn from(value: ReplayStoreError) -> Self {
        Self::ReplayStore(value)
    }
}

impl From<ServerVerificationError> for OnlineServiceError {
    fn from(value: ServerVerificationError) -> Self {
        Self::Verification(value)
    }
}

impl From<crate::resolution::MatchAttestationError> for OnlineServiceError {
    fn from(value: crate::resolution::MatchAttestationError) -> Self {
        Self::Attestation(value)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn admit_match_submission(
    verification_manifest_bytes: &[u8],
    replay_manifest_bytes: &[u8],
    artifacts: Vec<SubmittedMindArtifact>,
    match_id: &[u8],
    expected_mind_abi_hash: CanonicalHash,
    expected_mind_runtime_profile: MindRuntimeProfile,
    actual_initial_runtime_hash: CanonicalHash,
    limits: OnlineServiceLimits,
) -> Result<AcceptedMatchSubmission, OnlineServiceError> {
    if match_id.is_empty() || match_id.len() > limits.max_match_id_bytes {
        return Err(OnlineServiceError::InvalidMatchId);
    }
    if artifacts.len() > limits.max_artifacts {
        return Err(OnlineServiceError::TooManyArtifacts {
            actual: artifacts.len(),
            limit: limits.max_artifacts,
        });
    }
    if !artifacts.windows(2).all(|pair| pair[0].slot < pair[1].slot) {
        return Err(OnlineServiceError::ArtifactSlotsNotStrictlyOrdered);
    }
    let mut total_bytes = 0_usize;
    let mut accepted_artifacts = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        if artifact.bytes.len() > limits.max_artifact_bytes {
            return Err(OnlineServiceError::ArtifactTooLarge {
                slot: artifact.slot,
                actual: artifact.bytes.len(),
                limit: limits.max_artifact_bytes,
            });
        }
        total_bytes = total_bytes.checked_add(artifact.bytes.len()).ok_or(
            OnlineServiceError::TotalArtifactsTooLarge {
                actual: usize::MAX,
                limit: limits.max_total_artifact_bytes,
            },
        )?;
        if total_bytes > limits.max_total_artifact_bytes {
            return Err(OnlineServiceError::TotalArtifactsTooLarge {
                actual: total_bytes,
                limit: limits.max_total_artifact_bytes,
            });
        }
        inspect_mind_artifact(&artifact.bytes, expected_mind_runtime_profile).map_err(|error| {
            OnlineServiceError::ArtifactRuntimeProfile {
                slot: artifact.slot,
                message: error.to_string(),
            }
        })?;
        accepted_artifacts.push(AcceptedMindArtifact {
            slot: artifact.slot,
            artifact_hash: mind_artifact_hash(&artifact.bytes),
            bytes: artifact.bytes,
        });
    }

    let replay_manifest =
        ReplayManifest::from_bytes_with_limits(replay_manifest_bytes, limits.replay_manifest)?;
    let verification_manifest = MatchVerificationManifest::from_bytes_with_limits(
        verification_manifest_bytes,
        limits.verification_manifest,
    )?;
    if verification_manifest.match_id_hash() != online_match_id_hash(match_id) {
        return Err(OnlineServiceError::MatchIdMismatch);
    }
    let bindings: Vec<_> = accepted_artifacts
        .iter()
        .map(|artifact| MindArtifactBinding {
            slot: artifact.slot,
            artifact_hash: artifact.artifact_hash,
        })
        .collect();
    verification_manifest.verify_context(
        expected_mind_abi_hash,
        expected_mind_runtime_profile,
        actual_initial_runtime_hash,
        &bindings,
        &replay_manifest,
    )?;
    Ok(AcceptedMatchSubmission {
        verification_manifest,
        replay_manifest,
        artifacts: accepted_artifacts,
    })
}

/// Domain-separated hash for the service-visible match identifier. It remains
/// verification-envelope metadata and is never passed into the Mind ABI.
pub fn online_match_id_hash(match_id: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((MATCH_ID_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(MATCH_ID_DOMAIN);
    hasher.update((match_id.len() as u64).to_le_bytes());
    hasher.update(match_id);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

/// Drives verification from immutable stored segments while retaining only one
/// decoded segment at a time. The trusted execution host supplies each tick.
pub struct StoredVerificationSession<'a, S: ReplayArtifactStore> {
    store: &'a S,
    verifier: ServerMatchVerifier,
    replay_manifest: ReplayManifest,
    segment_index: usize,
    event_in_segment: usize,
    events_in_segment: usize,
}

impl<'a, S: ReplayArtifactStore> StoredVerificationSession<'a, S> {
    pub fn start(
        submission: AcceptedMatchSubmission,
        store: &'a S,
        actual_compiled_ruleset_hash: CanonicalHash,
        actual_state_hash: CanonicalHash,
    ) -> Result<Self, OnlineServiceError> {
        let first_descriptor = submission
            .replay_manifest
            .segment(0)
            .ok_or(ServerVerificationError::EmptyReplay)?;
        let first_segment = store.load_segment(first_descriptor.segment_hash)?;
        let events_in_segment = first_segment.event_count();
        let bindings = submission.artifact_bindings();
        let mind_abi_hash = submission.verification_manifest.mind_abi_hash();
        let mind_runtime_profile = submission.verification_manifest.mind_runtime_profile();
        let initial_runtime_hash = submission.verification_manifest.initial_runtime_hash();
        let verifier = ServerMatchVerifier::start(
            submission.verification_manifest,
            submission.replay_manifest.clone(),
            mind_abi_hash,
            mind_runtime_profile,
            initial_runtime_hash,
            &bindings,
            first_segment,
            actual_compiled_ruleset_hash,
            actual_state_hash,
        )?;
        Ok(Self {
            store,
            verifier,
            replay_manifest: submission.replay_manifest,
            segment_index: 0,
            event_in_segment: 0,
            events_in_segment,
        })
    }

    /// Verifies one server-generated completion batch. At a segment boundary,
    /// the next stored segment is loaded only after the preceding one finishes.
    pub fn verify_tick(
        &mut self,
        tick: &TickEvents,
        actual_compiled_ruleset_hash: CanonicalHash,
        actual_state_hash: CanonicalHash,
        limits: ReplayLimits,
    ) -> Result<(), OnlineServiceError> {
        self.verifier.verify_tick(tick, limits)?;
        self.event_in_segment += 1;
        if self.event_in_segment == self.events_in_segment
            && self.segment_index + 1 < self.replay_manifest.segment_count()
        {
            let next_index = self.segment_index + 1;
            let descriptor = self.replay_manifest.segment(next_index).ok_or(
                OnlineServiceError::InvalidPublication("missing replay segment descriptor"),
            )?;
            let segment = self.store.load_segment(descriptor.segment_hash)?;
            self.events_in_segment = segment.event_count();
            self.verifier.advance_segment(
                segment,
                actual_compiled_ruleset_hash,
                actual_state_hash,
            )?;
            self.segment_index = next_index;
            self.event_in_segment = 0;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<VerifiedMatchManifest, OnlineServiceError> {
        Ok(self.verifier.finish()?)
    }
}

pub trait ServerScorePolicy<C> {
    fn policy_hash(&self) -> CanonicalHash;

    fn derive_score(
        &self,
        simulation: &ReferenceSimulation,
        trusted_context: &C,
    ) -> Result<i64, String>;
}

pub struct ScoredVerifiedMatch {
    verified: VerifiedMatchManifest,
    season_hash: CanonicalHash,
    score_policy_hash: CanonicalHash,
    score: i64,
    final_state_hash: CanonicalHash,
}

pub fn derive_verified_score<C, P: ServerScorePolicy<C>>(
    verified: VerifiedMatchManifest,
    simulation: &ReferenceSimulation,
    season_hash: CanonicalHash,
    policy: &P,
    trusted_context: &C,
) -> Result<ScoredVerifiedMatch, OnlineServiceError> {
    let manifest = verified.manifest();
    if manifest.compiled_ruleset_hash() != simulation.compiled_ruleset_hash() {
        return Err(OnlineServiceError::TerminalRulesetMismatch);
    }
    let final_state_hash = simulation.state_hash();
    if manifest.final_cursor().previous_state_hash != final_state_hash {
        return Err(OnlineServiceError::TerminalStateMismatch);
    }
    let score = policy
        .derive_score(simulation, trusted_context)
        .map_err(OnlineServiceError::ScorePolicy)?;
    Ok(ScoredVerifiedMatch {
        verified,
        season_hash,
        score_policy_hash: policy.policy_hash(),
        score,
        final_state_hash,
    })
}

#[derive(Debug, Clone)]
pub struct LeaderboardPublication {
    bytes: Vec<u8>,
    publication_hash: CanonicalHash,
    key_id: CanonicalHash,
    season_hash: CanonicalHash,
    score_policy_hash: CanonicalHash,
    score: i64,
    final_state_hash: CanonicalHash,
    attestation: MatchAttestation,
    signature: [u8; ED25519_SIGNATURE_BYTES],
}

impl LeaderboardPublication {
    pub fn sign(
        scored: ScoredVerifiedMatch,
        key: &ServerSigningKey,
    ) -> Result<Self, OnlineServiceError> {
        let attested = AttestedMatchManifest::sign(scored.verified, key)?;
        let key_id = key.key_id();
        let body = encode_publication_body(
            key_id,
            scored.season_hash,
            scored.score_policy_hash,
            scored.score,
            scored.final_state_hash,
            attested.to_bytes(),
        );
        let signature = key.sign_message(&publication_signing_message(&body));
        let bytes = finish_publication(body, &signature);
        Self::from_bytes(
            &bytes,
            bytes.len(),
            MatchVerificationLimits {
                max_bytes: usize::MAX,
                max_mind_artifacts: usize::MAX,
            },
        )
    }

    pub fn from_bytes(
        bytes: &[u8],
        max_bytes: usize,
        manifest_limits: MatchVerificationLimits,
    ) -> Result<Self, OnlineServiceError> {
        if bytes.len() > max_bytes {
            return Err(OnlineServiceError::PublicationTooLarge {
                actual: bytes.len(),
                limit: max_bytes,
            });
        }
        let minimum = 8 + 2 + 32 * 4 + 8 + 8 + ED25519_SIGNATURE_BYTES + 32;
        if bytes.len() < minimum {
            return Err(OnlineServiceError::InvalidPublication("truncated"));
        }
        let hash_position = bytes.len() - 32;
        let signed_position = hash_position
            .checked_sub(ED25519_SIGNATURE_BYTES)
            .ok_or(OnlineServiceError::InvalidPublication("truncated"))?;
        let actual_hash = publication_hash(&bytes[..hash_position]);
        let stored_hash = CanonicalHash::from_bytes(
            bytes[hash_position..]
                .try_into()
                .map_err(|_| OnlineServiceError::InvalidPublication("truncated hash"))?,
        );
        if actual_hash != stored_hash {
            return Err(OnlineServiceError::InvalidPublication(
                "integrity hash mismatch",
            ));
        }
        let body = &bytes[..signed_position];
        let signature: [u8; ED25519_SIGNATURE_BYTES] = bytes[signed_position..hash_position]
            .try_into()
            .map_err(|_| OnlineServiceError::InvalidPublication("truncated signature"))?;
        let mut position = 0_usize;
        if take(body, &mut position, 8)? != PUBLICATION_MAGIC {
            return Err(OnlineServiceError::InvalidPublication("invalid magic"));
        }
        let version = read_u16(body, &mut position)?;
        if version != PUBLICATION_VERSION {
            return Err(OnlineServiceError::InvalidPublication(
                "unsupported version",
            ));
        }
        let key_id = read_hash(body, &mut position)?;
        let season_hash = read_hash(body, &mut position)?;
        let score_policy_hash = read_hash(body, &mut position)?;
        let score = read_i64(body, &mut position)?;
        let final_state_hash = read_hash(body, &mut position)?;
        let attestation_length = read_u64(body, &mut position)?;
        let attestation_length = usize::try_from(attestation_length)
            .map_err(|_| OnlineServiceError::InvalidPublication("attestation too large"))?;
        let attestation = MatchAttestation::from_bytes(
            take(body, &mut position, attestation_length)?,
            max_bytes,
            manifest_limits,
        )?;
        if position != body.len() {
            return Err(OnlineServiceError::InvalidPublication("trailing bytes"));
        }
        if attestation.key_id() != key_id
            || attestation.manifest().final_cursor().previous_state_hash != final_state_hash
        {
            return Err(OnlineServiceError::InvalidPublication(
                "attestation binding mismatch",
            ));
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            publication_hash: stored_hash,
            key_id,
            season_hash,
            score_policy_hash,
            score,
            final_state_hash,
            attestation,
            signature,
        })
    }

    pub fn verify(&self, public_key: &[u8]) -> Result<(), OnlineServiceError> {
        if public_key.len() != ED25519_PUBLIC_KEY_BYTES
            || attestation_key_id(public_key) != self.key_id
        {
            return Err(OnlineServiceError::InvalidPublication(
                "signing key mismatch",
            ));
        }
        let hash_position = self.bytes.len() - 32;
        let body_length = hash_position - ED25519_SIGNATURE_BYTES;
        UnparsedPublicKey::new(&ED25519, public_key)
            .verify(
                &publication_signing_message(&self.bytes[..body_length]),
                &self.signature,
            )
            .map_err(|_| OnlineServiceError::InvalidPublication("invalid signature"))?;
        UnparsedPublicKey::new(&ED25519, public_key)
            .verify(
                &self.attestation.signing_message(),
                self.attestation.signature(),
            )
            .map_err(|_| OnlineServiceError::InvalidPublication("invalid match attestation"))
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn publication_hash(&self) -> CanonicalHash {
        self.publication_hash
    }

    pub const fn season_hash(&self) -> CanonicalHash {
        self.season_hash
    }

    pub const fn score_policy_hash(&self) -> CanonicalHash {
        self.score_policy_hash
    }

    pub const fn score(&self) -> i64 {
        self.score
    }

    pub const fn final_state_hash(&self) -> CanonicalHash {
        self.final_state_hash
    }

    pub fn attestation(&self) -> &MatchAttestation {
        &self.attestation
    }

    pub fn verification_manifest_hash(&self) -> CanonicalHash {
        self.attestation.manifest().manifest_hash()
    }

    pub fn replay_manifest_hash(&self) -> CanonicalHash {
        self.attestation.manifest().replay_manifest_hash()
    }
}

fn encode_publication_body(
    key_id: CanonicalHash,
    season_hash: CanonicalHash,
    score_policy_hash: CanonicalHash,
    score: i64,
    final_state_hash: CanonicalHash,
    attestation: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + 2 + 32 * 4 + 8 + 8 + attestation.len());
    bytes.extend_from_slice(PUBLICATION_MAGIC);
    bytes.extend_from_slice(&PUBLICATION_VERSION.to_le_bytes());
    bytes.extend_from_slice(key_id.as_bytes());
    bytes.extend_from_slice(season_hash.as_bytes());
    bytes.extend_from_slice(score_policy_hash.as_bytes());
    bytes.extend_from_slice(&score.to_le_bytes());
    bytes.extend_from_slice(final_state_hash.as_bytes());
    bytes.extend_from_slice(&(attestation.len() as u64).to_le_bytes());
    bytes.extend_from_slice(attestation);
    bytes
}

fn finish_publication(mut body: Vec<u8>, signature: &[u8; ED25519_SIGNATURE_BYTES]) -> Vec<u8> {
    body.extend_from_slice(signature);
    let hash = publication_hash(&body);
    body.extend_from_slice(hash.as_bytes());
    body
}

fn publication_signing_message(body: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(8 + PUBLICATION_SIGNATURE_DOMAIN.len() + 2 + body.len());
    message.extend_from_slice(&(PUBLICATION_SIGNATURE_DOMAIN.len() as u64).to_le_bytes());
    message.extend_from_slice(PUBLICATION_SIGNATURE_DOMAIN);
    message.extend_from_slice(&PUBLICATION_VERSION.to_le_bytes());
    message.extend_from_slice(body);
    message
}

fn publication_hash(bytes: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((PUBLICATION_HASH_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(PUBLICATION_HASH_DOMAIN);
    hasher.update(PUBLICATION_VERSION.to_le_bytes());
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

fn take<'a>(
    bytes: &'a [u8],
    position: &mut usize,
    count: usize,
) -> Result<&'a [u8], OnlineServiceError> {
    let end = position
        .checked_add(count)
        .ok_or(OnlineServiceError::InvalidPublication("truncated"))?;
    let value = bytes
        .get(*position..end)
        .ok_or(OnlineServiceError::InvalidPublication("truncated"))?;
    *position = end;
    Ok(value)
}

fn read_u16(bytes: &[u8], position: &mut usize) -> Result<u16, OnlineServiceError> {
    Ok(u16::from_le_bytes(
        take(bytes, position, 2)?
            .try_into()
            .map_err(|_| OnlineServiceError::InvalidPublication("truncated integer"))?,
    ))
}

fn read_u64(bytes: &[u8], position: &mut usize) -> Result<u64, OnlineServiceError> {
    Ok(u64::from_le_bytes(
        take(bytes, position, 8)?
            .try_into()
            .map_err(|_| OnlineServiceError::InvalidPublication("truncated integer"))?,
    ))
}

fn read_i64(bytes: &[u8], position: &mut usize) -> Result<i64, OnlineServiceError> {
    Ok(i64::from_le_bytes(
        take(bytes, position, 8)?
            .try_into()
            .map_err(|_| OnlineServiceError::InvalidPublication("truncated integer"))?,
    ))
}

fn read_hash(bytes: &[u8], position: &mut usize) -> Result<CanonicalHash, OnlineServiceError> {
    Ok(CanonicalHash::from_bytes(
        take(bytes, position, 32)?
            .try_into()
            .map_err(|_| OnlineServiceError::InvalidPublication("truncated hash"))?,
    ))
}

#[derive(Debug, Clone, Copy)]
pub struct LeaderboardStoreLimits {
    pub max_publication_bytes: usize,
    pub max_entries_per_scan: usize,
    pub manifest: MatchVerificationLimits,
}

impl Default for LeaderboardStoreLimits {
    fn default() -> Self {
        Self {
            max_publication_bytes: 2 * 1024 * 1024,
            max_entries_per_scan: 1_000_000,
            manifest: MatchVerificationLimits::default(),
        }
    }
}

/// Append-only filesystem publication store. A complete signed record is made
/// visible with one atomic hard link; readers therefore see either no entry or
/// the full score/attestation/replay association.
#[derive(Debug, Clone)]
pub struct FileLeaderboardStore {
    root: PathBuf,
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    limits: LeaderboardStoreLimits,
}

impl FileLeaderboardStore {
    pub fn open(
        root: impl Into<PathBuf>,
        public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
        limits: LeaderboardStoreLimits,
    ) -> Result<Self, OnlineServiceError> {
        let store = Self {
            root: root.into(),
            public_key,
            limits,
        };
        fs::create_dir_all(store.root.join("seasons"))?;
        sync_directory(&store.root)?;
        Ok(store)
    }

    pub fn publish<S: ReplayArtifactStore>(
        &self,
        publication: &LeaderboardPublication,
        replay_store: &S,
    ) -> Result<bool, OnlineServiceError> {
        self.validate(publication)?;
        replay_store.load_manifest(publication.replay_manifest_hash())?;
        let path = self.entry_path(
            publication.season_hash(),
            publication.verification_manifest_hash(),
        );
        let parent = path.parent().ok_or(OnlineServiceError::InvalidPublication(
            "missing entry parent",
        ))?;
        self.ensure_season_directory(publication.season_hash())?;
        if path.exists() {
            return self
                .compare_existing(&path, publication.to_bytes())
                .map(|_| false);
        }
        let temporary = temporary_path(parent);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let result = (|| -> Result<bool, OnlineServiceError> {
            file.write_all(publication.to_bytes())?;
            file.sync_all()?;
            match fs::hard_link(&temporary, &path) {
                Ok(()) => {
                    sync_directory(parent)?;
                    Ok(true)
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    self.compare_existing(&path, publication.to_bytes())?;
                    Ok(false)
                }
                Err(error) => Err(error.into()),
            }
        })();
        let _ = fs::remove_file(temporary);
        result
    }

    pub fn load(
        &self,
        season_hash: CanonicalHash,
        verification_manifest_hash: CanonicalHash,
    ) -> Result<Option<LeaderboardPublication>, OnlineServiceError> {
        let path = self.entry_path(season_hash, verification_manifest_hash);
        let bytes = match read_bounded(&path, self.limits.max_publication_bytes)? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let publication = LeaderboardPublication::from_bytes(
            &bytes,
            self.limits.max_publication_bytes,
            self.limits.manifest,
        )?;
        self.validate(&publication)?;
        if publication.season_hash() != season_hash
            || publication.verification_manifest_hash() != verification_manifest_hash
        {
            return Err(OnlineServiceError::InvalidPublication(
                "entry path mismatch",
            ));
        }
        Ok(Some(publication))
    }

    pub fn leaderboard(
        &self,
        season_hash: CanonicalHash,
        result_limit: usize,
    ) -> Result<Vec<LeaderboardPublication>, OnlineServiceError> {
        if result_limit == 0 {
            return Ok(Vec::new());
        }
        let directory = self.season_directory(season_hash);
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut publications = Vec::new();
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_file()
                || entry.path().extension().and_then(|value| value.to_str()) != Some("blbpub")
            {
                continue;
            }
            if publications.len() >= self.limits.max_entries_per_scan {
                return Err(OnlineServiceError::TooManyLeaderboardEntries {
                    actual: publications.len() + 1,
                    limit: self.limits.max_entries_per_scan,
                });
            }
            let Some(bytes) = read_bounded(&entry.path(), self.limits.max_publication_bytes)?
            else {
                continue;
            };
            let publication = LeaderboardPublication::from_bytes(
                &bytes,
                self.limits.max_publication_bytes,
                self.limits.manifest,
            )?;
            self.validate(&publication)?;
            if publication.season_hash() != season_hash {
                return Err(OnlineServiceError::InvalidPublication(
                    "season path mismatch",
                ));
            }
            let expected_name = format!(
                "{}.blbpub",
                publication.verification_manifest_hash().to_hex()
            );
            if entry.file_name().to_str() != Some(expected_name.as_str()) {
                return Err(OnlineServiceError::InvalidPublication(
                    "entry filename mismatch",
                ));
            }
            publications.push(publication);
        }
        publications.sort_by(|left, right| {
            right.score().cmp(&left.score()).then_with(|| {
                left.verification_manifest_hash()
                    .cmp(&right.verification_manifest_hash())
            })
        });
        publications.truncate(result_limit);
        Ok(publications)
    }

    fn validate(&self, publication: &LeaderboardPublication) -> Result<(), OnlineServiceError> {
        if publication.to_bytes().len() > self.limits.max_publication_bytes {
            return Err(OnlineServiceError::PublicationTooLarge {
                actual: publication.to_bytes().len(),
                limit: self.limits.max_publication_bytes,
            });
        }
        publication.verify(&self.public_key)
    }

    fn compare_existing(&self, path: &Path, expected: &[u8]) -> Result<(), OnlineServiceError> {
        let actual = read_bounded(path, self.limits.max_publication_bytes)?
            .ok_or(OnlineServiceError::PublicationConflict)?;
        if actual == expected {
            Ok(())
        } else {
            Err(OnlineServiceError::PublicationConflict)
        }
    }

    fn season_directory(&self, season_hash: CanonicalHash) -> PathBuf {
        self.root
            .join("seasons")
            .join(season_hash.to_hex())
            .join("entries")
    }

    fn ensure_season_directory(
        &self,
        season_hash: CanonicalHash,
    ) -> Result<(), OnlineServiceError> {
        let seasons = self.root.join("seasons");
        let season = seasons.join(season_hash.to_hex());
        let entries = season.join("entries");
        fs::create_dir_all(&entries)?;
        sync_directory(&entries)?;
        sync_directory(&season)?;
        sync_directory(&seasons)?;
        Ok(())
    }

    fn entry_path(
        &self,
        season_hash: CanonicalHash,
        verification_manifest_hash: CanonicalHash,
    ) -> PathBuf {
        self.season_directory(season_hash)
            .join(format!("{}.blbpub", verification_manifest_hash.to_hex()))
    }
}

fn read_bounded(path: &Path, limit: usize) -> Result<Option<Vec<u8>>, OnlineServiceError> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let length = file.metadata()?.len();
    if length > limit as u64 {
        return Err(OnlineServiceError::PublicationTooLarge {
            actual: usize::try_from(length).unwrap_or(usize::MAX),
            limit,
        });
    }
    let mut bytes = Vec::with_capacity(usize::try_from(length).unwrap_or(limit).min(limit));
    Read::by_ref(&mut file)
        .take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(OnlineServiceError::PublicationTooLarge {
            actual: bytes.len(),
            limit,
        });
    }
    Ok(Some(bytes))
}

fn temporary_path(parent: &Path) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        ".publication-{}-{sequence}.tmp",
        std::process::id()
    ))
}

fn sync_directory(path: &Path) -> Result<(), OnlineServiceError> {
    File::open(path)?.sync_all()?;
    Ok(())
}
