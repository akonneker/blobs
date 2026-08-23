//! Native content-addressed storage for streamed replay artifacts.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::resolution::{
    CanonicalHash, ReplayManifest, ReplayManifestLimits, ReplaySegment, ReplaySegmentLimits,
};

const POINTER_MAGIC: &[u8; 8] = b"BLBPTR01";
const POINTER_VERSION: u16 = 1;
const POINTER_DOMAIN: &[u8] = b"blob.replay.current-manifest";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayStoreLimits {
    pub segment: ReplaySegmentLimits,
    pub manifest: ReplayManifestLimits,
    pub max_match_id_bytes: usize,
}

impl Default for ReplayStoreLimits {
    fn default() -> Self {
        Self {
            segment: ReplaySegmentLimits::default(),
            manifest: ReplayManifestLimits::default(),
            max_match_id_bytes: 128,
        }
    }
}

#[derive(Debug)]
pub enum ReplayStoreError {
    Io(std::io::Error),
    InvalidMatchId,
    ObjectTooLarge {
        kind: &'static str,
        actual: u64,
        limit: usize,
    },
    MissingObject {
        kind: &'static str,
        hash: CanonicalHash,
    },
    CorruptObject {
        kind: &'static str,
        hash: CanonicalHash,
        message: String,
    },
    ContentMismatch {
        kind: &'static str,
        hash: CanonicalHash,
    },
    PublishBusy,
    PublishConflict {
        expected: Option<CanonicalHash>,
        actual: Option<CanonicalHash>,
    },
    NonAppendOnlyManifest,
    SegmentLengthMismatch {
        hash: CanonicalHash,
        expected: u64,
        actual: u64,
    },
    InvalidPointer,
}

impl Display for ReplayStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => Display::fmt(error, formatter),
            Self::InvalidMatchId => write!(formatter, "invalid replay match ID"),
            Self::ObjectTooLarge {
                kind,
                actual,
                limit,
            } => write!(formatter, "{kind} is {actual} bytes; limit is {limit}"),
            Self::MissingObject { kind, hash } => {
                write!(formatter, "missing {kind} object {hash}")
            }
            Self::CorruptObject {
                kind,
                hash,
                message,
            } => write!(formatter, "corrupt {kind} object {hash}: {message}"),
            Self::ContentMismatch { kind, hash } => write!(
                formatter,
                "existing {kind} object {hash} does not match its content address"
            ),
            Self::PublishBusy => write!(formatter, "replay manifest publication is busy"),
            Self::PublishConflict { expected, actual } => write!(
                formatter,
                "replay manifest compare-and-swap conflict: expected {expected:?}, got {actual:?}"
            ),
            Self::NonAppendOnlyManifest => {
                write!(
                    formatter,
                    "replay manifest publication would rewrite history"
                )
            }
            Self::SegmentLengthMismatch {
                hash,
                expected,
                actual,
            } => write!(
                formatter,
                "segment {hash} length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidPointer => write!(formatter, "invalid current-manifest pointer"),
        }
    }
}

impl Error for ReplayStoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ReplayStoreError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayPublishOutcome {
    pub previous_manifest: Option<CanonicalHash>,
    pub current_manifest: CanonicalHash,
    pub appended_segments: usize,
}

pub trait ReplayArtifactStore {
    fn put_segment(&self, segment: &ReplaySegment) -> Result<CanonicalHash, ReplayStoreError>;

    fn load_segment(&self, hash: CanonicalHash) -> Result<ReplaySegment, ReplayStoreError>;

    fn current_manifest_hash(
        &self,
        match_id: &str,
    ) -> Result<Option<CanonicalHash>, ReplayStoreError>;

    fn load_manifest(&self, hash: CanonicalHash) -> Result<ReplayManifest, ReplayStoreError>;

    fn publish_manifest(
        &self,
        match_id: &str,
        manifest: &ReplayManifest,
        expected_previous: Option<CanonicalHash>,
    ) -> Result<ReplayPublishOutcome, ReplayStoreError>;
}

/// Filesystem implementation suitable for a single host or a shared POSIX
/// volume. Objects are immutable; publication is serialized per match with an
/// atomic directory lock and an atomic current-pointer rename.
#[derive(Debug, Clone)]
pub struct FileReplayStore {
    root: PathBuf,
    limits: ReplayStoreLimits,
}

impl FileReplayStore {
    pub fn open(
        root: impl Into<PathBuf>,
        limits: ReplayStoreLimits,
    ) -> Result<Self, ReplayStoreError> {
        let store = Self {
            root: root.into(),
            limits,
        };
        fs::create_dir_all(store.root.join("segments"))?;
        fs::create_dir_all(store.root.join("manifests"))?;
        fs::create_dir_all(store.root.join("matches"))?;
        sync_directory(&store.root)?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn current_manifest(
        &self,
        match_id: &str,
    ) -> Result<Option<ReplayManifest>, ReplayStoreError> {
        self.current_manifest_hash(match_id)?
            .map(|hash| self.load_manifest(hash))
            .transpose()
    }

    fn validate_match_id(&self, match_id: &str) -> Result<(), ReplayStoreError> {
        if match_id.is_empty()
            || match_id.len() > self.limits.max_match_id_bytes
            || !match_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(ReplayStoreError::InvalidMatchId);
        }
        Ok(())
    }

    fn object_path(&self, kind: &str, hash: CanonicalHash, extension: &str) -> PathBuf {
        let encoded = hash.to_string();
        self.root
            .join(kind)
            .join(&encoded[..2])
            .join(format!("{encoded}.{extension}"))
    }

    fn match_directory(&self, match_id: &str) -> Result<PathBuf, ReplayStoreError> {
        self.validate_match_id(match_id)?;
        Ok(self.root.join("matches").join(match_id))
    }

    fn current_pointer_path(&self, match_id: &str) -> Result<PathBuf, ReplayStoreError> {
        Ok(self.match_directory(match_id)?.join("CURRENT"))
    }

    fn read_bounded(
        &self,
        path: &Path,
        kind: &'static str,
        hash: CanonicalHash,
        limit: usize,
    ) -> Result<Vec<u8>, ReplayStoreError> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ReplayStoreError::MissingObject { kind, hash });
            }
            Err(error) => return Err(error.into()),
        };
        let length = file.metadata()?.len();
        if length > limit as u64 {
            return Err(ReplayStoreError::ObjectTooLarge {
                kind,
                actual: length,
                limit,
            });
        }
        let mut bytes = Vec::with_capacity(usize::try_from(length).unwrap_or(limit).min(limit));
        std::io::Read::by_ref(&mut file)
            .take((limit as u64).saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() > limit {
            return Err(ReplayStoreError::ObjectTooLarge {
                kind,
                actual: bytes.len() as u64,
                limit,
            });
        }
        Ok(bytes)
    }

    fn put_object(
        &self,
        path: &Path,
        bytes: &[u8],
        kind: &'static str,
        hash: CanonicalHash,
    ) -> Result<(), ReplayStoreError> {
        let parent = path
            .parent()
            .ok_or_else(|| ReplayStoreError::Io(std::io::Error::other("object has no parent")))?;
        fs::create_dir_all(parent)?;
        if path.exists() {
            let metadata = fs::metadata(path)?;
            if metadata.len() != bytes.len() as u64 {
                return Err(ReplayStoreError::ContentMismatch { kind, hash });
            }
            let mut existing = Vec::with_capacity(bytes.len());
            File::open(path)?
                .take((bytes.len() as u64).saturating_add(1))
                .read_to_end(&mut existing)?;
            return if existing == bytes {
                Ok(())
            } else {
                Err(ReplayStoreError::ContentMismatch { kind, hash })
            };
        }

        let temporary = temporary_path(parent, "object");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let result = (|| -> Result<(), ReplayStoreError> {
            file.write_all(bytes)?;
            file.sync_all()?;
            match fs::hard_link(&temporary, path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let metadata = fs::metadata(path)?;
                    if metadata.len() != bytes.len() as u64 {
                        return Err(ReplayStoreError::ContentMismatch { kind, hash });
                    }
                    let mut existing = Vec::with_capacity(bytes.len());
                    File::open(path)?
                        .take((bytes.len() as u64).saturating_add(1))
                        .read_to_end(&mut existing)?;
                    if existing != bytes {
                        return Err(ReplayStoreError::ContentMismatch { kind, hash });
                    }
                }
                Err(error) => return Err(error.into()),
            }
            sync_directory(parent)?;
            Ok(())
        })();
        let _ = fs::remove_file(&temporary);
        result
    }

    fn current_manifest_hash_unlocked(
        &self,
        match_id: &str,
    ) -> Result<Option<CanonicalHash>, ReplayStoreError> {
        let path = self.current_pointer_path(match_id)?;
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if file.metadata()?.len() != (8 + 2 + 32 + 32) as u64 {
            return Err(ReplayStoreError::InvalidPointer);
        }
        let mut bytes = Vec::with_capacity(8 + 2 + 32 + 32);
        file.read_to_end(&mut bytes)?;
        Ok(Some(decode_pointer(&bytes)?))
    }

    fn publish_pointer(
        &self,
        match_directory: &Path,
        manifest_hash: CanonicalHash,
    ) -> Result<(), ReplayStoreError> {
        let bytes = encode_pointer(manifest_hash);
        let temporary = temporary_path(match_directory, "pointer");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let result = (|| -> Result<(), ReplayStoreError> {
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&temporary, match_directory.join("CURRENT"))?;
            sync_directory(match_directory)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }
}

impl ReplayArtifactStore for FileReplayStore {
    fn put_segment(&self, segment: &ReplaySegment) -> Result<CanonicalHash, ReplayStoreError> {
        let hash = segment.segment_hash();
        let bytes = segment.to_bytes();
        if bytes.len() > self.limits.segment.max_segment_bytes {
            return Err(ReplayStoreError::ObjectTooLarge {
                kind: "replay segment",
                actual: bytes.len() as u64,
                limit: self.limits.segment.max_segment_bytes,
            });
        }
        ReplaySegment::from_bytes_with_limits(bytes, self.limits.segment).map_err(|error| {
            ReplayStoreError::CorruptObject {
                kind: "replay segment",
                hash,
                message: error.to_string(),
            }
        })?;
        self.put_object(
            &self.object_path("segments", hash, "blbseg"),
            bytes,
            "replay segment",
            hash,
        )?;
        Ok(hash)
    }

    fn load_segment(&self, hash: CanonicalHash) -> Result<ReplaySegment, ReplayStoreError> {
        let bytes = self.read_bounded(
            &self.object_path("segments", hash, "blbseg"),
            "replay segment",
            hash,
            self.limits.segment.max_segment_bytes,
        )?;
        let segment = ReplaySegment::from_bytes_with_limits(&bytes, self.limits.segment).map_err(
            |error| ReplayStoreError::CorruptObject {
                kind: "replay segment",
                hash,
                message: error.to_string(),
            },
        )?;
        if segment.segment_hash() != hash {
            return Err(ReplayStoreError::ContentMismatch {
                kind: "replay segment",
                hash,
            });
        }
        Ok(segment)
    }

    fn current_manifest_hash(
        &self,
        match_id: &str,
    ) -> Result<Option<CanonicalHash>, ReplayStoreError> {
        self.validate_match_id(match_id)?;
        self.current_manifest_hash_unlocked(match_id)
    }

    fn load_manifest(&self, hash: CanonicalHash) -> Result<ReplayManifest, ReplayStoreError> {
        let bytes = self.read_bounded(
            &self.object_path("manifests", hash, "blbman"),
            "replay manifest",
            hash,
            self.limits.manifest.max_manifest_bytes,
        )?;
        let manifest = ReplayManifest::from_bytes_with_limits(&bytes, self.limits.manifest)
            .map_err(|error| ReplayStoreError::CorruptObject {
                kind: "replay manifest",
                hash,
                message: error.to_string(),
            })?;
        if manifest.manifest_hash() != hash {
            return Err(ReplayStoreError::ContentMismatch {
                kind: "replay manifest",
                hash,
            });
        }
        Ok(manifest)
    }

    fn publish_manifest(
        &self,
        match_id: &str,
        manifest: &ReplayManifest,
        expected_previous: Option<CanonicalHash>,
    ) -> Result<ReplayPublishOutcome, ReplayStoreError> {
        ReplayManifest::from_bytes_with_limits(manifest.to_bytes(), self.limits.manifest).map_err(
            |error| ReplayStoreError::CorruptObject {
                kind: "replay manifest",
                hash: manifest.manifest_hash(),
                message: error.to_string(),
            },
        )?;
        let match_directory = self.match_directory(match_id)?;
        fs::create_dir_all(&match_directory)?;
        let _lock = PublishLock::acquire(match_directory.join(".publish-lock"))?;
        let actual_previous = self.current_manifest_hash_unlocked(match_id)?;
        if actual_previous != expected_previous {
            return Err(ReplayStoreError::PublishConflict {
                expected: expected_previous,
                actual: actual_previous,
            });
        }

        let previous_manifest = actual_previous
            .map(|hash| self.load_manifest(hash))
            .transpose()?;
        let previous_count = previous_manifest
            .as_ref()
            .map_or(0, ReplayManifest::segment_count);
        if let Some(previous) = &previous_manifest {
            if previous.compiled_ruleset_hash() != manifest.compiled_ruleset_hash()
                || previous.initial_state_hash() != manifest.initial_state_hash()
                || previous.segment_count() > manifest.segment_count()
                || !previous
                    .segments()
                    .zip(manifest.segments())
                    .all(|(left, right)| left == right)
            {
                return Err(ReplayStoreError::NonAppendOnlyManifest);
            }
        }

        for (index, descriptor) in manifest.segments().enumerate().skip(previous_count) {
            let segment = self.load_segment(descriptor.segment_hash)?;
            let actual_length = segment.to_bytes().len() as u64;
            if actual_length != descriptor.byte_length {
                return Err(ReplayStoreError::SegmentLengthMismatch {
                    hash: descriptor.segment_hash,
                    expected: descriptor.byte_length,
                    actual: actual_length,
                });
            }
            manifest.verify_segment(index, &segment).map_err(|error| {
                ReplayStoreError::CorruptObject {
                    kind: "replay segment",
                    hash: descriptor.segment_hash,
                    message: error.to_string(),
                }
            })?;
        }

        let manifest_hash = manifest.manifest_hash();
        self.put_object(
            &self.object_path("manifests", manifest_hash, "blbman"),
            manifest.to_bytes(),
            "replay manifest",
            manifest_hash,
        )?;
        self.publish_pointer(&match_directory, manifest_hash)?;
        Ok(ReplayPublishOutcome {
            previous_manifest: actual_previous,
            current_manifest: manifest_hash,
            appended_segments: manifest.segment_count() - previous_count,
        })
    }
}

struct PublishLock {
    path: PathBuf,
}

impl PublishLock {
    fn acquire(path: PathBuf) -> Result<Self, ReplayStoreError> {
        match fs::create_dir(&path) {
            Ok(()) => Ok(Self { path }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(ReplayStoreError::PublishBusy)
            }
            Err(error) => Err(error.into()),
        }
    }
}

impl Drop for PublishLock {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.path);
    }
}

fn temporary_path(parent: &Path, purpose: &str) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        ".{purpose}.{}.{}.tmp",
        std::process::id(),
        sequence
    ))
}

fn sync_directory(path: &Path) -> Result<(), ReplayStoreError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn encode_pointer(manifest_hash: CanonicalHash) -> Vec<u8> {
    let mut body = Vec::with_capacity(42);
    body.extend_from_slice(POINTER_MAGIC);
    body.extend_from_slice(&POINTER_VERSION.to_le_bytes());
    body.extend_from_slice(manifest_hash.as_bytes());
    let checksum = pointer_hash(&body);
    body.extend_from_slice(checksum.as_bytes());
    body
}

fn decode_pointer(bytes: &[u8]) -> Result<CanonicalHash, ReplayStoreError> {
    if bytes.len() != 8 + 2 + 32 + 32 || &bytes[..8] != POINTER_MAGIC {
        return Err(ReplayStoreError::InvalidPointer);
    }
    let version = u16::from_le_bytes(
        bytes[8..10]
            .try_into()
            .map_err(|_| ReplayStoreError::InvalidPointer)?,
    );
    if version != POINTER_VERSION {
        return Err(ReplayStoreError::InvalidPointer);
    }
    let expected = CanonicalHash::from_bytes(
        bytes[42..]
            .try_into()
            .map_err(|_| ReplayStoreError::InvalidPointer)?,
    );
    if pointer_hash(&bytes[..42]) != expected {
        return Err(ReplayStoreError::InvalidPointer);
    }
    Ok(CanonicalHash::from_bytes(
        bytes[10..42]
            .try_into()
            .map_err(|_| ReplayStoreError::InvalidPointer)?,
    ))
}

fn pointer_hash(body: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((POINTER_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(POINTER_DOMAIN);
    hasher.update(POINTER_VERSION.to_le_bytes());
    hasher.update((body.len() as u64).to_le_bytes());
    hasher.update(body);
    CanonicalHash::from_bytes(hasher.finalize().into())
}
