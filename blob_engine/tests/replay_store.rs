#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_engine::replay_store::{
    FileReplayStore, ReplayArtifactStore, ReplayStoreError, ReplayStoreLimits,
};
use blob_engine::resolution::{
    ActionRequest, DurationRule, ReferenceCheckpoint, ReferenceRuleset, ReferenceSimulation,
    ReplayBatchEvent, ReplayManifest, ReplayManifestLimits, ReplayRecorder, ReplaySegment,
    ReplaySegmentDescriptor, ReplaySegmentLimits,
};

static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "blob-replay-store-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn uniform_rules() -> ReferenceRuleset {
    let duration = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        wait_duration: duration,
        move_duration: duration,
        attack_duration: duration,
        consume_duration: duration,
        split_duration: duration,
        regurgitate_duration: duration,
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

fn segments(initial_energy: u64) -> (ReplaySegment, ReplaySegment) {
    let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
    let left = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), initial_energy, 100, 1)
        .unwrap();
    let right = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 2)
        .unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    let checkpoint_zero = ReferenceCheckpoint::from_simulation(&simulation);
    let mut recorder = ReplayRecorder::new(compiled, initial);
    let cursor_zero = recorder.cursor();
    let mut events: Vec<ReplayBatchEvent> = Vec::new();
    let mut checkpoint_two = None;

    for index in 0..4 {
        simulation.commit_action(left, ActionRequest::Wait).unwrap();
        simulation
            .commit_action(right, ActionRequest::Wait)
            .unwrap();
        events.push(
            recorder
                .record(&simulation.resolve_next_batch().unwrap())
                .unwrap(),
        );
        if index == 1 {
            checkpoint_two = Some(ReferenceCheckpoint::from_simulation(&simulation));
        }
    }

    let first = ReplaySegment::from_events(
        compiled,
        cursor_zero,
        checkpoint_zero,
        &events[..2],
        ReplaySegmentLimits::default(),
    )
    .unwrap();
    let second = ReplaySegment::from_events(
        compiled,
        first.end_cursor(),
        checkpoint_two.unwrap(),
        &events[2..],
        ReplaySegmentLimits::default(),
    )
    .unwrap();
    (first, second)
}

fn manifest(segments: &[&ReplaySegment]) -> ReplayManifest {
    let first = segments[0];
    ReplayManifest::from_segments(
        first.compiled_ruleset_hash(),
        first.start_cursor().previous_state_hash,
        segments
            .iter()
            .map(|segment| ReplaySegmentDescriptor::from_segment(segment))
            .collect(),
        ReplayManifestLimits::default(),
    )
    .unwrap()
}

#[test]
fn filesystem_store_publishes_verified_append_only_manifests() {
    let directory = TestDirectory::new();
    let store = FileReplayStore::open(directory.path(), ReplayStoreLimits::default()).unwrap();
    let (first, second) = segments(10);
    let first_manifest = manifest(&[&first]);
    let second_manifest = manifest(&[&first, &second]);

    assert!(matches!(
        store.current_manifest_hash("../escape"),
        Err(ReplayStoreError::InvalidMatchId)
    ));
    assert_eq!(store.put_segment(&first).unwrap(), first.segment_hash());
    assert_eq!(store.put_segment(&first).unwrap(), first.segment_hash());
    let published = store
        .publish_manifest("match_1", &first_manifest, None)
        .unwrap();
    assert_eq!(published.previous_manifest, None);
    assert_eq!(published.current_manifest, first_manifest.manifest_hash());
    assert_eq!(published.appended_segments, 1);
    assert_eq!(
        store.current_manifest_hash("match_1").unwrap(),
        Some(first_manifest.manifest_hash())
    );
    assert_eq!(
        store
            .current_manifest("match_1")
            .unwrap()
            .unwrap()
            .to_bytes(),
        first_manifest.to_bytes()
    );

    assert!(matches!(
        store.publish_manifest(
            "match_1",
            &second_manifest,
            Some(first_manifest.manifest_hash()),
        ),
        Err(ReplayStoreError::MissingObject { .. })
    ));
    assert_eq!(
        store.current_manifest_hash("match_1").unwrap(),
        Some(first_manifest.manifest_hash())
    );

    store.put_segment(&second).unwrap();
    let published = store
        .publish_manifest(
            "match_1",
            &second_manifest,
            Some(first_manifest.manifest_hash()),
        )
        .unwrap();
    assert_eq!(published.appended_segments, 1);
    assert_eq!(published.current_manifest, second_manifest.manifest_hash());
    assert!(matches!(
        store.publish_manifest(
            "match_1",
            &second_manifest,
            Some(first_manifest.manifest_hash()),
        ),
        Err(ReplayStoreError::PublishConflict { .. })
    ));
    let no_op = store
        .publish_manifest(
            "match_1",
            &second_manifest,
            Some(second_manifest.manifest_hash()),
        )
        .unwrap();
    assert_eq!(no_op.appended_segments, 0);

    let reopened = FileReplayStore::open(directory.path(), ReplayStoreLimits::default()).unwrap();
    assert_eq!(
        reopened.current_manifest_hash("match_1").unwrap(),
        Some(second_manifest.manifest_hash())
    );
    let loaded_second = reopened.load_segment(second.segment_hash()).unwrap();
    assert_eq!(loaded_second.to_bytes(), second.to_bytes());
}

#[test]
fn filesystem_store_rejects_history_rewrites_busy_publishers_and_corruption() {
    let directory = TestDirectory::new();
    let store = FileReplayStore::open(directory.path(), ReplayStoreLimits::default()).unwrap();
    let (first, _) = segments(10);
    let (alternate, _) = segments(11);
    let first_manifest = manifest(&[&first]);
    let alternate_manifest = manifest(&[&alternate]);
    store.put_segment(&first).unwrap();
    store.put_segment(&alternate).unwrap();
    store
        .publish_manifest("match-2", &first_manifest, None)
        .unwrap();

    assert!(matches!(
        store.publish_manifest(
            "match-2",
            &alternate_manifest,
            Some(first_manifest.manifest_hash()),
        ),
        Err(ReplayStoreError::NonAppendOnlyManifest)
    ));

    let lock = directory
        .path()
        .join("matches")
        .join("match-2")
        .join(".publish-lock");
    fs::create_dir(&lock).unwrap();
    assert!(matches!(
        store.publish_manifest(
            "match-2",
            &first_manifest,
            Some(first_manifest.manifest_hash()),
        ),
        Err(ReplayStoreError::PublishBusy)
    ));
    fs::remove_dir(lock).unwrap();

    let encoded = first.segment_hash().to_string();
    let segment_path = directory
        .path()
        .join("segments")
        .join(&encoded[..2])
        .join(format!("{encoded}.blbseg"));
    let mut bytes = fs::read(&segment_path).unwrap();
    bytes[32] ^= 0x80;
    fs::write(&segment_path, bytes).unwrap();
    assert!(matches!(
        store.load_segment(first.segment_hash()),
        Err(ReplayStoreError::CorruptObject { .. })
    ));

    let pointer = directory
        .path()
        .join("matches")
        .join("match-2")
        .join("CURRENT");
    fs::write(pointer, b"invalid").unwrap();
    assert!(matches!(
        store.current_manifest_hash("match-2"),
        Err(ReplayStoreError::InvalidPointer)
    ));
}

#[test]
fn filesystem_store_revalidates_artifacts_under_its_own_limits() {
    let directory = TestDirectory::new();
    let (first, second) = segments(10);
    let restrictive = ReplayStoreLimits {
        segment: ReplaySegmentLimits {
            max_events: 1,
            ..ReplaySegmentLimits::default()
        },
        manifest: ReplayManifestLimits {
            max_segments: 1,
            ..ReplayManifestLimits::default()
        },
        ..ReplayStoreLimits::default()
    };
    let store = FileReplayStore::open(directory.path(), restrictive).unwrap();
    assert!(matches!(
        store.put_segment(&first),
        Err(ReplayStoreError::CorruptObject { .. })
    ));

    let default_store =
        FileReplayStore::open(directory.path(), ReplayStoreLimits::default()).unwrap();
    default_store.put_segment(&first).unwrap();
    default_store.put_segment(&second).unwrap();
    let two_segment_manifest = manifest(&[&first, &second]);
    assert!(matches!(
        store.publish_manifest("limited", &two_segment_manifest, None),
        Err(ReplayStoreError::CorruptObject { .. })
    ));
    assert_eq!(store.current_manifest_hash("limited").unwrap(), None);
}
