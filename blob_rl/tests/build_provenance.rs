#[path = "../build_support.rs"]
mod build_support;
use std::path::PathBuf;

#[test]
fn provenance_tracks_uncommitted_content_and_is_checkout_location_independent() {
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();
    for root in [left.path(), right.path()] {
        std::fs::create_dir(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "original").unwrap();
    }
    let digest = |root| {
        let files = build_support::source_files(root, &[PathBuf::from("src")]).unwrap();
        build_support::source_digest(root, &files).unwrap()
    };
    let original = digest(left.path());
    assert_eq!(original, digest(right.path()));
    // No Git metadata exists or changes during these edits.
    std::fs::write(left.path().join("src/lib.rs"), "modified").unwrap();
    assert_ne!(original, digest(left.path()));
    std::fs::write(left.path().join("src/lib.rs"), "original").unwrap();
    std::fs::write(left.path().join("src/new.rs"), "new").unwrap();
    assert_ne!(original, digest(left.path()));
    std::fs::remove_file(left.path().join("src/new.rs")).unwrap();
    assert_eq!(original, digest(left.path()));
}
