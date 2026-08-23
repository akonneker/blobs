#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::PathBuf;

use blob_engine::mind_runtime::inspect_mind_artifact;
use blob_engine::resolution::MindRuntimeProfile;

fn workspace_artifact(relative: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("blob_engine must be in the workspace")
        .join(relative);
    fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "required language-PDK artifact {} is unavailable: {error}",
            path.display()
        )
    })
}

#[test]
#[ignore = "requires npm-built AssemblyScript conformance artifact"]
fn assemblyscript_pdk_artifact_matches_the_deterministic_profile() {
    let artifact =
        workspace_artifact("minds/assemblyscript_wait_mind/build/assemblyscript_wait_mind.wasm");
    let inspection = inspect_mind_artifact(&artifact, MindRuntimeProfile::ExtismPdkDeterministicV1)
        .expect("official AssemblyScript PDK artifact must be admitted");

    assert!(inspection.function_imports > 0);
    assert!(inspection.memories > 0);
}

#[test]
#[ignore = "requires TinyGo-built Go conformance artifact"]
fn go_pdk_artifact_matches_the_deterministic_profile() {
    let artifact = workspace_artifact("minds/go_wait_mind/build/go_wait_mind.wasm");
    let inspection = inspect_mind_artifact(&artifact, MindRuntimeProfile::ExtismPdkDeterministicV1)
        .expect("official Go PDK artifact must be admitted");

    assert!(inspection.function_imports > 0);
    assert!(inspection.memories > 0);
    assert!(inspection.has_reactor_initialize);
}
