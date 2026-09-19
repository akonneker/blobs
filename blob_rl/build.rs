mod build_support;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .to_owned();
    println!("cargo:rerun-if-env-changed=BLOB_CODE_REVISION");
    // Resolve indirection for worktrees and packed branches as well as ordinary checkouts.
    for name in [
        "HEAD".to_owned(),
        "index".to_owned(),
        "packed-refs".to_owned(),
    ]
    .into_iter()
    .chain(git(&root, &["symbolic-ref", "-q", "HEAD"]))
    {
        if let Some(path) = git(&root, &["rev-parse", "--git-path", &name]) {
            let path = root.join(path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    if root.join(".git").is_file() {
        println!("cargo:rerun-if-changed={}", root.join(".git").display());
    }
    let mut inputs: Vec<PathBuf> = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".cargo",
        "blob_rl/build.rs",
        "blob_rl/build_support.rs",
        "blob_rl/config",
        "blob_interface/interface",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect();
    let crates = ["blob_engine", "blob_interface", "blob_rl", "blob_policy"]
        .into_iter()
        .map(PathBuf::from)
        .chain(
            fs::read_dir(root.join("minds"))
                .unwrap()
                .map(|entry| PathBuf::from("minds").join(entry.unwrap().file_name())),
        );
    for directory in crates {
        inputs.push(directory.join("Cargo.toml"));
        inputs.push(directory.join("build.rs"));
        inputs.push(directory.join("src"));
    }
    for input in inputs.iter().filter(|input| root.join(input).exists()) {
        println!("cargo:rerun-if-changed={}", root.join(input).display());
    }
    let files =
        build_support::source_files(&root, &inputs).expect("cannot enumerate training sources");
    let digest = build_support::source_digest(&root, &files).expect("cannot hash training sources");
    let revision = env::var("BLOB_CODE_REVISION")
        .ok()
        .or_else(|| git(&root, &["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unversioned".into());
    // Always bind actual content, including dirty edits and builds without .git.
    println!("cargo:rustc-env=BLOB_CODE_REVISION={revision}+source:{digest}");
}
