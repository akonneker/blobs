use std::env;
use std::path::Path;
use std::process::Command;

fn command_output(program: &str, args: &[&str], workspace: &Path) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(workspace)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=BLOB_CODE_REVISION");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");

    if env::var_os("BLOB_CODE_REVISION").is_some() {
        return;
    }

    let workspace = Path::new("..");
    let Some(mut revision) = command_output("git", &["rev-parse", "HEAD"], workspace) else {
        return;
    };
    if revision.is_empty() {
        return;
    }

    let dirty = Command::new("git")
        .args([
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
        ])
        .current_dir(workspace)
        .output()
        .is_ok_and(|output| output.status.success() && !output.stdout.is_empty());
    if dirty {
        revision.push_str("+dirty");
    }
    println!("cargo:rustc-env=BLOB_CODE_REVISION={revision}");
}
