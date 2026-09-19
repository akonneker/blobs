use std::{env, fs, io::Read, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=BLOB_POLICY_WEIGHTS");
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("wasm32") {
        return;
    }
    let path = PathBuf::from(env::var_os("BLOB_POLICY_WEIGHTS").expect(
        "set BLOB_POLICY_WEIGHTS to an exported weights.bin; use scripts/build_learned_mind.py",
    ));
    println!("cargo:rerun-if-changed={}", path.display());
    let mut bytes = Vec::new();
    fs::File::open(&path)
        .unwrap()
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(
        bytes.len() <= 16 * 1024 * 1024
            && (bytes.starts_with(b"BLPOL001")
                || bytes.starts_with(b"BLCMP001")
                || bytes.starts_with(b"BLCMR001")),
        "invalid deployment weights"
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("weights.bin"),
        bytes,
    )
    .unwrap();
}
