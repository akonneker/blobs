//! Generate both raw Wasm and structured selectors without deleting discoveries.

use std::{fs, path::Path};

fn main() -> std::io::Result<()> {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/wasm_admission");
    fs::create_dir_all(&corpus)?;
    for mode in 0..16 {
        for import in 0..13 {
            for (flags, limits) in [(0, 0), (3, 14)] {
                let selector = [mode, import, flags, limits];
                let (bytes, _) = blob_fuzz::wasm_admission::module(&selector);
                let stem = format!("{mode}-{import}-{flags}-{limits}");
                fs::write(corpus.join(format!("module-{stem}")), bytes)?;
                fs::write(corpus.join(format!("selector-{stem}")), selector)?;
            }
        }
    }
    Ok(())
}
