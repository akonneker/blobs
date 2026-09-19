use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub fn source_files(root: &Path, inputs: &[PathBuf]) -> io::Result<Vec<PathBuf>> {
    fn visit(root: &Path, relative: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
        let path = root.join(relative);
        if !path.exists() {
            return Ok(());
        }
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                visit(root, &relative.join(entry?.file_name()), files)?;
            }
        } else {
            files.push(relative.to_owned());
        }
        Ok(())
    }
    let mut files = Vec::new();
    for input in inputs {
        visit(root, input, &mut files)?;
    }
    files.sort();
    files.dedup();
    Ok(files)
}

pub fn source_digest(root: &Path, files: &[PathBuf]) -> io::Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"blob.training.source-tree.v1");
    for relative in files {
        let name = relative.to_string_lossy().replace('\\', "/");
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        let mut file = fs::File::open(root.join(relative))?;
        hash.update(file.metadata()?.len().to_le_bytes());
        let mut buffer = [0; 16 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
