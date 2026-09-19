//! Bounded artifact reads use one opened file and enforce the limit on bytes read.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub(crate) fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let length = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > max_bytes {
        return Err(format!(
            "{} is {length} bytes; limit is {max_bytes}",
            path.display()
        ));
    }
    read_stream_bounded(file, max_bytes)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn read_stream_bounded(reader: impl Read, max_bytes: u64) -> io::Result<Vec<u8>> {
    let probe_limit = max_bytes.checked_add(1).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "byte limit cannot be u64::MAX")
    })?;
    let mut bytes = Vec::new();
    reader.take(probe_limit).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("artifact exceeds {max_bytes}-byte limit while reading"),
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::fs;

    #[test]
    fn reads_exact_limit_and_rejects_one_extra_byte() {
        for limit in [0, 1, 16, 65536] {
            let exact = vec![7; limit];
            assert_eq!(
                read_stream_bounded(exact.as_slice(), limit as u64).unwrap(),
                exact
            );
            assert_eq!(
                read_stream_bounded(vec![7; limit + 1].as_slice(), limit as u64)
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidData
            );
        }
        assert_eq!(
            read_stream_bounded(io::empty(), u64::MAX)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn growing_or_unending_stream_consumes_only_the_limit_probe() {
        struct Endless<'a>(&'a Cell<usize>);
        impl Read for Endless<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                buffer.fill(1);
                self.0.set(self.0.get() + buffer.len());
                Ok(buffer.len())
            }
        }
        let consumed = Cell::new(0);
        assert!(read_stream_bounded(Endless(&consumed), 16).is_err());
        assert_eq!(consumed.get(), 17);

        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("growing.json");
        fs::write(&path, b"{}").unwrap();
        let file = File::open(&path).unwrap();
        assert!(file.metadata().unwrap().len() <= 16);
        // Deterministically grow the same inode after its size was inspected.
        fs::write(&path, [1; 64]).unwrap();
        assert_eq!(
            read_stream_bounded(file, 16).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert!(read_bounded(&path, 16).unwrap_err().contains("limit is 16"));
    }

    #[test]
    fn opened_file_is_not_retargeted_by_path_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("artifact.json");
        fs::write(&path, b"old").unwrap();
        let opened = File::open(&path).unwrap();
        let replacement = temporary.path().join("replacement.json");
        fs::write(&replacement, b"new").unwrap();
        fs::rename(replacement, &path).unwrap();
        assert_eq!(read_stream_bounded(opened, 3).unwrap(), b"old");
        assert_eq!(read_bounded(&path, 3).unwrap(), b"new");
        assert!(read_bounded(&temporary.path().join("missing"), 3).is_err());
    }
}
