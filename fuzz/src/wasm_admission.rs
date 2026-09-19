//! Structural admission fuzzing; this never compiles or executes guest code.

use blob_engine::mind_runtime::{MindArtifactInspection, inspect_mind_artifact};
use blob_engine::resolution::MindRuntimeProfile;

const PROFILE: MindRuntimeProfile = MindRuntimeProfile::ExtismPdkDeterministicV1;
const HEADER: &[u8] = b"\0asm\x01\0\0\0";
const I32: u8 = 0x7f;
const I64: u8 = 0x7e;

// The published deterministic PDK contract, independent of the admission
// implementation's private signature lookup. Unknown capabilities are forbidden.
const IMPORTS: &[(&str, &[u8], &[u8])] = &[
    ("input_length", &[], &[I64]),
    ("input_load_u8", &[I64], &[I32]),
    ("input_load_u64", &[I64], &[I64]),
    ("alloc", &[I64], &[I64]),
    ("free", &[I64], &[]),
    ("length", &[I64], &[I64]),
    ("length_unsafe", &[I64], &[I64]),
    ("load_u8", &[I64], &[I32]),
    ("load_u64", &[I64], &[I64]),
    ("error_set", &[I64], &[]),
    ("store_u8", &[I64, I32], &[]),
    ("store_u64", &[I64, I64], &[]),
    ("output_set", &[I64, I64], &[]),
];

fn leb(mut value: usize, bytes: &mut Vec<u8>) {
    loop {
        let byte = (value & 127) as u8;
        value >>= 7;
        bytes.push(byte | if value == 0 { 0 } else { 128 });
        if value == 0 {
            break;
        }
    }
}

fn name(value: &str, bytes: &mut Vec<u8>) {
    leb(value.len(), bytes);
    bytes.extend_from_slice(value.as_bytes());
}

fn section(id: u8, contents: &[u8], bytes: &mut Vec<u8>) {
    bytes.push(id);
    leb(contents.len(), bytes);
    bytes.extend_from_slice(contents);
}

fn func_type(params: &[u8], results: &[u8], bytes: &mut Vec<u8>) {
    bytes.push(0x60);
    leb(params.len(), bytes);
    bytes.extend_from_slice(params);
    leb(results.len(), bytes);
    bytes.extend_from_slice(results);
}

/// Generate small valid core modules with known admission outcomes. Negative
/// cases each alter one profile constraint; they are not malformed Wasm noise.
pub fn module(selector: &[u8; 4]) -> (Vec<u8>, Option<MindArtifactInspection>) {
    let mode = selector[0] % 16;
    let (import_name, params, results) = IMPORTS[usize::from(selector[1]) % IMPORTS.len()];
    let mut params = params.to_vec();
    let mut results = results.to_vec();
    if mode == 3 {
        params.push(I32);
    }
    if mode == 4 {
        results = if results.is_empty() {
            vec![I32]
        } else {
            vec![]
        };
    }
    let has_import = mode != 14;
    let function_imports = usize::from(has_import && mode != 5 && mode != 15);
    let has_start = selector[2] & 1 != 0;
    let has_initialize = selector[2] & 2 != 0 || mode == 8;
    let memories = if mode == 9 {
        3
    } else if mode == 11 || mode == 12 {
        1
    } else {
        usize::from(selector[3] % 3)
    };
    let tables = if mode == 10 {
        5
    } else {
        usize::from((selector[3] / 3) % 5)
    };
    let mut bytes = HEADER.to_vec();
    let mut types = vec![3];
    func_type(&[], &[I32], &mut types);
    func_type(&[], &[], &mut types);
    func_type(&params, &results, &mut types);
    section(1, &types, &mut bytes);
    if has_import {
        let mut imports = vec![1];
        name(
            if mode == 1 {
                "wasi_snapshot_preview1"
            } else {
                "extism:host/env"
            },
            &mut imports,
        );
        name(
            if mode == 2 { "config_get" } else { import_name },
            &mut imports,
        );
        match mode {
            5 => imports.extend_from_slice(&[2, 0, 0]), // memory import, min 0
            15 => imports.extend_from_slice(&[1, 0x70, 0, 0]), // table import
            _ => imports.extend_from_slice(&[0, 2]),    // function of type 2
        }
        section(2, &imports, &mut bytes);
    }
    section(3, &[2, 0, 1], &mut bytes); // reference and empty initializer
    if tables > 0 {
        let mut table_section = vec![tables as u8];
        for _ in 0..tables {
            table_section.extend_from_slice(&[0x70, 0, 0]);
        }
        section(4, &table_section, &mut bytes);
    }
    if memories > 0 {
        let mut memory_section = vec![memories as u8];
        for _ in 0..memories {
            memory_section.extend_from_slice(match mode {
                11 => &[4, 0],    // 64-bit memory
                12 => &[3, 0, 1], // shared, max 1
                _ => &[0, 0],
            });
        }
        section(5, &memory_section, &mut bytes);
    }
    let mut exports = vec![1 + u8::from(has_initialize)];
    name(
        if mode == 6 {
            "other_function"
        } else {
            "reference_mind_function"
        },
        &mut exports,
    );
    exports.push(0);
    leb(
        if mode == 13 {
            0
        } else {
            function_imports + usize::from(mode == 7)
        },
        &mut exports,
    );
    if has_initialize {
        name("_initialize", &mut exports);
        exports.push(0);
        leb(function_imports + usize::from(mode != 8), &mut exports);
    }
    section(7, &exports, &mut bytes);
    if has_start {
        section(8, &[(function_imports + 1) as u8], &mut bytes);
    }
    section(10, &[2, 4, 0, 0x41, 0, 0x0b, 2, 0, 0x0b], &mut bytes);
    let expected = matches!(mode, 0 | 14).then_some(MindArtifactInspection {
        function_imports,
        memories,
        tables,
        has_start,
        has_reactor_initialize: has_initialize,
    });
    (bytes, expected)
}

fn inspect_with_custom_section(bytes: &[u8], payload: &[u8]) {
    if let Ok(expected) = inspect_mind_artifact(bytes, PROFILE) {
        let mut extended = bytes.to_vec();
        let mut custom = Vec::new();
        name("fuzz-metadata", &mut custom);
        custom.extend_from_slice(&payload[..payload.len().min(64)]);
        section(0, &custom, &mut extended);
        assert_eq!(inspect_mind_artifact(&extended, PROFILE), Ok(expected));
    }
}

pub fn check(data: &[u8]) {
    if data.len() > 8192 {
        return;
    }
    inspect_with_custom_section(data, data);
    // Repair the header to reach section parsing even after header mutations.
    let mut repaired = HEADER.to_vec();
    repaired.extend_from_slice(data.get(8..).unwrap_or_default());
    inspect_with_custom_section(&repaired, data);
    let mut selector = [0; 4];
    for (to, from) in selector.iter_mut().zip(data) {
        *to = *from;
    }
    let (bytes, expected) = module(&selector);
    let actual = inspect_mind_artifact(&bytes, PROFILE);
    assert_eq!(
        actual.as_ref().ok(),
        expected.as_ref(),
        "selector {selector:?}: {actual:?}"
    );
    inspect_with_custom_section(&bytes, data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_admission_contract() {
        for mode in 0..16 {
            for import in 0..IMPORTS.len() as u8 {
                for flags in 0..4 {
                    for limits in 0..15 {
                        let selector = [mode, import, flags, limits];
                        let (bytes, _) = module(&selector);
                        wasmparser::Validator::new().validate_all(&bytes).unwrap();
                        check(&selector);
                    }
                }
            }
        }
    }
}
