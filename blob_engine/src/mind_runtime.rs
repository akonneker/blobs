//! Structural admission for untrusted Mind WebAssembly artifacts.
//!
//! This validates the versioned capability profile without compiling or
//! executing the module. The execution host repeats its own link-time checks.

use std::error::Error;
use std::fmt::{Display, Formatter};

use wasmparser::{Encoding, ExternalKind, FuncType, Parser, Payload, TypeRef, ValType, Validator};

use crate::resolution::MindRuntimeProfile;

const EXTISM_ENV: &str = "extism:host/env";
const REFERENCE_EXPORT: &str = "reference_mind_function";
const MAX_MEMORIES: usize = 2;
const MAX_TABLES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MindArtifactInspection {
    pub function_imports: usize,
    pub memories: usize,
    pub tables: usize,
    pub has_start: bool,
    pub has_reactor_initialize: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MindRuntimeProfileError {
    InvalidWasm(String),
    ComponentEncoding,
    UnsupportedImport { module: String, name: String },
    UnsupportedImportKind { module: String, name: String },
    WrongFunctionType(String),
    MissingReferenceExport,
    UnsupportedMemory,
    TooManyMemories(usize),
    UnsupportedTable,
    TooManyTables(usize),
}

impl Display for MindRuntimeProfileError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWasm(error) => write!(formatter, "invalid WebAssembly: {error}"),
            Self::ComponentEncoding => write!(formatter, "WebAssembly components are not admitted"),
            Self::UnsupportedImport { module, name } => {
                write!(formatter, "unsupported Mind import {module}::{name}")
            }
            Self::UnsupportedImportKind { module, name } => {
                write!(formatter, "Mind import {module}::{name} is not a function")
            }
            Self::WrongFunctionType(name) => {
                write!(
                    formatter,
                    "Extism PDK import/export {name} has the wrong type"
                )
            }
            Self::MissingReferenceExport => {
                write!(
                    formatter,
                    "Mind does not export {REFERENCE_EXPORT}: () -> i32"
                )
            }
            Self::UnsupportedMemory => write!(
                formatter,
                "Mind memories must be unshared 32-bit memories with standard pages"
            ),
            Self::TooManyMemories(count) => {
                write!(
                    formatter,
                    "Mind declares {count} memories; limit is {MAX_MEMORIES}"
                )
            }
            Self::UnsupportedTable => {
                write!(formatter, "Mind tables must be unshared 32-bit tables")
            }
            Self::TooManyTables(count) => {
                write!(
                    formatter,
                    "Mind declares {count} tables; limit is {MAX_TABLES}"
                )
            }
        }
    }
}

impl Error for MindRuntimeProfileError {}

pub fn inspect_mind_artifact(
    bytes: &[u8],
    profile: MindRuntimeProfile,
) -> Result<MindArtifactInspection, MindRuntimeProfileError> {
    match profile {
        MindRuntimeProfile::ExtismPdkDeterministicV1 => inspect_extism_deterministic_v1(bytes),
    }
}

fn inspect_extism_deterministic_v1(
    bytes: &[u8],
) -> Result<MindArtifactInspection, MindRuntimeProfileError> {
    Validator::new()
        .validate_all(bytes)
        .map_err(|error| MindRuntimeProfileError::InvalidWasm(error.to_string()))?;

    let mut types = Vec::<FuncType>::new();
    let mut imported_function_types = Vec::<u32>::new();
    let mut defined_function_types = Vec::<u32>::new();
    let mut imports = Vec::<(String, u32)>::new();
    let mut reference_export = None;
    let mut reactor_initialize_export = None;
    let mut memories = 0_usize;
    let mut tables = 0_usize;
    let mut has_start = false;

    for payload in Parser::new(0).parse_all(bytes) {
        match payload.map_err(|error| MindRuntimeProfileError::InvalidWasm(error.to_string()))? {
            Payload::Version { encoding, .. } if encoding != Encoding::Module => {
                return Err(MindRuntimeProfileError::ComponentEncoding)
            }
            Payload::TypeSection(reader) => {
                for ty in reader.into_iter_err_on_gc_types() {
                    types.push(ty.map_err(|error| {
                        MindRuntimeProfileError::InvalidWasm(error.to_string())
                    })?);
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader {
                    let import = import
                        .map_err(|error| MindRuntimeProfileError::InvalidWasm(error.to_string()))?;
                    if import.module != EXTISM_ENV || expected_signature(import.name).is_none() {
                        return Err(MindRuntimeProfileError::UnsupportedImport {
                            module: import.module.into(),
                            name: import.name.into(),
                        });
                    }
                    let TypeRef::Func(type_index) = import.ty else {
                        return Err(MindRuntimeProfileError::UnsupportedImportKind {
                            module: import.module.into(),
                            name: import.name.into(),
                        });
                    };
                    imported_function_types.push(type_index);
                    imports.push((import.name.into(), type_index));
                }
            }
            Payload::FunctionSection(reader) => {
                for type_index in reader {
                    defined_function_types.push(type_index.map_err(|error| {
                        MindRuntimeProfileError::InvalidWasm(error.to_string())
                    })?);
                }
            }
            Payload::MemorySection(reader) => {
                for memory in reader {
                    let memory = memory
                        .map_err(|error| MindRuntimeProfileError::InvalidWasm(error.to_string()))?;
                    if memory.memory64 || memory.shared || memory.page_size_log2.is_some() {
                        return Err(MindRuntimeProfileError::UnsupportedMemory);
                    }
                    memories += 1;
                }
            }
            Payload::TableSection(reader) => {
                for table in reader {
                    let table = table
                        .map_err(|error| MindRuntimeProfileError::InvalidWasm(error.to_string()))?;
                    if table.ty.table64 || table.ty.shared {
                        return Err(MindRuntimeProfileError::UnsupportedTable);
                    }
                    tables += 1;
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export
                        .map_err(|error| MindRuntimeProfileError::InvalidWasm(error.to_string()))?;
                    if export.name == REFERENCE_EXPORT && export.kind == ExternalKind::Func {
                        reference_export = Some(export.index);
                    }
                    if export.name == "_initialize" && export.kind == ExternalKind::Func {
                        reactor_initialize_export = Some(export.index);
                    }
                }
            }
            Payload::StartSection { .. } => has_start = true,
            _ => {}
        }
    }

    if memories > MAX_MEMORIES {
        return Err(MindRuntimeProfileError::TooManyMemories(memories));
    }
    if tables > MAX_TABLES {
        return Err(MindRuntimeProfileError::TooManyTables(tables));
    }
    for (name, type_index) in &imports {
        let ty = types
            .get(*type_index as usize)
            .ok_or_else(|| MindRuntimeProfileError::WrongFunctionType(name.clone()))?;
        let (params, results) = expected_signature(name).expect("import name was admitted");
        if ty.params() != params || ty.results() != results {
            return Err(MindRuntimeProfileError::WrongFunctionType(name.clone()));
        }
    }

    let function_type = |function_index: u32| {
        if let Some(type_index) = imported_function_types.get(function_index as usize) {
            types.get(*type_index as usize)
        } else {
            let defined_index = function_index as usize - imported_function_types.len();
            defined_function_types
                .get(defined_index)
                .and_then(|type_index| types.get(*type_index as usize))
        }
    };
    let reference_export =
        reference_export.ok_or(MindRuntimeProfileError::MissingReferenceExport)?;
    let ty =
        function_type(reference_export).ok_or(MindRuntimeProfileError::MissingReferenceExport)?;
    if !ty.params().is_empty() || ty.results() != [ValType::I32] {
        return Err(MindRuntimeProfileError::MissingReferenceExport);
    }
    if let Some(initialize) = reactor_initialize_export {
        let ty = function_type(initialize)
            .ok_or_else(|| MindRuntimeProfileError::WrongFunctionType("_initialize".into()))?;
        if !ty.params().is_empty() || !ty.results().is_empty() {
            return Err(MindRuntimeProfileError::WrongFunctionType(
                "_initialize".into(),
            ));
        }
    }

    Ok(MindArtifactInspection {
        function_imports: imports.len(),
        memories,
        tables,
        has_start,
        has_reactor_initialize: reactor_initialize_export.is_some(),
    })
}

fn expected_signature(name: &str) -> Option<(&'static [ValType], &'static [ValType])> {
    const NONE: &[ValType] = &[];
    const I32: &[ValType] = &[ValType::I32];
    const I64: &[ValType] = &[ValType::I64];
    const I64_I32: &[ValType] = &[ValType::I64, ValType::I32];
    const I64_I64: &[ValType] = &[ValType::I64, ValType::I64];
    match name {
        "input_length" => Some((NONE, I64)),
        "input_load_u8" => Some((I64, I32)),
        "input_load_u64" | "length" | "length_unsafe" | "alloc" | "load_u64" => Some((I64, I64)),
        "load_u8" => Some((I64, I32)),
        "free" | "error_set" => Some((I64, NONE)),
        "store_u8" => Some((I64_I32, NONE)),
        "store_u64" | "output_set" => Some((I64_I64, NONE)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_mind() -> Vec<u8> {
        let mut bytes = vec![
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // module header
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, // () -> i32
            0x03, 0x02, 0x01, 0x00, // one function of type zero
            0x07, 0x1b, 0x01, 0x17, // one 23-byte export name
        ];
        bytes.extend_from_slice(b"reference_mind_function");
        bytes.extend_from_slice(&[
            0x00, 0x00, // function export, function zero
            0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x00, 0x0b, // return zero
        ]);
        bytes
    }

    #[test]
    fn admits_the_minimal_profile_and_rejects_non_wasm() {
        let inspection = inspect_mind_artifact(
            &minimal_mind(),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
        )
        .unwrap();
        assert_eq!(inspection.function_imports, 0);
        assert!(!inspection.has_start);
        assert!(!inspection.has_reactor_initialize);
        assert!(matches!(
            inspect_mind_artifact(b"not wasm", MindRuntimeProfile::ExtismPdkDeterministicV1),
            Err(MindRuntimeProfileError::InvalidWasm(_))
        ));
    }
}
