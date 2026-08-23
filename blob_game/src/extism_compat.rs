//! Restricted executor for the byte-oriented Extism PDK contract used by Minds.
//!
//! This deliberately implements only the deterministic input/output functions
//! imported by admitted Mind artifacts. Every call creates a fresh Store,
//! host byte arena, and guest instance; only immutable compiled code and
//! store-independent host functions are reused.

use std::fs;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use blob_engine::mind_runtime::inspect_mind_artifact;
use blob_engine::resolution::MindRuntimeProfile;
use extism::{Manifest, Wasm};
use smallvec::SmallVec;
use wasmtime::{
    Caller, Config, Engine, ExternType, InstancePre, Linker, Module, Store, StoreLimits,
    StoreLimitsBuilder, ValType,
};

const EXTISM_ENV: &str = "extism:host/env";
const REFERENCE_EXPORT: &str = "reference_mind_function";
const WASM_PAGE_BYTES: u64 = 65_536;
const EPOCH_TICK: Duration = Duration::from_millis(10);

#[derive(Default)]
struct CompatState {
    input: Vec<u8>,
    // Normal Mind decisions fit in these inline buffers. Larger or unusually
    // allocation-heavy PDK calls transparently spill to the heap, preserving
    // the full bounded Extism contract without imposing heap churn on every
    // ordinary pristine invocation.
    host_memory: SmallVec<[u8; 128]>,
    allocations: SmallVec<[(u64, u64); 4]>,
    free_ranges: SmallVec<[(u64, u64); 4]>,
    output: Option<(u64, u64)>,
    error_offset: Option<u64>,
    max_host_memory_bytes: usize,
    limits: StoreLimits,
}

impl CompatState {
    fn new(input: Vec<u8>, max_memory_bytes: usize) -> Self {
        let mut host_memory = SmallVec::new();
        host_memory.push(0);
        Self {
            input,
            host_memory,
            max_host_memory_bytes: max_memory_bytes,
            limits: StoreLimitsBuilder::new()
                .memory_size(max_memory_bytes)
                .instances(2)
                .memories(2)
                .tables(4)
                .build(),
            ..Self::default()
        }
    }
}

pub(crate) struct ExtismCompatExecutor {
    engine: Engine,
    instance_pre: InstancePre<CompatState>,
    max_memory_bytes: usize,
    timeout_ticks: u64,
    has_reactor_initialize: bool,
    ticker_stop: Option<mpsc::Sender<()>>,
    ticker: Option<JoinHandle<()>>,
}

impl ExtismCompatExecutor {
    pub(crate) fn new(manifest: &Manifest, mut config: Config) -> Result<Self, String> {
        let max_pages = manifest
            .memory
            .max_pages
            .ok_or("Extism-compatible executor requires a bounded memory limit")?;
        let max_memory_bytes = usize::try_from(max_pages)
            .ok()
            .and_then(|pages| pages.checked_mul(WASM_PAGE_BYTES as usize))
            .ok_or("Extism-compatible memory limit does not fit this platform")?;

        let wasm = load_single_module(manifest)?;
        let inspection = inspect_mind_artifact(&wasm, MindRuntimeProfile::ExtismPdkDeterministicV1)
            .map_err(|error| error.to_string())?;
        config.epoch_interruption(true);
        let engine = Engine::new(&config).map_err(|error| error.to_string())?;
        let module = Module::new(&engine, wasm).map_err(|error| error.to_string())?;
        validate_imports(&module)?;
        let linker = compat_linker(&engine)?;
        let instance_pre = linker
            .instantiate_pre(&module)
            .map_err(|error| error.to_string())?;

        let timeout_ms = manifest.timeout_ms.unwrap_or(1_000).max(1);
        let timeout_ticks = timeout_ms.div_ceil(EPOCH_TICK.as_millis() as u64).max(1);
        let (ticker_stop, ticker_rx) = mpsc::channel();
        let ticker_engine = engine.clone();
        let ticker = std::thread::spawn(move || {
            while let Err(mpsc::RecvTimeoutError::Timeout) = ticker_rx.recv_timeout(EPOCH_TICK) {
                ticker_engine.increment_epoch();
            }
        });

        Ok(Self {
            engine,
            instance_pre,
            max_memory_bytes,
            timeout_ticks,
            has_reactor_initialize: inspection.has_reactor_initialize,
            ticker_stop: Some(ticker_stop),
            ticker: Some(ticker),
        })
    }

    pub(crate) fn call(&self, input: Vec<u8>, max_output_bytes: usize) -> Result<Vec<u8>, String> {
        let mut store = Store::new(&self.engine, CompatState::new(input, self.max_memory_bytes));
        store.limiter(|state| &mut state.limits);
        store.epoch_deadline_trap();
        store.set_epoch_deadline(self.timeout_ticks);

        let instance = self
            .instance_pre
            .instantiate(&mut store)
            .map_err(|error| error.to_string())?;
        if self.has_reactor_initialize {
            instance
                .get_typed_func::<(), ()>(&mut store, "_initialize")
                .and_then(|initialize| initialize.call(&mut store, ()))
                .map_err(|error| error.to_string())?;
        }
        let decide = instance
            .get_typed_func::<(), i32>(&mut store, REFERENCE_EXPORT)
            .map_err(|error| error.to_string())?;
        let return_code = decide
            .call(&mut store, ())
            .map_err(|error| error.to_string())?;

        if let Some(offset) = store.data().error_offset {
            let error = read_allocation(&store, offset, None)?;
            return Err(String::from_utf8_lossy(&error).into_owned());
        }
        if return_code != 0 {
            return Err(format!(
                "{REFERENCE_EXPORT} returned error code {return_code}"
            ));
        }
        let Some((offset, length)) = store.data().output else {
            return Ok(Vec::new());
        };
        let length = usize::try_from(length).map_err(|_| "Mind output length does not fit")?;
        if length > max_output_bytes {
            return Err(format!(
                "reference Mind output is {length} bytes; limit is {max_output_bytes}"
            ));
        }
        if length == 0 {
            return Ok(Vec::new());
        }
        read_allocation(&store, offset, Some(length))
    }
}

impl Drop for ExtismCompatExecutor {
    fn drop(&mut self) {
        self.ticker_stop.take();
        if let Some(ticker) = self.ticker.take() {
            let _ = ticker.join();
        }
    }
}

pub(crate) fn load_single_module(manifest: &Manifest) -> Result<Vec<u8>, String> {
    if manifest.wasm.len() != 1 {
        return Err("Extism-compatible executor requires exactly one Mind module".into());
    }
    match &manifest.wasm[0] {
        Wasm::File { path, .. } => fs::read(path)
            .map_err(|error| format!("failed to read Mind module {}: {error}", path.display())),
        Wasm::Data { data, .. } => Ok(data.clone()),
        Wasm::Url { .. } => {
            Err("URL Mind modules are not admitted by the compatibility host".into())
        }
    }
}

fn validate_imports(module: &Module) -> Result<(), String> {
    for import in module.imports() {
        if import.module() != EXTISM_ENV {
            return Err(format!(
                "unsupported Mind import {}::{}",
                import.module(),
                import.name()
            ));
        }
        match import.ty() {
            ExternType::Func(ty) => validate_function_import(import.name(), &ty)?,
            _ => {
                return Err(format!(
                    "unsupported Extism Mind import {}::{}",
                    import.module(),
                    import.name()
                ));
            }
        }
    }
    match module.get_export(REFERENCE_EXPORT) {
        Some(ExternType::Func(ty)) => {
            validate_signature(REFERENCE_EXPORT, &ty, &[], &[ValType::I32])?;
        }
        _ => return Err(format!("Mind does not export {REFERENCE_EXPORT}")),
    }
    Ok(())
}

fn validate_function_import(name: &str, ty: &wasmtime::FuncType) -> Result<(), String> {
    match name {
        "input_length" => validate_signature(name, ty, &[], &[ValType::I64]),
        "input_load_u8" => validate_signature(name, ty, &[ValType::I64], &[ValType::I32]),
        "input_load_u64" => validate_signature(name, ty, &[ValType::I64], &[ValType::I64]),
        "alloc" => validate_signature(name, ty, &[ValType::I64], &[ValType::I64]),
        "free" => validate_signature(name, ty, &[ValType::I64], &[]),
        "length" | "length_unsafe" => {
            validate_signature(name, ty, &[ValType::I64], &[ValType::I64])
        }
        "load_u8" => validate_signature(name, ty, &[ValType::I64], &[ValType::I32]),
        "load_u64" => validate_signature(name, ty, &[ValType::I64], &[ValType::I64]),
        "error_set" => validate_signature(name, ty, &[ValType::I64], &[]),
        "store_u8" => validate_signature(name, ty, &[ValType::I64, ValType::I32], &[]),
        "store_u64" => validate_signature(name, ty, &[ValType::I64, ValType::I64], &[]),
        "output_set" => validate_signature(name, ty, &[ValType::I64, ValType::I64], &[]),
        _ => Err(format!(
            "unsupported Extism PDK import {EXTISM_ENV}::{name}"
        )),
    }
}

fn validate_signature(
    name: &str,
    ty: &wasmtime::FuncType,
    params: &[ValType],
    results: &[ValType],
) -> Result<(), String> {
    let actual_params: Vec<_> = ty.params().collect();
    let actual_results: Vec<_> = ty.results().collect();
    let types_match = |actual: &[ValType], expected: &[ValType]| {
        actual.len() == expected.len()
            && actual.iter().zip(expected).all(|(actual, expected)| {
                matches!(
                    (actual, expected),
                    (ValType::I32, ValType::I32) | (ValType::I64, ValType::I64)
                )
            })
    };
    if types_match(&actual_params, params) && types_match(&actual_results, results) {
        Ok(())
    } else {
        Err(format!(
            "Extism PDK import/export {name} has the wrong type"
        ))
    }
}

fn compat_linker(engine: &Engine) -> Result<Linker<CompatState>, String> {
    let mut linker = Linker::new(engine);
    linker
        .func_wrap(
            EXTISM_ENV,
            "input_length",
            |caller: Caller<'_, CompatState>| caller.data().input.len() as i64,
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "input_load_u8",
            |caller: Caller<'_, CompatState>, offset: i64| -> i32 {
                usize::try_from(offset)
                    .ok()
                    .and_then(|offset| caller.data().input.get(offset).copied())
                    .unwrap_or(0) as i32
            },
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "input_load_u64",
            |caller: Caller<'_, CompatState>, offset: i64| -> i64 {
                let Some(offset) = usize::try_from(offset).ok() else {
                    return 0;
                };
                let Some(end) = offset.checked_add(8) else {
                    return 0;
                };
                caller
                    .data()
                    .input
                    .get(offset..end)
                    .and_then(|bytes| bytes.try_into().ok())
                    .map(i64::from_le_bytes)
                    .unwrap_or(0)
            },
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "alloc",
            |mut caller: Caller<'_, CompatState>, length: i64| -> wasmtime::Result<i64> {
                allocate(&mut caller, length)
            },
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "free",
            |mut caller: Caller<'_, CompatState>, offset: i64| {
                let offset = u64::try_from(offset)
                    .map_err(|_| wasmtime::Error::msg("negative Extism allocation offset"))?;
                free_allocation(caller.data_mut(), offset);
                Ok(())
            },
        )
        .map_err(|error| error.to_string())?;
    for name in ["length", "length_unsafe"] {
        linker
            .func_wrap(
                EXTISM_ENV,
                name,
                |caller: Caller<'_, CompatState>, offset: i64| -> i64 {
                    u64::try_from(offset)
                        .ok()
                        .and_then(|offset| {
                            caller
                                .data()
                                .allocations
                                .iter()
                                .find_map(|&(start, length)| (start == offset).then_some(length))
                        })
                        .and_then(|length| i64::try_from(length).ok())
                        .unwrap_or(0)
                },
            )
            .map_err(|error| error.to_string())?;
    }
    linker
        .func_wrap(
            EXTISM_ENV,
            "load_u8",
            |caller: Caller<'_, CompatState>, offset: i64| -> wasmtime::Result<i32> {
                Ok(read_memory(&caller, offset, 1)?[0] as i32)
            },
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "load_u64",
            |caller: Caller<'_, CompatState>, offset: i64| -> wasmtime::Result<i64> {
                let bytes: [u8; 8] = read_memory(&caller, offset, 8)?
                    .try_into()
                    .expect("requested eight bytes");
                Ok(i64::from_le_bytes(bytes))
            },
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "store_u8",
            |mut caller: Caller<'_, CompatState>, offset: i64, value: i32| {
                write_memory(&mut caller, offset, &[value as u8])
            },
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "store_u64",
            |mut caller: Caller<'_, CompatState>, offset: i64, value: i64| {
                write_memory(&mut caller, offset, &value.to_le_bytes())
            },
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "output_set",
            |mut caller: Caller<'_, CompatState>, offset: i64, length: i64| {
                let offset = u64::try_from(offset)
                    .map_err(|_| wasmtime::Error::msg("negative Extism output offset"))?;
                let length = u64::try_from(length)
                    .map_err(|_| wasmtime::Error::msg("negative Extism output length"))?;
                caller.data_mut().output = Some((offset, length));
                Ok(())
            },
        )
        .map_err(|error| error.to_string())?;
    linker
        .func_wrap(
            EXTISM_ENV,
            "error_set",
            |mut caller: Caller<'_, CompatState>, offset: i64| {
                let offset = u64::try_from(offset)
                    .map_err(|_| wasmtime::Error::msg("negative Extism error offset"))?;
                caller.data_mut().error_offset = (offset != 0).then_some(offset);
                Ok(())
            },
        )
        .map_err(|error| error.to_string())?;
    Ok(linker)
}

fn allocate(caller: &mut Caller<'_, CompatState>, length: i64) -> wasmtime::Result<i64> {
    let length = u64::try_from(length)
        .map_err(|_| wasmtime::Error::msg("negative Extism allocation length"))?;
    if length == 0 {
        return Ok(0);
    }
    let reusable = caller
        .data()
        .free_ranges
        .iter()
        .position(|&(_, available)| available >= length);
    if let Some(reusable) = reusable {
        let state = caller.data_mut();
        let (offset, available) = state.free_ranges.swap_remove(reusable);
        if available > length {
            state
                .free_ranges
                .push((offset + length, available - length));
        }
        state.allocations.push((offset, length));
        return i64::try_from(offset)
            .map_err(|_| wasmtime::Error::msg("Extism allocation offset overflow"));
    }
    let current_bytes = caller.data().host_memory.len() as u64;
    let offset = current_bytes
        .checked_add(7)
        .map(|offset| offset & !7)
        .ok_or_else(|| wasmtime::Error::msg("Extism allocation offset overflow"))?;
    let end = offset
        .checked_add(length)
        .ok_or_else(|| wasmtime::Error::msg("Extism allocation overflow"))?;
    let end = usize::try_from(end)
        .map_err(|_| wasmtime::Error::msg("Extism allocation does not fit this platform"))?;
    if end > caller.data().max_host_memory_bytes {
        return Err(wasmtime::Error::msg("Extism host memory limit exceeded"));
    }
    let state = caller.data_mut();
    state.host_memory.resize(end, 0);
    state.allocations.push((offset, length));
    i64::try_from(offset).map_err(|_| wasmtime::Error::msg("Extism allocation offset overflow"))
}

fn free_allocation(state: &mut CompatState, offset: u64) {
    let Some(allocation) = state
        .allocations
        .iter()
        .position(|&(start, _)| start == offset)
    else {
        return;
    };
    let (_, mut length) = state.allocations.swap_remove(allocation);
    let mut start = offset;
    if let Some(previous) = state
        .free_ranges
        .iter()
        .position(|&(candidate, candidate_length)| {
            candidate.checked_add(candidate_length) == Some(offset)
        })
    {
        let (previous_start, previous_length) = state.free_ranges.swap_remove(previous);
        start = previous_start;
        length += previous_length;
    }
    if let Some(next) = state
        .free_ranges
        .iter()
        .position(|&(candidate, _)| start.checked_add(length) == Some(candidate))
    {
        let (_, next_length) = state.free_ranges.swap_remove(next);
        length += next_length;
    }
    state.free_ranges.push((start, length));
}

fn write_memory(
    caller: &mut Caller<'_, CompatState>,
    offset: i64,
    bytes: &[u8],
) -> wasmtime::Result<()> {
    let offset_u64 =
        u64::try_from(offset).map_err(|_| wasmtime::Error::msg("negative Extism memory offset"))?;
    ensure_allocated_range(caller.data(), offset_u64, bytes.len())?;
    let offset = usize::try_from(offset_u64)
        .map_err(|_| wasmtime::Error::msg("Extism memory offset does not fit this platform"))?;
    let end = offset
        .checked_add(bytes.len())
        .ok_or_else(|| wasmtime::Error::msg("Extism memory range overflow"))?;
    let target = caller
        .data_mut()
        .host_memory
        .get_mut(offset..end)
        .ok_or_else(|| wasmtime::Error::msg("Extism memory write is outside an allocation"))?;
    target.copy_from_slice(bytes);
    Ok(())
}

fn read_memory(
    caller: &Caller<'_, CompatState>,
    offset: i64,
    length: usize,
) -> wasmtime::Result<Vec<u8>> {
    let offset_u64 =
        u64::try_from(offset).map_err(|_| wasmtime::Error::msg("negative Extism memory offset"))?;
    ensure_allocated_range(caller.data(), offset_u64, length)?;
    let offset = usize::try_from(offset_u64)
        .map_err(|_| wasmtime::Error::msg("Extism memory offset does not fit this platform"))?;
    let end = offset
        .checked_add(length)
        .ok_or_else(|| wasmtime::Error::msg("Extism memory range overflow"))?;
    caller
        .data()
        .host_memory
        .get(offset..end)
        .map(|bytes| bytes.to_vec())
        .ok_or_else(|| wasmtime::Error::msg("Extism memory read is outside an allocation"))
}

fn ensure_allocated_range(state: &CompatState, offset: u64, length: usize) -> wasmtime::Result<()> {
    let length = u64::try_from(length)
        .map_err(|_| wasmtime::Error::msg("Extism memory length does not fit this platform"))?;
    let end = offset
        .checked_add(length)
        .ok_or_else(|| wasmtime::Error::msg("Extism memory range overflow"))?;
    let contained = state.allocations.iter().any(|&(start, allocation_length)| {
        start <= offset
            && start
                .checked_add(allocation_length)
                .is_some_and(|allocation_end| end <= allocation_end)
    });
    if contained {
        Ok(())
    } else {
        Err(wasmtime::Error::msg(
            "Extism memory access is outside an allocation",
        ))
    }
}

fn read_allocation(
    store: &Store<CompatState>,
    offset: u64,
    requested_length: Option<usize>,
) -> Result<Vec<u8>, String> {
    let allocation_length = store
        .data()
        .allocations
        .iter()
        .find_map(|&(start, length)| (start == offset).then_some(length))
        .ok_or_else(|| format!("Extism output offset {offset} is not an allocation"))?;
    let length = requested_length.unwrap_or(
        usize::try_from(allocation_length).map_err(|_| "Extism allocation length does not fit")?,
    );
    if u64::try_from(length).map_err(|_| "Extism output length does not fit")? > allocation_length {
        return Err("Extism output exceeds its allocation".into());
    }
    let offset = usize::try_from(offset).map_err(|_| "Extism output offset does not fit")?;
    let end = offset
        .checked_add(length)
        .ok_or("Extism output range overflow")?;
    store
        .data()
        .host_memory
        .get(offset..end)
        .map(|bytes| bytes.to_vec())
        .ok_or_else(|| "Extism output is outside guest memory".into())
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use blob_interface::randomness::PrivateRandom;
    use blob_interface::reference_mind::{
        CurrentTileObservation, EFFORT_BURST_BIT, EFFORT_GENTLE_BIT, EFFORT_STANDARD_BIT,
        LocalObservation, REFERENCE_SIGNAL_CHANNELS, ReferenceActionSpace, ReferenceMindInput,
        ReferenceSelfState,
    };
    use blob_interface::reference_mind_converter::{
        ReferenceMindLimits, capnp_to_reference_mind_decision, reference_mind_input_to_capnp,
    };
    use extism_manifest::MemoryOptions;

    use super::*;

    fn test_manifest(wat: &str, timeout: Duration) -> Manifest {
        Manifest::new([Wasm::data(wat::parse_str(wat).expect("valid test WAT"))])
            .with_memory_options(MemoryOptions::new().with_max_pages(2))
            .with_timeout(timeout)
    }

    #[test]
    fn rejects_non_extism_and_capability_imports() {
        for wat in [
            r#"(module
                (import "wasi_snapshot_preview1" "clock_time_get"
                    (func (param i32 i64 i32) (result i32)))
                (func (export "reference_mind_function") (result i32) i32.const 0))"#,
            r#"(module
                (import "extism:host/env" "config_get"
                    (func (param i64) (result i64)))
                (func (export "reference_mind_function") (result i32) i32.const 0))"#,
        ] {
            let error = ExtismCompatExecutor::new(
                &test_manifest(wat, Duration::from_millis(25)),
                Config::new(),
            )
            .err()
            .expect("capability import should be rejected");
            assert!(error.contains("unsupported"), "{error}");
        }
    }

    #[test]
    fn traps_nonterminating_minds_at_the_manifest_deadline() {
        let wat = r#"(module
            (func (export "reference_mind_function") (result i32)
                (loop $forever br $forever)
                i32.const 0))"#;
        let executor = ExtismCompatExecutor::new(
            &test_manifest(wat, Duration::from_millis(20)),
            Config::new(),
        )
        .unwrap();
        let started = Instant::now();
        let error = executor.call(Vec::new(), 64).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(!error.is_empty());
    }

    #[test]
    fn supports_the_deterministic_extism_memory_contract() {
        let wat = r#"(module
            (import "extism:host/env" "alloc" (func $alloc (param i64) (result i64)))
            (import "extism:host/env" "free" (func $free (param i64)))
            (import "extism:host/env" "length" (func $length (param i64) (result i64)))
            (import "extism:host/env" "length_unsafe" (func $length_unsafe (param i64) (result i64)))
            (import "extism:host/env" "store_u64" (func $store_u64 (param i64 i64)))
            (import "extism:host/env" "load_u64" (func $load_u64 (param i64) (result i64)))
            (import "extism:host/env" "output_set" (func $output_set (param i64 i64)))
            (func (export "reference_mind_function") (result i32)
                (local $offset i64)
                i64.const 8
                call $alloc
                local.tee $offset
                i64.const 72623859790382856
                call $store_u64
                local.get $offset
                call $length
                i64.const 8
                i64.ne
                if unreachable end
                local.get $offset
                call $length_unsafe
                i64.const 8
                i64.ne
                if unreachable end
                local.get $offset
                call $load_u64
                i64.const 72623859790382856
                i64.ne
                if unreachable end
                local.get $offset
                i64.const 8
                call $output_set
                i32.const 0))"#;
        let executor = ExtismCompatExecutor::new(
            &test_manifest(wat, Duration::from_millis(25)),
            Config::new(),
        )
        .unwrap();
        assert_eq!(
            executor.call(Vec::new(), 8).unwrap(),
            [8, 7, 6, 5, 4, 3, 2, 1]
        );
    }

    #[test]
    fn enforces_allocation_ranges_and_reuses_freed_space() {
        let out_of_range = r#"(module
            (import "extism:host/env" "alloc" (func $alloc (param i64) (result i64)))
            (import "extism:host/env" "store_u64" (func $store_u64 (param i64 i64)))
            (func (export "reference_mind_function") (result i32)
                i64.const 1
                call $alloc
                i64.const 7
                call $store_u64
                i32.const 0))"#;
        let executor = ExtismCompatExecutor::new(
            &test_manifest(out_of_range, Duration::from_millis(25)),
            Config::new(),
        )
        .unwrap();
        assert!(executor.call(Vec::new(), 8).is_err());

        let reuse = r#"(module
            (import "extism:host/env" "alloc" (func $alloc (param i64) (result i64)))
            (import "extism:host/env" "free" (func $free (param i64)))
            (func (export "reference_mind_function") (result i32)
                (local $offset i64)
                i64.const 65500
                call $alloc
                local.tee $offset
                call $free
                i64.const 65500
                call $alloc
                local.get $offset
                i64.ne
                if unreachable end
                i32.const 0))"#;
        let executor = ExtismCompatExecutor::new(
            &test_manifest(reuse, Duration::from_millis(25)),
            Config::new(),
        )
        .unwrap();
        assert_eq!(executor.call(Vec::new(), 8).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn inline_host_storage_spills_without_changing_the_memory_contract() {
        let wat = r#"(module
            (import "extism:host/env" "alloc" (func $alloc (param i64) (result i64)))
            (import "extism:host/env" "store_u8" (func $store_u8 (param i64 i32)))
            (import "extism:host/env" "output_set" (func $output_set (param i64 i64)))
            (func (export "reference_mind_function") (result i32)
                (local $output i64)
                i64.const 1 call $alloc drop
                i64.const 1 call $alloc drop
                i64.const 1 call $alloc drop
                i64.const 1 call $alloc drop
                i64.const 1 call $alloc drop
                i64.const 300 call $alloc local.tee $output
                i32.const 42 call $store_u8
                local.get $output i64.const 1 call $output_set
                i32.const 0))"#;
        let executor = ExtismCompatExecutor::new(
            &test_manifest(wat, Duration::from_millis(25)),
            Config::new(),
        )
        .unwrap();
        assert_eq!(executor.call(Vec::new(), 1).unwrap(), [42]);
    }

    #[test]
    #[ignore = "release-only compatible-call stage benchmark; run with --ignored --nocapture"]
    fn benchmark_fresh_call_stages() {
        const WARMUP_CALLS: usize = 2_000;
        const MEASURED_CALLS: usize = 50_000;
        let wat = r#"(module
            (func (export "reference_mind_function") (result i32)
                i32.const 0))"#;
        let manifest = test_manifest(wat, Duration::from_secs(1));

        for (allocator, config) in [
            (
                "pooling",
                crate::plugin_pool::pooling_config(&manifest, 1).unwrap(),
            ),
            ("on-demand", Config::new()),
        ] {
            let executor = ExtismCompatExecutor::new(&manifest, config).unwrap();
            for _ in 0..WARMUP_CALLS {
                black_box(executor.call(Vec::new(), 0).unwrap());
            }

            let mut store_setup = Duration::ZERO;
            let mut instantiate = Duration::ZERO;
            let mut export_lookup = Duration::ZERO;
            let mut guest_call = Duration::ZERO;
            let mut result_read = Duration::ZERO;
            let total_started = Instant::now();
            for _ in 0..MEASURED_CALLS {
                let started = Instant::now();
                let mut store = Store::new(
                    &executor.engine,
                    CompatState::new(Vec::new(), executor.max_memory_bytes),
                );
                store.limiter(|state| &mut state.limits);
                store.epoch_deadline_trap();
                store.set_epoch_deadline(executor.timeout_ticks);
                store_setup += started.elapsed();

                let started = Instant::now();
                let instance = executor.instance_pre.instantiate(&mut store).unwrap();
                instantiate += started.elapsed();

                let started = Instant::now();
                let decide = instance
                    .get_typed_func::<(), i32>(&mut store, REFERENCE_EXPORT)
                    .unwrap();
                export_lookup += started.elapsed();

                let started = Instant::now();
                let return_code = black_box(decide.call(&mut store, ()).unwrap());
                guest_call += started.elapsed();

                let started = Instant::now();
                assert_eq!(return_code, 0);
                assert!(store.data().output.is_none());
                result_read += started.elapsed();
            }
            let total = total_started.elapsed();
            let ns_per_call =
                |duration: Duration| duration.as_nanos() as f64 / MEASURED_CALLS as f64;
            println!(
                "compat fresh-call {allocator:<9}: total {:>8.0} ns/call | \
                 store {:>7.0} | instantiate {:>7.0} | export {:>6.0} | \
                 guest {:>6.0} | result {:>5.0}",
                ns_per_call(total),
                ns_per_call(store_setup),
                ns_per_call(instantiate),
                ns_per_call(export_lookup),
                ns_per_call(guest_call),
                ns_per_call(result_read),
            );
        }
    }

    #[test]
    #[ignore = "release-only Extism byte-contract benchmark; run with --ignored --nocapture"]
    fn benchmark_extism_byte_contract() {
        const WARMUP_CALLS: usize = 2_000;
        const MEASURED_CALLS: usize = 20_000;
        let wat = r#"(module
            (import "extism:host/env" "input_length" (func $input_length (result i64)))
            (import "extism:host/env" "input_load_u64" (func $input_load_u64 (param i64) (result i64)))
            (import "extism:host/env" "alloc" (func $alloc (param i64) (result i64)))
            (import "extism:host/env" "store_u64" (func $store_u64 (param i64 i64)))
            (import "extism:host/env" "output_set" (func $output_set (param i64 i64)))
            (func (export "reference_mind_function") (result i32)
                (local $length i64) (local $output i64) (local $cursor i64)
                call $input_length
                local.tee $length
                call $alloc
                local.set $output
                (loop $copy
                    local.get $output
                    local.get $cursor
                    i64.add
                    local.get $cursor
                    call $input_load_u64
                    call $store_u64
                    local.get $cursor
                    i64.const 8
                    i64.add
                    local.tee $cursor
                    local.get $length
                    i64.lt_u
                    br_if $copy)
                local.get $output
                local.get $length
                call $output_set
                i32.const 0))"#;
        let manifest = test_manifest(wat, Duration::from_secs(1));
        for (allocator, config) in [
            (
                "pooling",
                crate::plugin_pool::pooling_config(&manifest, 1).unwrap(),
            ),
            ("on-demand", Config::new()),
        ] {
            let executor = ExtismCompatExecutor::new(&manifest, config).unwrap();
            for payload_bytes in [96, 512, 2_016] {
                let input: Vec<u8> = (0..payload_bytes).map(|index| index as u8).collect();
                for _ in 0..WARMUP_CALLS {
                    black_box(executor.call(input.clone(), payload_bytes).unwrap());
                }
                let started = Instant::now();
                for _ in 0..MEASURED_CALLS {
                    let output = black_box(executor.call(input.clone(), payload_bytes).unwrap());
                    assert_eq!(output, input);
                }
                let elapsed = started.elapsed();
                println!(
                    "compat byte-contract {allocator:<9} {payload_bytes:>4} bytes: \
                     {:>8.0} ns/call, {:>8.0} MiB/s",
                    elapsed.as_nanos() as f64 / MEASURED_CALLS as f64,
                    (payload_bytes * MEASURED_CALLS) as f64
                        / elapsed.as_secs_f64()
                        / (1024.0 * 1024.0),
                );
            }
        }
    }

    #[test]
    #[ignore = "release-only maintained-Mind call benchmark; run with --ignored --nocapture"]
    fn benchmark_maintained_mind_call() {
        const WARMUP_CALLS: usize = 1_000;
        const MEASURED_CALLS: usize = 10_000;
        let mind_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../target/wasm32-unknown-unknown/release/simple_mind.wasm");
        if !mind_path.exists() {
            eprintln!("Skipping: build simple_mind.wasm first");
            return;
        }
        let manifest = Manifest::new([Wasm::file(mind_path)])
            .with_memory_options(MemoryOptions::new().with_max_pages(1_024))
            .with_timeout(Duration::from_secs(1));
        let limits = ReferenceMindLimits::default();
        let input = ReferenceMindInput {
            self_state: ReferenceSelfState {
                core_mass: 10,
                assimilated_energy: 100,
                gut_energy: 0,
                metabolism_remainder: 0,
                carried_material_mass: 0,
                marker: 0,
                guarded: false,
                last_outcome: None,
            },
            current_tile: CurrentTileObservation {
                elevation: 0,
                plant_energy: 0,
                plant_capacity: 100,
                plant_growth_rate: 1,
                loose_energy: 0,
                diffuse_energy: 0,
                signal_energy: [0; REFERENCE_SIGNAL_CHANNELS],
            },
            slots: (0..8)
                .map(|slot| LocalObservation {
                    slot,
                    dx: (slot % 3) as i8 - 1,
                    dy: (slot / 3) as i8 - 1,
                    distance_cost_q10: 1_024,
                    reachable: true,
                    elevation: Some(0),
                    plant_energy: Some(0),
                    plant_capacity: Some(100),
                    plant_growth_rate: Some(1),
                    loose_energy: Some(0),
                    diffuse_energy: Some(0),
                    signal_energy: Some([0; REFERENCE_SIGNAL_CHANNELS]),
                    neighbor: None,
                })
                .collect(),
            action_space: ReferenceActionSpace {
                wait_enabled: true,
                guard_enabled: true,
                consume_enabled: true,
                excavate_enabled: true,
                deposit_terrain_enabled: true,
                signal_enabled: true,
                move_targets: 0xff,
                attack_targets: 0xff,
                split_targets: 0xff,
                regurgitate_targets: 0xff,
                effort_mask: EFFORT_GENTLE_BIT | EFFORT_STANDARD_BIT | EFFORT_BURST_BIT,
                max_consume_amount: 16,
                gut_capacity: 64,
                max_private_memory_bytes: limits.max_private_memory_bytes as u32,
                minimum_survival_energy: 1,
                child_core_mass: 10,
                metabolism_rate_numerator: 1,
                metabolism_rate_denominator: 1_024,
                terrain_mass_per_elevation: 10,
                signal_emission_cost: 1,
            },
            private_memory: Vec::new(),
            randomness: PrivateRandom::ZERO,
        };
        let input = reference_mind_input_to_capnp(&input, limits).unwrap();

        for (allocator, config) in [
            (
                "pooling",
                crate::plugin_pool::pooling_config(&manifest, 1).unwrap(),
            ),
            ("on-demand", Config::new()),
        ] {
            let executor = ExtismCompatExecutor::new(&manifest, config).unwrap();
            let mut output_bytes = 0;
            for _ in 0..WARMUP_CALLS {
                output_bytes = executor
                    .call(input.clone(), limits.max_action_bytes)
                    .unwrap()
                    .len();
            }
            let mut executor_time = Duration::ZERO;
            let mut decode_time = Duration::ZERO;
            let started = Instant::now();
            for _ in 0..MEASURED_CALLS {
                let phase = Instant::now();
                let output = executor
                    .call(input.clone(), limits.max_action_bytes)
                    .unwrap();
                executor_time += phase.elapsed();
                let phase = Instant::now();
                black_box(capnp_to_reference_mind_decision(&output, limits).unwrap());
                decode_time += phase.elapsed();
            }
            let elapsed = started.elapsed();
            println!(
                "compat maintained-Mind {allocator:<9}: {:>8.0} ns/action \
                 (executor {:>8.0}, decode {:>5.0}; {} input bytes, \
                 {output_bytes} output bytes)",
                elapsed.as_nanos() as f64 / MEASURED_CALLS as f64,
                executor_time.as_nanos() as f64 / MEASURED_CALLS as f64,
                decode_time.as_nanos() as f64 / MEASURED_CALLS as f64,
                input.len(),
            );
        }
    }
}
