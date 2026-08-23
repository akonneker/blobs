/// Persistent plugin worker pool for parallel WASM mind invocations.
///
/// Instead of spawning threads per tick (expensive), worker threads are spawned once
/// at game creation and communicate via channels. Compatible-executor workers
/// share one immutable compiled runtime and create a pristine guest instance
/// for every invocation. Stock Extism retains a worker-local compiled handle
/// because its host-function user-data container is deliberately not Sync.
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;

use blob_engine::mind_runtime::inspect_mind_artifact;
use blob_engine::resolution::MindRuntimeProfile;
use blob_interface::reference_mind::ReferenceMindDecision;
use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, capnp_to_reference_mind_decision,
};
use blob_interface::types::CellId;
use extism::{CompiledPlugin, Manifest, Plugin, PluginBuilder};
use wasmtime::{Config, PoolingAllocationConfig};

use crate::config::WasmExecutorKind;
use crate::extism_compat::{ExtismCompatExecutor, load_single_module};

const MIN_REFERENCE_INVOCATIONS_PER_ACTIVE_WORKER: usize = 64;
const WASM_PAGE_BYTES: usize = 65_536;
// Each worker executes only one fresh plugin at a time. Eight slots per
// concurrent worker leave room for the Extism kernel plus the guest while
// preserving the aggregate capacity of the former per-worker engines.
const POOLED_RUNTIME_RESOURCES_PER_WORKER: u32 = 8;

pub(crate) fn pooling_config(
    manifest: &Manifest,
    maximum_concurrent_calls: usize,
) -> Result<Config, String> {
    if maximum_concurrent_calls == 0 {
        return Err("pooling allocator requires at least one concurrent call".into());
    }
    let max_pages = manifest
        .memory
        .max_pages
        .ok_or("pooling allocator requires a bounded manifest memory limit")?;
    let max_memory_size = usize::try_from(max_pages)
        .ok()
        .and_then(|pages| pages.checked_mul(WASM_PAGE_BYTES))
        .ok_or("pooled manifest memory limit does not fit this platform")?;
    let maximum_concurrent_calls = u32::try_from(maximum_concurrent_calls)
        .map_err(|_| "pooled worker count does not fit Wasmtime limits")?;
    let total_resources = POOLED_RUNTIME_RESOURCES_PER_WORKER
        .checked_mul(maximum_concurrent_calls)
        .ok_or("pooled runtime resource count overflow")?;
    let mut pooling = PoolingAllocationConfig::new();
    pooling
        .total_core_instances(total_resources)
        .total_memories(total_resources)
        .total_tables(total_resources)
        .max_memories_per_module(4)
        .max_tables_per_module(4)
        .max_memory_size(max_memory_size)
        .max_unused_warm_slots(maximum_concurrent_calls);
    let mut config = Config::new();
    config.allocation_strategy(pooling);
    Ok(config)
}

fn reference_active_worker_count(pool_size: usize, input_count: usize) -> usize {
    input_count
        .div_ceil(MIN_REFERENCE_INVOCATIONS_PER_ACTIVE_WORKER)
        .clamp(1, pool_size.max(1))
}

/// Work item sent to a plugin worker
pub enum WorkItem {
    ProcessReferenceBatch {
        inputs: Vec<(CellId, Vec<u8>)>,
        limits: ReferenceMindLimits,
        result_tx: mpsc::SyncSender<WorkResult>,
    },
    /// Shutdown the worker thread
    Shutdown,
}

/// Result from a plugin worker
pub enum WorkResult {
    ReferenceBatch {
        decisions: Vec<(CellId, Result<ReferenceMindDecision, String>)>,
    },
}

/// Handle to a single plugin worker thread
pub struct PluginWorker {
    work_tx: mpsc::Sender<WorkItem>,
    handle: Option<JoinHandle<()>>,
}

/// Pool of persistent plugin worker threads for a single team
pub struct PluginPool {
    workers: Vec<PluginWorker>,
}

enum WorkerExecutor {
    Extism(Box<CompiledPlugin>),
    ExtismCompat(Arc<ExtismCompatExecutor>),
}

impl WorkerExecutor {
    fn call(&self, input_bytes: Vec<u8>, max_output_bytes: usize) -> Result<Vec<u8>, String> {
        match self {
            Self::Extism(compiled) => Plugin::new_from_compiled(compiled)
                .and_then(|mut plugin| {
                    plugin.call::<&Vec<u8>, Vec<u8>>("reference_mind_function", &input_bytes)
                })
                .map_err(|error| error.to_string()),
            Self::ExtismCompat(executor) => executor.call(input_bytes, max_output_bytes),
        }
    }
}

fn compile_extism_worker(
    manifest: &Manifest,
    use_pooling_allocator: bool,
) -> Result<WorkerExecutor, String> {
    let builder = PluginBuilder::new(manifest.clone());
    let builder = if use_pooling_allocator {
        builder.with_wasmtime_config(pooling_config(manifest, 1)?)
    } else {
        builder
    };
    builder
        .compile()
        .map(|compiled| WorkerExecutor::Extism(Box::new(compiled)))
        .map_err(|error| error.to_string())
}

fn compile_shared_compat_runtime(
    manifest: &Manifest,
    maximum_concurrent_calls: usize,
    use_pooling_allocator: bool,
) -> Result<Arc<ExtismCompatExecutor>, String> {
    let config = if use_pooling_allocator {
        pooling_config(manifest, maximum_concurrent_calls)?
    } else {
        Config::new()
    };
    ExtismCompatExecutor::new(manifest, config).map(Arc::new)
}

impl PluginPool {
    /// Create a new plugin pool by spawning worker threads
    ///
    /// The compatible executor compiles once and all workers share that
    /// immutable runtime. Mutable WebAssembly state is never reused between
    /// invocations. Stock Extism compiles one worker-local descriptor because
    /// its public compiled type is not thread-safe.
    pub fn new(
        manifest: Manifest,
        worker_count: usize,
        use_pooling_allocator: bool,
        executor_kind: WasmExecutorKind,
    ) -> Result<Self, String> {
        if worker_count == 0 {
            return Err("plugin pool requires at least one worker".into());
        }
        let artifact = load_single_module(&manifest)?;
        inspect_mind_artifact(&artifact, MindRuntimeProfile::ExtismPdkDeterministicV1)
            .map_err(|error| error.to_string())?;
        let shared_compat = match executor_kind {
            WasmExecutorKind::Extism => None,
            WasmExecutorKind::ExtismCompat => Some(compile_shared_compat_runtime(
                &manifest,
                worker_count,
                use_pooling_allocator,
            )?),
        };
        let mut pending_workers = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let manifest = manifest.clone();
            let shared_compat = shared_compat.clone();
            let (work_tx, work_rx) = mpsc::channel::<WorkItem>();
            let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);

            let handle = std::thread::spawn(move || {
                let runtime = match executor_kind {
                    WasmExecutorKind::Extism => {
                        compile_extism_worker(&manifest, use_pooling_allocator)
                    }
                    WasmExecutorKind::ExtismCompat => Ok(WorkerExecutor::ExtismCompat(
                        shared_compat.expect("compatible runtime was compiled before workers"),
                    )),
                };
                let runtime = match runtime {
                    Ok(runtime) => {
                        let _ = ready_tx.send(Ok(()));
                        runtime
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                loop {
                    let item = match work_rx.recv() {
                        Ok(item) => item,
                        Err(_) => break, // Channel closed
                    };

                    match item {
                        WorkItem::ProcessReferenceBatch {
                            inputs,
                            limits,
                            result_tx,
                        } => {
                            let mut decisions = Vec::with_capacity(inputs.len());
                            for (cell_id, input_bytes) in inputs {
                                let decision = runtime
                                    .call(input_bytes, limits.max_action_bytes)
                                    .and_then(|output| {
                                        capnp_to_reference_mind_decision(&output, limits)
                                            .map_err(|error| error.to_string())
                                    });
                                decisions.push((cell_id, decision));
                            }
                            let _ = result_tx.send(WorkResult::ReferenceBatch { decisions });
                        }
                        WorkItem::Shutdown => {
                            break;
                        }
                    }
                }
            });

            pending_workers.push((
                worker_index,
                ready_rx,
                PluginWorker {
                    work_tx,
                    handle: Some(handle),
                },
            ));
        }

        let mut workers = Vec::with_capacity(worker_count);
        for (worker_index, ready_rx, worker) in pending_workers {
            match ready_rx.recv() {
                Ok(Ok(())) => workers.push(worker),
                Ok(Err(error)) => {
                    return Err(format!(
                        "failed to compile plugin worker {worker_index}: {error}"
                    ));
                }
                Err(error) => {
                    return Err(format!(
                        "plugin worker {worker_index} exited during startup: {error}"
                    ));
                }
            }
        }

        Ok(PluginPool { workers })
    }

    /// Number of workers in the pool
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub(crate) fn active_worker_count(&self, input_count: usize) -> usize {
        reference_active_worker_count(self.workers.len(), input_count)
    }

    fn send_reference_batch(
        &self,
        worker: usize,
        inputs: Vec<(CellId, Vec<u8>)>,
        limits: ReferenceMindLimits,
    ) -> Result<mpsc::Receiver<WorkResult>, String> {
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        self.workers
            .get(worker)
            .ok_or_else(|| format!("reference Mind worker {worker} does not exist"))?
            .work_tx
            .send(WorkItem::ProcessReferenceBatch {
                inputs,
                limits,
                result_tx,
            })
            .map_err(|error| format!("reference Mind worker is unavailable: {error}"))?;
        Ok(result_rx)
    }

    pub(crate) fn process_reference_worker_batch(
        &self,
        worker: usize,
        inputs: Vec<(CellId, Vec<u8>)>,
        limits: ReferenceMindLimits,
    ) -> Result<Vec<(CellId, ReferenceMindDecision)>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let result_rx = self.send_reference_batch(worker, inputs, limits)?;
        collect_reference_batch(result_rx.recv())
    }

    /// Invokes the explicit reference ABI export in pristine instances. Any
    /// trap, missing export, malformed output, or bound violation fails the
    /// batch; authoritative execution never silently substitutes an action.
    pub fn process_reference_batch(
        &self,
        inputs: Vec<(CellId, Vec<u8>)>,
        limits: ReferenceMindLimits,
    ) -> Result<Vec<(CellId, ReferenceMindDecision)>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let input_count = inputs.len();
        let active_workers = self.active_worker_count(input_count);
        let chunk_size = input_count.div_ceil(active_workers);
        let mut first_error = None;
        let mut pending_batches = Vec::with_capacity(active_workers);
        let mut inputs = inputs.into_iter();
        for worker in 0..active_workers {
            let batch: Vec<_> = inputs.by_ref().take(chunk_size).collect();
            if batch.is_empty() {
                break;
            }
            match self.send_reference_batch(worker, batch, limits) {
                Ok(result_rx) => pending_batches.push(result_rx),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            };
        }

        let mut results = Vec::with_capacity(input_count);
        for result_rx in pending_batches {
            match collect_reference_batch(result_rx.recv()) {
                Ok(batch) => results.extend(batch),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        results.sort_by_key(|(cell_id, _)| cell_id.0);
        Ok(results)
    }

    /// Shutdown all worker threads
    fn shutdown(&mut self) {
        for worker in &self.workers {
            let _ = worker.work_tx.send(WorkItem::Shutdown);
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

fn collect_reference_batch(
    received: Result<WorkResult, mpsc::RecvError>,
) -> Result<Vec<(CellId, ReferenceMindDecision)>, String> {
    let WorkResult::ReferenceBatch { decisions } =
        received.map_err(|error| format!("reference Mind worker exited: {error}"))?;
    let mut results = Vec::with_capacity(decisions.len());
    let mut first_error = None;
    for (cell_id, decision) in decisions {
        match decision {
            Ok(decision) => results.push((cell_id, decision)),
            Err(error) => {
                first_error.get_or_insert_with(|| {
                    format!("reference Mind failed for cell {cell_id:?}: {error}")
                });
            }
        }
    }
    if let Some(error) = first_error {
        Err(error)
    } else {
        results.sort_by_key(|(cell_id, _)| cell_id.0);
        Ok(results)
    }
}

impl Drop for PluginPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use extism::Wasm;
    use extism_manifest::MemoryOptions;

    use super::*;

    fn maintained_mind_manifest(name: &str) -> Option<Manifest> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .join("target/wasm32-unknown-unknown/release")
            .join(format!("{name}.wasm"));
        path.is_file().then(|| {
            Manifest::new([Wasm::file(path)])
                .with_memory_options(MemoryOptions::new().with_max_pages(1024))
                .with_timeout(Duration::from_secs(1))
        })
    }

    #[test]
    fn active_worker_count_preserves_capacity_without_oversharding() {
        assert_eq!(reference_active_worker_count(16, 1), 1);
        assert_eq!(reference_active_worker_count(16, 64), 1);
        assert_eq!(reference_active_worker_count(16, 65), 2);
        assert_eq!(reference_active_worker_count(16, 256), 4);
        assert_eq!(reference_active_worker_count(4, 10_000), 4);
    }

    #[test]
    fn pooling_requires_an_explicit_memory_bound() {
        assert!(pooling_config(&Manifest::default(), 1).is_err());

        let mut manifest = Manifest::default();
        manifest.memory.max_pages = Some(16);
        assert!(pooling_config(&manifest, 4).is_ok());
        assert!(pooling_config(&manifest, 0).is_err());
        assert!(pooling_config(&manifest, usize::MAX).is_err());
    }

    #[test]
    #[ignore = "release-only compatible-runtime startup diagnostic"]
    fn benchmark_shared_compatible_runtime_startup() {
        let Some(manifest) = maintained_mind_manifest("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };

        println!("Compatible executor compiled-runtime startup");
        for workers in [1_usize, 4, 16] {
            let legacy_started = Instant::now();
            let legacy_runtimes = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..workers)
                    .map(|_| {
                        let manifest = manifest.clone();
                        scope.spawn(move || {
                            compile_shared_compat_runtime(&manifest, 1, true).unwrap()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });
            let legacy = legacy_started.elapsed();
            drop(legacy_runtimes);

            let shared_started = Instant::now();
            let shared = compile_shared_compat_runtime(&manifest, workers, true).unwrap();
            let shared_elapsed = shared_started.elapsed();
            drop(shared);

            println!(
                "{workers:>2} workers  per-worker {:>8.2} ms  shared {:>8.2} ms  {:>5.2}x faster",
                legacy.as_secs_f64() * 1_000.0,
                shared_elapsed.as_secs_f64() * 1_000.0,
                legacy.as_secs_f64() / shared_elapsed.as_secs_f64(),
            );
        }
    }
}
