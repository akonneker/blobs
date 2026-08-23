use crate::config::{CellConfig, MemoryConfig, StateConfig};
use crate::plugin_pool::PluginPool;
use blob_engine::engine::{
    CellConfig as EngineCellConfig, Engine, ReferenceHostMode, ReferenceReplayStreamCheckpoint,
    ReferenceRuntimeCheckpoint, ReplayStreamConfig, TickEvents,
};
use blob_engine::resolution::{
    CanonicalHash, CheckpointLimits, ReferenceCheckpoint, ReferenceObservationBatch,
    ReferenceRuleset, ReplayBatchEvent, ReplayBundle, ReplayBundleLimits, ReplayLimits,
    ReplayManifest, ReplayManifestLimits, ReplaySegment, ReplaySegmentLimits,
};
use blob_interface::cell::Cell;
use blob_interface::randomness::{PrivateRandomDeriver, match_secret_from_seed};
use blob_interface::reference_mind::ReferenceMindDecision;
use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, reference_mind_input_to_capnp,
};
use blob_interface::types::{CellId, Coordinate, TeamId};
use blob_interface::world::{EnergySource, World};
use std::collections::HashMap;
use std::path::PathBuf;

use bincode;
use extism::{Error, Manifest, Wasm};
use extism_manifest::MemoryOptions;
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use image::{ImageBuffer, Rgb, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};

pub struct Game {
    pub world: World,
    pub teams: HashMap<TeamId, PluginPool>,
    match_secret: [u8; 32],
    pub cells: HashMap<CellId, Cell>,
    pub coordinate_map: HashMap<Coordinate, CellId>,
    pub inv_coordinate_map: HashMap<CellId, Coordinate>,
    pub iteration: u64,
    pub max_iterations: u64,
    pub cell_config: CellConfig,
    pub memory_config: MemoryConfig,
    next_cell_id: usize,
    reference_engine: Option<Engine>,
}

/// Serializable representation of the game state
/// This excludes non-serializable data like WebAssembly plugins
#[derive(Serialize, Deserialize)]
pub struct GameState {
    pub world: World,
    pub cells: HashMap<CellId, Cell>,
    pub coordinate_map: HashMap<Coordinate, CellId>,
    pub inv_coordinate_map: HashMap<CellId, Coordinate>,
    pub iteration: u64,
    pub max_iterations: u64,
    pub cell_config: CellConfig,
    pub memory_config: MemoryConfig,
    pub next_cell_id: usize,
    match_secret: [u8; 32],
    reference: ReferenceSaveState,
    // WebAssembly plugins are reconstructed when loading.
}

#[derive(Serialize, Deserialize)]
struct ReferenceSaveState {
    canonical_checkpoint: Vec<u8>,
    resolver_iteration: u64,
    authoritative_runtime_hash: [u8; 32],
    replay_checkpoint_interval: Option<u64>,
    replay_bundle: Option<Vec<u8>>,
    replay_stream: Option<ReplayStreamSaveState>,
}

#[derive(Serialize, Deserialize)]
struct ReplayStreamSaveState {
    config: ReplayStreamConfigSaveState,
    manifest: Vec<u8>,
    active_checkpoint: Vec<u8>,
    active_events: Vec<Vec<u8>>,
    active_event_bytes: u64,
    pending_segment: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize)]
struct ReplayStreamConfigSaveState {
    max_events_per_segment: u64,
    max_event_bytes_per_segment: u64,
    max_segment_bytes: u64,
    max_segment_events: u64,
    max_frame_bytes: u64,
    max_commitments: u64,
    max_outcomes: u64,
    max_claims: u64,
    max_deaths: u64,
    max_births: u64,
    max_delta_tiles: u64,
    max_delta_cells: u64,
    max_private_memory_bytes: u64,
    max_checkpoint_bytes: u64,
    max_checkpoint_tiles: u64,
    max_checkpoint_cells: u64,
    max_checkpoint_private_memory_bytes: u64,
    max_manifest_bytes: u64,
    max_manifest_segments: u64,
}
const GAME_STATE_MAGIC: &[u8; 8] = b"BLBGST01";
const GAME_STATE_VERSION: u16 = 11;
const MAX_GAME_STATE_BYTES: usize = 128 * 1024 * 1024;
const PARALLEL_WASM_OBSERVATION_THRESHOLD: usize = 256;

impl ReferenceSaveState {
    fn from_runtime(runtime: ReferenceRuntimeCheckpoint, runtime_hash: CanonicalHash) -> Self {
        let (replay_checkpoint_interval, replay_bundle) =
            runtime
                .replay
                .map_or((None, None), |(checkpoint_interval, bundle)| {
                    (Some(checkpoint_interval), Some(bundle.to_bytes().to_vec()))
                });
        let replay_stream = runtime
            .replay_stream
            .map(ReplayStreamSaveState::from_runtime);
        Self {
            canonical_checkpoint: runtime.canonical.to_bytes(),
            resolver_iteration: runtime.iteration,
            authoritative_runtime_hash: *runtime_hash.as_bytes(),
            replay_checkpoint_interval,
            replay_bundle,
            replay_stream,
        }
    }

    fn into_runtime(self) -> Result<ReferenceRuntimeCheckpoint, String> {
        let canonical = ReferenceCheckpoint::from_bytes(&self.canonical_checkpoint)
            .map_err(|error| format!("invalid canonical checkpoint: {error}"))?;
        let replay = match (self.replay_checkpoint_interval, self.replay_bundle) {
            (None, None) => None,
            (Some(interval), Some(bytes)) => Some((
                interval,
                ReplayBundle::from_bytes(&bytes)
                    .map_err(|error| format!("invalid saved replay bundle: {error}"))?,
            )),
            _ => return Err("saved replay metadata is incomplete".into()),
        };
        let replay_stream = self
            .replay_stream
            .map(ReplayStreamSaveState::into_runtime)
            .transpose()?;
        Ok(ReferenceRuntimeCheckpoint {
            canonical,
            iteration: self.resolver_iteration,
            replay,
            replay_stream,
        })
    }
}

impl ReplayStreamSaveState {
    fn from_runtime(runtime: ReferenceReplayStreamCheckpoint) -> Self {
        Self {
            config: ReplayStreamConfigSaveState::from_runtime(runtime.config),
            manifest: runtime.manifest.to_bytes().to_vec(),
            active_checkpoint: runtime.active_checkpoint.to_bytes(),
            active_events: runtime
                .active_events
                .iter()
                .map(ReplayBatchEvent::to_bytes)
                .collect(),
            active_event_bytes: runtime.active_event_bytes as u64,
            pending_segment: runtime
                .pending_segment
                .map(|segment| segment.to_bytes().to_vec()),
        }
    }

    fn into_runtime(self) -> Result<ReferenceReplayStreamCheckpoint, String> {
        let config = self.config.into_runtime()?;
        let manifest =
            ReplayManifest::from_bytes_with_limits(&self.manifest, config.manifest_limits)
                .map_err(|error| format!("invalid replay stream manifest: {error}"))?;
        let active_checkpoint = ReferenceCheckpoint::from_bytes_with_limits(
            &self.active_checkpoint,
            config.segment_limits.checkpoint,
        )
        .map_err(|error| format!("invalid replay stream active checkpoint: {error}"))?;
        let active_events = self
            .active_events
            .into_iter()
            .map(|bytes| {
                ReplayBatchEvent::from_bytes_with_limits(&bytes, config.segment_limits.frame)
                    .map_err(|error| format!("invalid replay stream active event: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pending_segment = self
            .pending_segment
            .map(|bytes| {
                ReplaySegment::from_bytes_with_limits(&bytes, config.segment_limits)
                    .map_err(|error| format!("invalid pending replay segment: {error}"))
            })
            .transpose()?;
        Ok(ReferenceReplayStreamCheckpoint {
            config,
            active_start_cursor: manifest.final_cursor(),
            manifest,
            active_checkpoint,
            active_events,
            active_event_bytes: saved_usize(self.active_event_bytes, "active replay bytes")?,
            pending_segment,
        })
    }
}

impl ReplayStreamConfigSaveState {
    fn from_runtime(config: ReplayStreamConfig) -> Self {
        let frame = config.segment_limits.frame;
        let checkpoint = config.segment_limits.checkpoint;
        Self {
            max_events_per_segment: config.max_events_per_segment as u64,
            max_event_bytes_per_segment: config.max_event_bytes_per_segment as u64,
            max_segment_bytes: config.segment_limits.max_segment_bytes as u64,
            max_segment_events: config.segment_limits.max_events as u64,
            max_frame_bytes: frame.max_frame_bytes as u64,
            max_commitments: frame.max_commitments as u64,
            max_outcomes: frame.max_outcomes as u64,
            max_claims: frame.max_claims as u64,
            max_deaths: frame.max_deaths as u64,
            max_births: frame.max_births as u64,
            max_delta_tiles: frame.max_delta_tiles as u64,
            max_delta_cells: frame.max_delta_cells as u64,
            max_private_memory_bytes: frame.max_private_memory_bytes as u64,
            max_checkpoint_bytes: checkpoint.max_checkpoint_bytes as u64,
            max_checkpoint_tiles: checkpoint.max_tiles as u64,
            max_checkpoint_cells: checkpoint.max_cells as u64,
            max_checkpoint_private_memory_bytes: checkpoint.max_private_memory_bytes as u64,
            max_manifest_bytes: config.manifest_limits.max_manifest_bytes as u64,
            max_manifest_segments: config.manifest_limits.max_segments as u64,
        }
    }

    fn into_runtime(self) -> Result<ReplayStreamConfig, String> {
        Ok(ReplayStreamConfig {
            max_events_per_segment: saved_usize(
                self.max_events_per_segment,
                "stream events per segment",
            )?,
            max_event_bytes_per_segment: saved_usize(
                self.max_event_bytes_per_segment,
                "stream bytes per segment",
            )?,
            segment_limits: ReplaySegmentLimits {
                max_segment_bytes: saved_usize(self.max_segment_bytes, "segment bytes")?,
                max_events: saved_usize(self.max_segment_events, "segment events")?,
                frame: ReplayLimits {
                    max_frame_bytes: saved_usize(self.max_frame_bytes, "frame bytes")?,
                    max_commitments: saved_usize(self.max_commitments, "frame commitments")?,
                    max_outcomes: saved_usize(self.max_outcomes, "frame outcomes")?,
                    max_claims: saved_usize(self.max_claims, "frame claims")?,
                    max_deaths: saved_usize(self.max_deaths, "frame deaths")?,
                    max_births: saved_usize(self.max_births, "frame births")?,
                    max_delta_tiles: saved_usize(self.max_delta_tiles, "frame delta tiles")?,
                    max_delta_cells: saved_usize(self.max_delta_cells, "frame delta cells")?,
                    max_private_memory_bytes: saved_usize(
                        self.max_private_memory_bytes,
                        "frame private memory",
                    )?,
                },
                checkpoint: CheckpointLimits {
                    max_checkpoint_bytes: saved_usize(
                        self.max_checkpoint_bytes,
                        "checkpoint bytes",
                    )?,
                    max_tiles: saved_usize(self.max_checkpoint_tiles, "checkpoint tiles")?,
                    max_cells: saved_usize(self.max_checkpoint_cells, "checkpoint cells")?,
                    max_private_memory_bytes: saved_usize(
                        self.max_checkpoint_private_memory_bytes,
                        "checkpoint private memory",
                    )?,
                },
            },
            manifest_limits: ReplayManifestLimits {
                max_manifest_bytes: saved_usize(self.max_manifest_bytes, "manifest bytes")?,
                max_segments: saved_usize(self.max_manifest_segments, "manifest segments")?,
            },
        })
    }
}

fn saved_usize(value: u64, field: &'static str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("saved {field} does not fit this platform"))
}

fn prepare_reference_wasm_chunk(
    observations: &ReferenceObservationBatch<'_>,
    work: &[(CellId, u64, u64)],
    limits: ReferenceMindLimits,
    random_deriver: &PrivateRandomDeriver,
) -> Result<Vec<(CellId, Vec<u8>)>, Error> {
    let mut prepared = Vec::with_capacity(work.len());
    let mut scratch_input = None;
    for (cell_id, lineage, sequence) in work {
        let actor = blob_engine::resolution::CellKey(
            u64::try_from(cell_id.0)
                .map_err(|_| Error::msg("cell ID does not fit reference key"))?,
        );
        let randomness = random_deriver.derive(*lineage, *sequence);
        if let Some(input) = scratch_input.as_mut() {
            observations
                .reference_mind_input_into(actor, randomness, input)
                .map_err(|error| Error::msg(error.to_string()))?;
        } else {
            scratch_input = Some(
                observations
                    .reference_mind_input(actor, randomness)
                    .map_err(|error| Error::msg(error.to_string()))?,
            );
        }
        let bytes = reference_mind_input_to_capnp(
            scratch_input
                .as_ref()
                .ok_or_else(|| Error::msg("reference Mind scratch input disappeared"))?,
            limits,
        )
        .map_err(|error| Error::msg(error.to_string()))?;
        prepared.push((*cell_id, bytes));
    }
    Ok(prepared)
}

impl Game {
    pub fn new(
        world_width: usize,
        world_height: usize,
        max_iterations: u64,
        cell_config: CellConfig,
        seed: Option<u64>,
        memory_config: MemoryConfig,
    ) -> Self {
        let actual_seed = seed.unwrap_or(0); // Use 0 as a default seed if None
        Self::new_with_match_secret(
            world_width,
            world_height,
            max_iterations,
            cell_config,
            Some(actual_seed),
            memory_config,
            match_secret_from_seed(actual_seed),
        )
    }

    /// Creates a game with an independently supplied decision-randomness
    /// secret. An authoritative server should generate this unpredictably and
    /// keep it out of public replay/submission data while the match is live.
    pub fn new_with_match_secret(
        world_width: usize,
        world_height: usize,
        max_iterations: u64,
        cell_config: CellConfig,
        _seed: Option<u64>,
        memory_config: MemoryConfig,
        match_secret: [u8; 32],
    ) -> Self {
        Game {
            world: World::new(world_width, world_height),
            teams: HashMap::new(),
            cells: HashMap::new(),
            coordinate_map: HashMap::new(),
            inv_coordinate_map: HashMap::new(),
            iteration: 0,
            max_iterations,
            cell_config,
            memory_config,
            next_cell_id: 0,
            match_secret,
            reference_engine: None,
        }
    }

    pub fn add_team(&mut self, team_id: TeamId, mind_path: &PathBuf) -> Result<(), String> {
        self.add_team_with_abi(team_id, mind_path)
    }

    fn add_team_with_abi(&mut self, team_id: TeamId, mind_path: &PathBuf) -> Result<(), String> {
        if self.reference_engine.is_some() || self.iteration != 0 {
            return Err("cannot add a team after authoritative execution starts".into());
        }
        if self.teams.contains_key(&team_id) {
            return Err("team already has a Mind pool".into());
        }
        if !mind_path.exists() {
            return Err(format!("Script file '{}' not found.", mind_path.display()));
        }

        let pool_size = self.effective_pool_size();
        let wasm_file = Wasm::file(mind_path.clone());
        let manifest = Manifest::new([wasm_file])
            .with_memory_options(
                MemoryOptions::new()
                    .with_max_pages(self.memory_config.max_pages)
                    .with_max_var_bytes(self.memory_config.max_var_bytes),
            )
            .disallow_all_hosts()
            .with_timeout(self.memory_config.timeout_duration());
        println!(
            "Created plugin pool of size {} for team {:?}",
            pool_size, team_id
        );
        let pool = PluginPool::new(
            manifest,
            pool_size,
            self.memory_config.use_pooling_allocator,
            self.memory_config.wasm_executor,
        )
        .map_err(|error| {
            format!(
                "Failed to compile mind script '{}': {error}. See diagnostics above.",
                mind_path.display()
            )
        })?;
        self.teams.insert(team_id, pool);
        // Place starting cells for this team in a cluster
        self.place_team_cluster(team_id)?;

        Ok(())
    }

    /// Get the effective plugin pool size, resolving 0 to available parallelism
    pub fn effective_pool_size(&self) -> usize {
        if self.memory_config.plugin_pool_size > 0 {
            self.memory_config.plugin_pool_size
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }
    }

    /// Place a team's starting cells in a spiral checkerboard cluster with a central plant
    fn place_team_cluster(&mut self, team_id: TeamId) -> Result<(), String> {
        let num_teams = self.teams.len();
        let team_index = team_id.0;

        // Calculate cluster center based on team index for even spacing
        let cluster_center = self.calculate_cluster_center(team_index, num_teams);

        // Find the maximum plant values from the energy config
        let max_plant_rate = self.get_max_plant_rate();
        let max_plant_energy = self.get_max_plant_energy();

        // Place a central plant at the cluster center
        self.world.set_energy_at(
            cluster_center,
            Some(EnergySource::Plant {
                rate: max_plant_rate,
                current_energy: max_plant_energy,
                max_energy: max_plant_energy,
            }),
        );

        // Generate spiral checkerboard positions around the center
        let cell_positions = self.generate_spiral_checkerboard_positions(
            cluster_center,
            self.cell_config.starting_cells_per_team,
        );

        // Place cells at the generated positions. Aggregate diagnostics: writing
        // one line per cell dominates setup for large online matches.
        let mut placed = 0usize;
        let mut occupied = 0usize;
        for position in cell_positions {
            // Check if position is valid and unoccupied
            if !self.coordinate_map.contains_key(&position) {
                let cell_id_val = self.next_cell_id;
                self.next_cell_id += 1;
                let new_cell_id = CellId(cell_id_val);
                let initial_energy_for_cell = self
                    .cell_config
                    .initial_energy
                    .min(self.cell_config.max_energy);
                let cell = Cell::new(
                    new_cell_id,
                    team_id,
                    initial_energy_for_cell,
                    self.cell_config.min_energy,
                );
                self.cells.insert(new_cell_id, cell);
                self.coordinate_map.insert(position, new_cell_id);
                self.inv_coordinate_map.insert(new_cell_id, position);
                placed += 1;
            } else {
                occupied += 1;
            }
        }
        println!("Placed {placed} starting cells for team {team_id:?}");
        if occupied != 0 {
            eprintln!(
                "Warning: skipped {occupied} occupied starting positions for team {team_id:?}"
            );
        }

        Ok(())
    }

    /// Calculate the center position for a team's cluster based on even spacing
    fn calculate_cluster_center(&self, team_index: usize, num_teams: usize) -> Coordinate {
        let world_width = self.world.dimensions.0;
        let world_height = self.world.dimensions.1;

        // Calculate grid dimensions for team placement
        let teams_per_row = (num_teams as f64).sqrt().ceil() as usize;
        let teams_per_col = (num_teams + teams_per_row - 1) / teams_per_row; // Ceiling division

        let row = team_index / teams_per_row;
        let col = team_index % teams_per_row;

        // Calculate spacing to evenly distribute clusters
        let spacing_x = world_width / teams_per_row;
        let spacing_y = world_height / teams_per_col;

        // Center each cluster within its allocated space
        let center_x = col * spacing_x + spacing_x / 2;
        let center_y = row * spacing_y + spacing_y / 2;

        Coordinate {
            x: center_x.min(world_width - 1),
            y: center_y.min(world_height - 1),
        }
    }

    /// Generate positions in a spiral checkerboard pattern around a center point
    /// Following the pattern:
    /// 3x3: 3X4    5x5: 10X11X12
    ///      X0X          X3X4X
    ///      2X1          9X0X5
    ///                   X2X1X
    ///                   8X7X6
    fn generate_spiral_checkerboard_positions(
        &self,
        center: Coordinate,
        num_cells: usize,
    ) -> Vec<Coordinate> {
        let mut positions = Vec::new();

        if num_cells == 0 {
            return positions;
        }

        let mut ring = 0; // Start with ring 0 (the center)
        let max_ring = center
            .x
            .max(self.world.dimensions.0 - 1 - center.x)
            .max(center.y)
            .max(self.world.dimensions.1 - 1 - center.y);

        while positions.len() < num_cells && ring <= max_ring {
            // Generate positions for current ring
            let ring_positions = self.generate_ring_checkerboard_positions(center, ring);

            for pos in ring_positions {
                if positions.len() >= num_cells {
                    break;
                }
                // Only add valid positions within world bounds
                if pos.x < self.world.dimensions.0 && pos.y < self.world.dimensions.1 {
                    positions.push(pos);
                }
            }

            ring += 1;
        }

        positions
    }

    /// Generate checkerboard positions for a specific ring around the center
    fn generate_ring_checkerboard_positions(
        &self,
        center: Coordinate,
        ring: usize,
    ) -> Vec<Coordinate> {
        let mut positions = Vec::new();

        if ring == 0 {
            // Ring 0 is just the center
            positions.push(center);
            return positions;
        }

        let ring_i = ring as i32;

        // Calculate the center's checkerboard parity
        let center_parity = (center.x + center.y) % 2;

        // Generate all positions in the ring
        for dx in -ring_i..=ring_i {
            for dy in -ring_i..=ring_i {
                // Check if this position is on the ring boundary
                if dx.abs() == ring_i || dy.abs() == ring_i {
                    let x = center.x as i32 + dx;
                    let y = center.y as i32 + dy;

                    // Check bounds
                    if x >= 0 && y >= 0 {
                        let coord = Coordinate {
                            x: x as usize,
                            y: y as usize,
                        };

                        // Apply checkerboard pattern (match the center's parity)
                        if (coord.x + coord.y) % 2 == center_parity {
                            positions.push(coord);
                        }
                    }
                }
            }
        }

        positions
    }

    /// Get the maximum plant rate from the energy configuration  
    fn get_max_plant_rate(&self) -> u32 {
        // Since we don't have direct access to the energy config here,
        // we'll use a reasonable maximum based on typical config values
        // This could be made configurable in the future
        50 // High rate for central cluster plants
    }

    /// Get the maximum plant energy from the energy configuration
    fn get_max_plant_energy(&self) -> u32 {
        // Since we don't have direct access to the energy config here,
        // we'll use a reasonable maximum based on typical config values
        // This could be made configurable in the future
        300 // High energy for central cluster plants
    }

    pub fn run(&mut self, verbose: bool) -> Result<(), Error> {
        while self.iteration < self.max_iterations {
            if verbose {
                println!("Iteration {}", self.iteration);
            }
            self.tick(verbose)?;
        }
        Ok(())
    }

    /// Run a specified number of game steps (iterations)
    /// Returns the number of steps actually executed (may be less if max_iterations is reached)
    pub fn step(&mut self, steps: u64, verbose: bool) -> Result<u64, Error> {
        let mut executed_steps = 0;
        for _ in 0..steps {
            if self.iteration >= self.max_iterations {
                break;
            }
            if verbose {
                println!("Iteration {}", self.iteration);
            }
            self.tick(verbose)?;
            executed_steps += 1;
        }
        Ok(executed_steps)
    }

    fn get_reference_wasm_decisions_for(
        &mut self,
        eligible: &[CellId],
    ) -> Result<Vec<(CellId, ReferenceMindDecision)>, Error> {
        let mut cell_ids = eligible.to_vec();
        cell_ids.sort_by_key(|cell_id| cell_id.0);
        let mut invocations = Vec::with_capacity(cell_ids.len());
        for cell_id in cell_ids {
            let cell = self
                .reference_engine
                .as_mut()
                .and_then(|engine| engine.cells.get_mut(&cell_id))
                .ok_or_else(|| Error::msg("reference WASM cell disappeared"))?;
            let lineage = cell.random_lineage;
            let sequence = cell.decision_sequence;
            cell.decision_sequence = cell
                .decision_sequence
                .checked_add(1)
                .ok_or_else(|| Error::msg("cell decision sequence overflow"))?;
            invocations.push((cell_id, cell.team_id, lineage, sequence));
        }

        let match_secret = self.match_secret;
        let random_deriver = PrivateRandomDeriver::new(&match_secret);
        let mut work_by_team: HashMap<TeamId, Vec<(CellId, u64, u64)>> = HashMap::new();
        for (cell_id, team_id, lineage, sequence) in invocations {
            work_by_team
                .entry(team_id)
                .or_default()
                .push((cell_id, lineage, sequence));
        }

        let simulation = self
            .reference_engine
            .as_ref()
            .and_then(Engine::reference_simulation)
            .ok_or_else(|| Error::msg("reference simulation was not initialized"))?;
        let limits = ReferenceMindLimits {
            max_private_memory_bytes: simulation.rules().max_private_memory_bytes,
            ..ReferenceMindLimits::default()
        };
        let observations = simulation.observation_batch();
        let mut decisions = Vec::new();
        for (team_id, work) in work_by_team {
            let pool = self
                .teams
                .get(&team_id)
                .ok_or_else(|| Error::msg(format!("team {team_id:?} has no WASM pool")))?;
            let active_workers = pool.active_worker_count(work.len());
            let run_parallel =
                active_workers > 1 && work.len() >= PARALLEL_WASM_OBSERVATION_THRESHOLD;
            if run_parallel {
                let chunk_size = work.len().div_ceil(active_workers);
                let chunks: Vec<Result<Vec<(CellId, ReferenceMindDecision)>, Error>> = work
                    .par_chunks(chunk_size)
                    .enumerate()
                    .map(|(worker, chunk)| {
                        let inputs = prepare_reference_wasm_chunk(
                            &observations,
                            chunk,
                            limits,
                            &random_deriver,
                        )?;
                        pool.process_reference_worker_batch(worker, inputs, limits)
                            .map_err(Error::msg)
                    })
                    .collect();
                for chunk in chunks {
                    decisions.extend(chunk?);
                }
            } else {
                let inputs =
                    prepare_reference_wasm_chunk(&observations, &work, limits, &random_deriver)?;
                decisions.extend(
                    pool.process_reference_batch(inputs, limits)
                        .map_err(Error::msg)?,
                );
            }
        }
        decisions.sort_by_key(|(cell_id, _)| cell_id.0);
        Ok(decisions)
    }

    fn ensure_reference_engine(&mut self) -> Result<(), Error> {
        if self.reference_engine.is_some() {
            return Ok(());
        }
        let config = EngineCellConfig {
            min_energy: self.cell_config.min_energy,
            initial_energy: self.cell_config.initial_energy,
            starting_cells_per_team: 0,
            max_energy: self.cell_config.max_energy,
            min_attack_power: self.cell_config.min_attack_power,
            max_attack_power: self.cell_config.max_attack_power,
            max_energy_for_attack_scaling: self.cell_config.max_energy_for_attack_scaling,
        };
        let mut engine = Engine::new_with_match_secret(
            self.world.dimensions.0,
            self.world.dimensions.1,
            self.max_iterations,
            config,
            Some(0),
            self.match_secret,
            ReferenceRuleset::default(),
        );
        engine
            .replace_setup_projection(
                self.world.clone(),
                self.cells.clone(),
                self.coordinate_map.clone(),
                self.inv_coordinate_map.clone(),
            )
            .map_err(Error::msg)?;
        self.reference_engine = Some(engine);
        Ok(())
    }

    fn sync_reference_host_metadata(&mut self) {
        let Some(engine) = self.reference_engine.as_mut() else {
            return;
        };
        if engine.reference_simulation().is_some() {
            return;
        }
        for (cell_id, cell) in &self.cells {
            if let Some(projected) = engine.cells.get_mut(cell_id) {
                projected.team_id = cell.team_id;
                projected.age = cell.age;
                projected.random_lineage = cell.random_lineage;
                projected.decision_sequence = cell.decision_sequence;
                projected.message_queue = cell.message_queue.clone();
            }
        }
    }

    fn tick_reference(&mut self, verbose: bool) -> Result<TickEvents, Error> {
        self.tick_reference_with_host_mode(verbose, ReferenceHostMode::Projected)
    }

    fn tick_reference_with_host_mode(
        &mut self,
        verbose: bool,
        host_mode: ReferenceHostMode,
    ) -> Result<TickEvents, Error> {
        self.ensure_reference_engine()?;
        self.reference_engine
            .as_mut()
            .expect("reference engine initialized")
            .set_reference_host_mode(host_mode)
            .map_err(Error::msg)?;
        self.reference_engine
            .as_mut()
            .expect("reference engine initialized")
            .initialize_reference_state()
            .map_err(Error::msg)?;
        if self
            .reference_engine
            .as_ref()
            .is_some_and(Engine::reference_replay_stream_needs_drain)
        {
            return Err(Error::msg(
                "sealed replay segment must be drained before continuing",
            ));
        }
        let ready = self
            .reference_engine
            .as_ref()
            .expect("reference engine initialized")
            .ready_cell_ids();
        let reference_ready = ready.clone();
        let decision_sequences_before: Vec<(CellId, u64)> = ready
            .iter()
            .filter_map(|cell_id| {
                self.reference_engine
                    .as_ref()
                    .and_then(|engine| engine.cells.get(cell_id))
                    .map(|cell| (*cell_id, cell.decision_sequence))
            })
            .collect();
        let decisions: HashMap<CellId, ReferenceMindDecision> =
            match self.get_reference_wasm_decisions_for(&reference_ready) {
                Ok(decisions) => decisions.into_iter().collect(),
                Err(error) => {
                    for (cell_id, sequence) in decision_sequences_before {
                        if let Some(cell) = self
                            .reference_engine
                            .as_mut()
                            .and_then(|engine| engine.cells.get_mut(&cell_id))
                        {
                            cell.decision_sequence = sequence;
                        }
                    }
                    return Err(error);
                }
            };

        let tick_events = self
            .reference_engine
            .as_mut()
            .expect("reference engine initialized")
            .tick_reference_with_overrides(&decisions, verbose)
            .map_err(Error::msg)?;
        match host_mode {
            ReferenceHostMode::Projected => self.apply_reference_projection_delta(&tick_events)?,
            ReferenceHostMode::MetadataOnly => {
                let engine = self
                    .reference_engine
                    .as_ref()
                    .ok_or_else(|| Error::msg("reference engine was not initialized"))?;
                self.iteration = engine.iteration;
                self.next_cell_id = engine.next_cell_id();
            }
        }
        Ok(tick_events)
    }

    fn apply_reference_projection_delta(&mut self, events: &TickEvents) -> Result<(), Error> {
        events
            .reference_batch
            .as_ref()
            .ok_or_else(|| Error::msg("reference tick did not return a canonical batch"))?;
        let engine = self
            .reference_engine
            .as_ref()
            .ok_or_else(|| Error::msg("reference engine was not initialized"))?;

        // Private dispatch counters are not renderer fields, so a visual no-op
        // may omit this cell from the projection invalidation set. Keep the
        // projected save envelope current without making the trusted path pay
        // for this copy.
        for commitment in &events.reference_commitments {
            let id = CellId(
                usize::try_from(commitment.actor.0)
                    .map_err(|_| Error::msg("committed cell key does not fit CellId"))?,
            );
            if let (Some(source), Some(target)) = (engine.cells.get(&id), self.cells.get_mut(&id)) {
                target.decision_sequence = source.decision_sequence;
            }
        }
        for cell in self.cells.values_mut() {
            cell.age = cell.age.saturating_add(1);
        }
        for id in &events.reference_projection_cells {
            let old_coordinate = self.inv_coordinate_map.get(id).copied();
            let new_coordinate = engine.inv_coordinate_map.get(id).copied();
            if old_coordinate == new_coordinate {
                continue;
            }
            if let Some(coordinate) = self.inv_coordinate_map.remove(id) {
                if self.coordinate_map.get(&coordinate) == Some(id) {
                    self.coordinate_map.remove(&coordinate);
                }
            }
        }
        for id in &events.reference_projection_cells {
            let Some(cell) = engine.cells.get(id) else {
                self.cells.remove(id);
                continue;
            };
            self.cells.insert(*id, cell.clone());
            let coordinate = *engine.inv_coordinate_map.get(id).ok_or_else(|| {
                Error::msg("changed cell coordinate is missing from engine projection")
            })?;
            if self.inv_coordinate_map.get(id) != Some(&coordinate) {
                if self.coordinate_map.insert(coordinate, *id).is_some() {
                    return Err(Error::msg("multiple projected cells occupy one tile"));
                }
                self.inv_coordinate_map.insert(*id, coordinate);
            }
        }
        for &index in &events.reference_projection_tiles {
            if index >= self.world.elevation.len()
                || index >= self.world.energy.len()
                || index >= self.world.pheromone.len()
            {
                return Err(Error::msg("changed tile is outside the game projection"));
            }
            self.world.elevation[index] = engine.world.elevation[index];
            self.world.energy[index] = engine.world.energy[index].clone();
            self.world.pheromone[index] = engine.world.pheromone[index];
        }
        self.iteration = engine.iteration;
        self.next_cell_id = engine.next_cell_id();
        Ok(())
    }

    /// Runs the normal pristine-instance WASM decision path and returns the
    /// canonical commitments/report required by server replay verification.
    pub fn tick_reference_for_verification(&mut self, verbose: bool) -> Result<TickEvents, Error> {
        self.tick_reference_with_host_mode(verbose, ReferenceHostMode::MetadataOnly)
    }

    /// Materializes the authoritative start frontier without invoking a Mind.
    pub fn initialize_reference_for_verification(&mut self) -> Result<(), Error> {
        self.ensure_reference_engine()?;
        self.sync_reference_host_metadata();
        self.reference_engine
            .as_mut()
            .expect("reference engine initialized")
            .set_reference_host_mode(ReferenceHostMode::MetadataOnly)
            .map_err(Error::msg)?;
        self.reference_engine
            .as_mut()
            .expect("reference engine initialized")
            .initialize_reference_state()
            .map_err(Error::msg)
    }

    pub fn reference_state_hash(&self) -> Option<blob_engine::resolution::CanonicalHash> {
        self.reference_engine
            .as_ref()
            .and_then(Engine::authoritative_state_hash)
    }

    pub fn reference_canonical_state_hash(&self) -> Option<CanonicalHash> {
        self.reference_engine
            .as_ref()?
            .reference_simulation()
            .map(|simulation| simulation.state_hash())
    }

    pub fn reference_compiled_ruleset_hash(&self) -> Option<CanonicalHash> {
        self.reference_engine
            .as_ref()?
            .reference_simulation()
            .map(|simulation| simulation.compiled_ruleset_hash())
    }

    pub fn start_reference_replay_recording(
        &mut self,
        checkpoint_interval: u64,
    ) -> Result<(), Error> {
        self.ensure_reference_engine()?;
        self.sync_reference_host_metadata();
        self.reference_engine
            .as_mut()
            .ok_or_else(|| Error::msg("reference engine was not initialized"))?
            .start_reference_replay_recording(checkpoint_interval)
            .map_err(Error::msg)
    }

    pub fn export_reference_replay_bundle(&self) -> Result<ReplayBundle, Error> {
        self.reference_engine
            .as_ref()
            .ok_or_else(|| Error::msg("reference engine was not initialized"))?
            .export_reference_replay_bundle(ReplayBundleLimits::default())
            .map_err(Error::msg)
    }

    pub fn start_reference_replay_streaming(
        &mut self,
        config: ReplayStreamConfig,
    ) -> Result<(), Error> {
        self.ensure_reference_engine()?;
        self.sync_reference_host_metadata();
        self.reference_engine
            .as_mut()
            .ok_or_else(|| Error::msg("reference engine was not initialized"))?
            .start_reference_replay_streaming(config)
            .map_err(Error::msg)
    }

    pub fn reference_replay_stream_manifest(&self) -> Result<ReplayManifest, Error> {
        self.reference_engine
            .as_ref()
            .ok_or_else(|| Error::msg("reference engine was not initialized"))?
            .reference_replay_stream_manifest()
            .map_err(Error::msg)
    }

    pub fn take_reference_replay_segment(&mut self) -> Option<ReplaySegment> {
        self.reference_engine
            .as_mut()
            .and_then(Engine::take_reference_replay_segment)
    }

    pub fn flush_reference_replay_stream(&mut self) -> Result<bool, Error> {
        self.reference_engine
            .as_mut()
            .ok_or_else(|| Error::msg("reference engine was not initialized"))?
            .flush_reference_replay_stream()
            .map_err(Error::msg)
    }

    pub fn tick(&mut self, verbose: bool) -> Result<(), Error> {
        self.tick_reference(verbose)?;
        Ok(())
    }

    /// Enhanced tick method that includes automatic state saving
    pub fn tick_with_auto_save(
        &mut self,
        verbose: bool,
        state_config: &StateConfig,
    ) -> Result<(), Error> {
        self.tick(verbose)?;

        // Auto-save state if configured
        if let Err(e) = self.auto_save_state(state_config) {
            eprintln!("Warning: Failed to auto-save state: {}", e);
        }

        Ok(())
    }

    /// Generate an image representation of the current game state
    ///
    /// Creates a PNG image showing:
    /// - Terrain elevation (darker = higher elevation)
    /// - Energy sources (green for scattered energy, bright green for plants)
    /// - Cells (colored by team ID, brightness based on energy)
    /// - Pheromones (blue overlay)
    ///
    /// Returns the image data as RgbImage for use in GUI applications.
    /// Optionally saves to disk if save_to_disk is true and filename is provided.
    pub fn generate_board_image(
        &self,
        scale: u32,
        filename: Option<&str>,
        save_to_disk: bool,
    ) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let width = self.world.dimensions.0 as u32;
        let height = self.world.dimensions.1 as u32;
        let img_width = width * scale;
        let img_height = height * scale;

        let mut img: RgbImage = ImageBuffer::new(img_width, img_height);

        // Find min/max elevation for normalization
        let min_elevation = self.world.elevation.iter().min().copied().unwrap_or(0);
        let max_elevation = self.world.elevation.iter().max().copied().unwrap_or(0);
        let elevation_range = max_elevation - min_elevation;

        // Define team colors (cycling through a palette)
        let team_colors = [
            [255, 100, 100], // Red
            [100, 100, 255], // Blue
            [255, 255, 100], // Yellow
            [255, 100, 255], // Magenta
            [100, 255, 255], // Cyan
            [255, 165, 0],   // Orange
            [128, 0, 128],   // Purple
            [0, 128, 0],     // Dark Green
            [165, 42, 42],   // Brown
            [255, 20, 147],  // Deep Pink
        ];

        // Compute tile colors in parallel using rayon
        // Extract field refs to avoid capturing &self (which contains non-Sync PluginPool)
        let world_width = self.world.dimensions.0;
        let elevation_ref = &self.world.elevation;
        let energy_ref = &self.world.energy;
        let pheromone_ref = &self.world.pheromone;
        let coord_map_ref = &self.coordinate_map;
        let cells_ref = &self.cells;
        let max_energy = self.cell_config.max_energy;

        let tile_colors: Vec<[u8; 3]> = (0..(width as usize * height as usize))
            .into_par_iter()
            .map(|idx| {
                let x = idx % world_width;
                let y = idx / world_width;
                let coord = Coordinate { x, y };

                // Base color: terrain elevation (grayscale)
                let elevation = elevation_ref[idx];
                let normalized_elevation = if elevation_range > 0 {
                    ((elevation - min_elevation) as f32 / elevation_range as f32 * 128.0) as u8 + 64
                } else {
                    128
                };

                let mut base_color = [
                    normalized_elevation,
                    normalized_elevation,
                    normalized_elevation,
                ];

                // Energy sources overlay
                if let Some(energy_source) = &energy_ref[idx] {
                    match energy_source {
                        EnergySource::Scattered(amount) => {
                            let intensity = (*amount as f32 / 200.0).min(1.0);
                            base_color[1] =
                                (base_color[1] as f32 + intensity * 100.0).min(255.0) as u8;
                        }
                        EnergySource::Plant {
                            current_energy,
                            max_energy,
                            ..
                        } => {
                            let intensity = (*current_energy as f32 / *max_energy as f32).min(1.0);
                            base_color[0] = (base_color[0] as f32 * (1.0 - intensity * 0.5)) as u8;
                            base_color[1] = (255.0 * intensity
                                + base_color[1] as f32 * (1.0 - intensity))
                                as u8;
                            base_color[2] = (base_color[2] as f32 * (1.0 - intensity * 0.5)) as u8;
                        }
                    }
                }

                // Pheromone overlay (blue tint)
                if let Some(pheromone) = pheromone_ref[idx] {
                    let intensity = (pheromone as f32 / 255.0).min(1.0);
                    base_color[2] = (base_color[2] as f32 + intensity * 80.0).min(255.0) as u8;
                }

                // Cell overlay (dominant color)
                if let Some(cell_id) = coord_map_ref.get(&coord) {
                    if let Some(cell) = cells_ref.get(cell_id) {
                        let team_index = cell.team_id.0 % team_colors.len();
                        let team_color = team_colors[team_index];

                        let energy_ratio = (cell.energy as f32 / max_energy as f32).min(1.0);
                        let brightness = 0.3 + energy_ratio * 0.7;

                        base_color[0] = (team_color[0] as f32 * brightness) as u8;
                        base_color[1] = (team_color[1] as f32 * brightness) as u8;
                        base_color[2] = (team_color[2] as f32 * brightness) as u8;

                        if cell.defending {
                            base_color = [255, 255, 255];
                        } else if cell.loaded {
                            base_color[0] = (base_color[0] as f32 * 0.7) as u8;
                            base_color[1] = (base_color[1] as f32 * 0.7) as u8;
                            base_color[2] = (base_color[2] as f32 * 0.7) as u8;
                        }
                    }
                }

                base_color
            })
            .collect();

        // Write scaled pixels from precomputed tile colors
        for y in 0..height {
            for x in 0..width {
                let tile_idx = y as usize * world_width + x as usize;
                let base_color = tile_colors[tile_idx];

                for dy in 0..scale {
                    for dx in 0..scale {
                        let pixel_x = x * scale + dx;
                        let pixel_y = y * scale + dy;
                        if pixel_x < img_width && pixel_y < img_height {
                            img.put_pixel(pixel_x, pixel_y, Rgb(base_color));
                        }
                    }
                }
            }
        }

        // Optionally save the image to disk
        if save_to_disk {
            if let Some(file_name) = filename {
                img.save(file_name)?;
                println!("Board image saved to: {}", file_name);
            } else {
                return Err("Filename must be provided when save_to_disk is true".into());
            }
        }

        Ok(img)
    }

    /// Generate an image for the current iteration with automatic filename
    ///
    /// Creates a PNG image with filename format: "board_iteration_{iteration}.png"
    /// This is useful for creating animations or tracking game progress over time.
    /// Returns the image data and optionally saves to disk.
    pub fn generate_iteration_image(
        &self,
        scale: u32,
        save_to_disk: bool,
    ) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let filename = format!("board_iteration_{:06}.png", self.iteration);
        self.generate_board_image(scale, Some(&filename), save_to_disk)
    }

    /// Save the current game state to a file
    ///
    /// Serializes the game state (excluding WebAssembly plugins) and optionally compresses it
    pub fn save_state(
        &mut self,
        filename: &str,
        compress: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.ensure_reference_engine()?;
        self.sync_reference_host_metadata();
        let engine = self
            .reference_engine
            .as_mut()
            .ok_or("reference engine was not initialized")?;
        if engine.reference_host_mode() == ReferenceHostMode::MetadataOnly {
            return Err("metadata-only verification does not maintain a renderer-compatible game save; export the canonical checkpoint and replay instead".into());
        }
        engine.initialize_reference_state().map_err(Error::msg)?;
        let runtime = engine.export_reference_checkpoint().map_err(Error::msg)?;
        let runtime_hash = engine
            .authoritative_state_hash()
            .ok_or("reference runtime hash is unavailable")?;
        let reference = ReferenceSaveState::from_runtime(runtime, runtime_hash);
        let state = GameState {
            world: self.world.clone(),
            cells: self.cells.clone(),
            coordinate_map: self.coordinate_map.clone(),
            inv_coordinate_map: self.inv_coordinate_map.clone(),
            iteration: self.iteration,
            max_iterations: self.max_iterations,
            cell_config: self.cell_config.clone(),
            memory_config: self.memory_config.clone(),
            next_cell_id: self.next_cell_id,
            match_secret: self.match_secret,
            reference,
        };

        let payload = bincode::serialize(&state)?;
        let mut serialized = Vec::with_capacity(GAME_STATE_MAGIC.len() + 2 + payload.len());
        serialized.extend_from_slice(GAME_STATE_MAGIC);
        serialized.extend_from_slice(&GAME_STATE_VERSION.to_le_bytes());
        serialized.extend_from_slice(&payload);
        if serialized.len() > MAX_GAME_STATE_BYTES {
            return Err(format!(
                "game state is {} bytes; limit is {}",
                serialized.len(),
                MAX_GAME_STATE_BYTES
            )
            .into());
        }

        if compress {
            let file = fs::File::create(filename)?;
            let mut encoder = GzEncoder::new(file, Compression::default());
            encoder.write_all(&serialized)?;
            encoder.finish()?;
        } else {
            fs::write(filename, serialized)?;
        }

        println!("Game state saved to: {}", filename);
        Ok(())
    }

    /// Load a game state from a file
    ///
    /// Deserializes the game state and returns it. The caller must reconstruct
    /// the WebAssembly plugins using `from_state_with_teams`.
    pub fn load_state(
        filename: &str,
        compressed: bool,
    ) -> Result<GameState, Box<dyn std::error::Error>> {
        let data = if compressed {
            let file = fs::File::open(filename)?;
            let mut decoder = GzDecoder::new(file).take((MAX_GAME_STATE_BYTES + 1) as u64);
            let mut buffer = Vec::new();
            decoder.read_to_end(&mut buffer)?;
            buffer
        } else {
            fs::read(filename)?
        };
        if data.len() > MAX_GAME_STATE_BYTES {
            return Err(format!(
                "game state is {} bytes; limit is {}",
                data.len(),
                MAX_GAME_STATE_BYTES
            )
            .into());
        }

        if !data.starts_with(GAME_STATE_MAGIC) || data.len() < GAME_STATE_MAGIC.len() + 2 {
            return Err("game state is not a current versioned checkpoint".into());
        }
        let version = u16::from_le_bytes(
            data[GAME_STATE_MAGIC.len()..GAME_STATE_MAGIC.len() + 2]
                .try_into()
                .map_err(|_| "versioned game state header is truncated")?,
        );
        if version != GAME_STATE_VERSION {
            return Err(format!(
                "unsupported game state version {version}; expected {GAME_STATE_VERSION}"
            )
            .into());
        }
        let state = bincode::deserialize(&data[GAME_STATE_MAGIC.len() + 2..])?;
        println!("Game state loaded from: {}", filename);
        Ok(state)
    }

    /// Create a new Game instance from a saved GameState
    ///
    /// This reconstructs the WebAssembly plugins and allows continuing the
    /// canonical simulation from a saved state.
    pub fn from_state_with_teams(
        state: GameState,
        team_mind_paths: &[(TeamId, PathBuf)],
        _seed_override: Option<u64>,
    ) -> Result<Self, String> {
        let GameState {
            world,
            cells,
            coordinate_map,
            inv_coordinate_map,
            iteration,
            max_iterations,
            cell_config,
            memory_config,
            next_cell_id,
            match_secret,
            reference,
        } = state;
        let mut game = Game {
            world,
            teams: HashMap::new(),
            match_secret,
            cells,
            coordinate_map,
            inv_coordinate_map,
            iteration,
            max_iterations,
            cell_config,
            memory_config,
            next_cell_id,
            reference_engine: None,
        };

        // Reconstruct WebAssembly plugin pools without placing new starting cells
        let pool_size = game.effective_pool_size();
        for (team_id, mind_path) in team_mind_paths {
            if !mind_path.exists() {
                return Err(format!("Script file '{}' not found.", mind_path.display()));
            }

            let wasm_file = Wasm::file(mind_path.clone());
            let manifest = Manifest::new([wasm_file])
                .with_memory_options(
                    MemoryOptions::new()
                        .with_max_pages(game.memory_config.max_pages)
                        .with_max_var_bytes(game.memory_config.max_var_bytes),
                )
                .disallow_all_hosts()
                .with_timeout(game.memory_config.timeout_duration());
            let pool = PluginPool::new(
                manifest,
                pool_size,
                game.memory_config.use_pooling_allocator,
                game.memory_config.wasm_executor,
            )
            .map_err(|error| {
                format!(
                    "Failed to compile mind script '{}': {error}. See diagnostics above.",
                    mind_path.display()
                )
            })?;

            game.teams.insert(*team_id, pool);
            println!(
                "Reconstructed team {:?} with pool size {} from {:?}",
                team_id, pool_size, mind_path
            );
        }

        let expected_hash = CanonicalHash::from_bytes(reference.authoritative_runtime_hash);
        let runtime = reference.into_runtime()?;
        game.ensure_reference_engine()
            .map_err(|error| error.to_string())?;
        let engine = game
            .reference_engine
            .as_mut()
            .ok_or("reference engine was not initialized")?;
        engine.restore_reference_checkpoint(runtime)?;
        let actual_hash = engine
            .authoritative_state_hash()
            .ok_or("restored reference runtime hash is unavailable")?;
        if actual_hash != expected_hash {
            return Err(format!(
                "restored authoritative runtime hash mismatch: expected {expected_hash}, got {actual_hash}"
            ));
        }
        game.world = engine.world.clone();
        game.cells = engine.cells.clone();
        game.coordinate_map = engine.coordinate_map.clone();
        game.inv_coordinate_map = engine.inv_coordinate_map.clone();
        game.iteration = engine.iteration;
        game.next_cell_id = engine.next_cell_id();

        Ok(game)
    }

    /// Save state automatically based on configuration
    ///
    /// Uses the state configuration to determine filename, compression, etc.
    pub fn auto_save_state(
        &mut self,
        state_config: &StateConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.iteration % state_config.save_interval == 0 {
            // Create state directory if it doesn't exist
            fs::create_dir_all(&state_config.state_directory)?;

            let filename = format!(
                "{}/game_state_{:06}.{}",
                state_config.state_directory,
                self.iteration,
                if state_config.compress_states {
                    "gz"
                } else {
                    "bin"
                }
            );

            self.save_state(&filename, state_config.compress_states)?;

            // Clean up old states if auto_cleanup is enabled
            if state_config.auto_cleanup {
                self.cleanup_old_states(state_config)?;
            }
        }
        Ok(())
    }

    /// Clean up old state files based on configuration
    fn cleanup_old_states(
        &self,
        state_config: &StateConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dir = fs::read_dir(&state_config.state_directory)?;
        let mut state_files: Vec<_> = dir
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.is_file() {
                    let filename = path.file_name()?.to_str()?;
                    if filename.starts_with("game_state_") {
                        Some((path.clone(), filename.to_string()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Sort by filename (which includes iteration number)
        state_files.sort_by(|a, b| a.1.cmp(&b.1));

        // Remove oldest files if we exceed the limit
        if state_files.len() > state_config.max_disk_states {
            let files_to_remove = state_files.len() - state_config.max_disk_states;
            for (path, _) in state_files.iter().take(files_to_remove) {
                fs::remove_file(path)?;
                println!("Cleaned up old state file: {:?}", path);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CellConfig, MemoryConfig};
    use crate::world_gen::generate_terrain;
    use blob_engine::resolution::ActionRequest;
    use blob_interface::reference_mind::ReferenceMemoryUpdate;
    use std::path::PathBuf;

    /// Helper: Create a game with given pool_size and seed
    fn create_test_game(pool_size: usize, seed: u64) -> Game {
        let mut memory_config = MemoryConfig::default();
        memory_config.plugin_pool_size = pool_size;
        Game::new(
            64,
            64,
            1000,
            CellConfig::default(),
            Some(seed),
            memory_config,
        )
    }

    fn create_reference_test_game(pool_size: usize, seed: u64) -> Game {
        let mut memory_config = MemoryConfig::default();
        memory_config.plugin_pool_size = pool_size;
        Game::new(
            64,
            64,
            1000,
            CellConfig::default(),
            Some(seed),
            memory_config,
        )
    }

    /// Helper: Get path to a WASM mind, return None if not found
    fn wasm_mind_path(name: &str) -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("target/wasm32-unknown-unknown/release")
            .join(format!("{name}.wasm"));
        if path.exists() { Some(path) } else { None }
    }

    fn language_mind_path(language: &str, name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("blob_game must be in the workspace")
            .join("minds")
            .join(language)
            .join("build")
            .join(format!("{name}.wasm"))
    }

    fn assert_game_matches_reference_engine(game: &Game) {
        let engine = game.reference_engine.as_ref().unwrap();
        assert!(game.world == engine.world);
        assert_eq!(game.cells, engine.cells);
        assert_eq!(game.coordinate_map, engine.coordinate_map);
        assert_eq!(game.inv_coordinate_map, engine.inv_coordinate_map);
        assert_eq!(game.iteration, engine.iteration);
        assert_eq!(game.next_cell_id, engine.next_cell_id());
    }

    #[test]
    fn test_game_creation_sequential() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };
        let mut game = create_test_game(1, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        assert_eq!(game.teams.len(), 1);
        assert_eq!(game.teams[&TeamId(0)].len(), 1);
    }

    #[test]
    fn test_game_creation_parallel() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };
        let mut game = create_test_game(4, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        assert_eq!(game.teams[&TeamId(0)].len(), 4);
    }

    #[test]
    fn test_duplicate_wasm_team_registration_is_rejected_without_more_cells() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };
        let mut game = create_test_game(1, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        let cells_before = game.cells.len();

        let error = game.add_team(TeamId(0), &mind_path).unwrap_err();

        assert!(error.contains("already has a Mind pool"));
        assert_eq!(game.cells.len(), cells_before);
        assert_eq!(game.teams.len(), 1);
    }

    #[test]
    fn test_starting_cells_placed() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };
        let mut game = create_test_game(1, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        assert_eq!(game.cells.len(), game.cell_config.starting_cells_per_team);
        for (_, cell) in &game.cells {
            assert_eq!(cell.team_id, TeamId(0));
            assert_eq!(cell.energy, game.cell_config.initial_energy);
        }
    }

    #[test]
    fn large_starting_cluster_uses_the_available_board_instead_of_a_fixed_ring_limit() {
        let game = Game::new(
            128,
            128,
            1,
            CellConfig::default(),
            Some(42),
            MemoryConfig::default(),
        );
        let positions =
            game.generate_spiral_checkerboard_positions(Coordinate { x: 64, y: 64 }, 5_000);
        let unique: std::collections::HashSet<_> = positions.iter().copied().collect();

        assert_eq!(positions.len(), 5_000);
        assert_eq!(unique.len(), positions.len());
        assert!(positions.iter().all(|position| {
            position.x < game.world.dimensions.0 && position.y < game.world.dimensions.1
        }));
    }

    #[test]
    fn test_multiple_teams() {
        let Some(mind1) = wasm_mind_path("simple_mind") else {
            return;
        };
        let Some(mind2) = wasm_mind_path("aggressive_mind") else {
            return;
        };
        let mut game = create_test_game(2, 42);
        game.add_team(TeamId(0), &mind1).unwrap();
        game.add_team(TeamId(1), &mind2).unwrap();
        assert_eq!(game.teams.len(), 2);
        assert_eq!(game.teams[&TeamId(0)].len(), 2);
        assert_eq!(game.teams[&TeamId(1)].len(), 2);
    }

    #[test]
    fn test_tick_sequential() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            return;
        };
        let mut game = create_test_game(1, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        game.world.elevation = generate_terrain(64, 64, Some(42), 10);
        game.tick(false).unwrap();
        assert_eq!(game.iteration, 1);
    }

    #[test]
    fn test_tick_parallel() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            return;
        };
        let mut game = create_test_game(4, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        game.world.elevation = generate_terrain(64, 64, Some(42), 10);
        game.tick(false).unwrap();
        assert_eq!(game.iteration, 1);
    }

    #[test]
    fn test_cell_aging() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            return;
        };
        let mut game = create_test_game(1, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        game.world.elevation = generate_terrain(64, 64, Some(42), 10);

        for (_, cell) in &game.cells {
            assert_eq!(cell.age, 0);
        }
        game.tick(false).unwrap();
        for (_, cell) in &game.cells {
            assert_eq!(cell.age, 1);
        }
        game.tick(false).unwrap();
        for (_, cell) in &game.cells {
            assert_eq!(cell.age, 2);
        }
    }

    #[test]
    fn test_effective_pool_size() {
        let game = create_test_game(4, 42);
        assert_eq!(game.effective_pool_size(), 4);
        let game_auto = create_test_game(0, 42);
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert_eq!(game_auto.effective_pool_size(), available);
    }

    #[test]
    fn test_step_advances_iteration() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            return;
        };
        let mut game = create_test_game(1, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        let executed = game.step(10, false).unwrap();
        assert_eq!(executed, 10);
        assert_eq!(game.iteration, 10);
    }

    #[test]
    fn test_step_respects_max_iterations() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            return;
        };
        let mut memory_config = MemoryConfig::default();
        memory_config.plugin_pool_size = 1;
        let mut game = Game::new(32, 32, 5, CellConfig::default(), Some(42), memory_config);
        game.add_team(TeamId(0), &mind_path).unwrap();
        let executed = game.step(100, false).unwrap();
        assert_eq!(executed, 5);
    }

    #[test]
    fn test_image_generation() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            return;
        };
        let mut game = create_test_game(1, 42);
        game.add_team(TeamId(0), &mind_path).unwrap();
        let img = game.generate_board_image(4, None, false).unwrap();
        assert_eq!(img.width(), 256); // 64 * 4
        assert_eq!(img.height(), 256);
    }

    #[test]
    fn test_sequential_vs_parallel_consistency() {
        let Some(mind1) = wasm_mind_path("simple_mind") else {
            return;
        };
        let Some(mind2) = wasm_mind_path("aggressive_mind") else {
            return;
        };

        let terrain = generate_terrain(64, 64, Some(42), 10);

        let mut game_seq = create_test_game(1, 42);
        game_seq.add_team(TeamId(0), &mind1).unwrap();
        game_seq.add_team(TeamId(1), &mind2).unwrap();
        game_seq.world.elevation = terrain.clone();
        game_seq.step(50, false).unwrap();

        let mut game_par = create_test_game(4, 42);
        game_par.add_team(TeamId(0), &mind1).unwrap();
        game_par.add_team(TeamId(1), &mind2).unwrap();
        game_par.world.elevation = terrain;
        game_par.step(50, false).unwrap();

        assert!(game_seq.world == game_par.world);
        assert_eq!(game_seq.cells, game_par.cells);
        assert_eq!(game_seq.coordinate_map, game_par.coordinate_map);
        assert_eq!(game_seq.inv_coordinate_map, game_par.inv_coordinate_map);
        assert_eq!(game_seq.iteration, game_par.iteration);
        assert_eq!(game_seq.next_cell_id, game_par.next_cell_id);
        assert_eq!(game_seq.match_secret, game_par.match_secret);
    }

    #[test]
    fn test_stateful_wasm_is_pristine_and_worker_invariant() {
        let Some(canary) = wasm_mind_path("isolation_canary") else {
            eprintln!("Skipping: isolation_canary.wasm not found");
            return;
        };
        let mut sequential = create_test_game(1, 91);
        let mut parallel = create_test_game(4, 91);
        sequential.add_team(TeamId(0), &canary).unwrap();
        parallel.add_team(TeamId(0), &canary).unwrap();

        sequential.step(3, false).unwrap();
        parallel.step(3, false).unwrap();

        assert!(sequential.cells.values().all(|cell| !cell.defending));
        assert!(parallel.cells.values().all(|cell| !cell.defending));
        assert!(sequential.world == parallel.world);
        assert_eq!(sequential.cells, parallel.cells);
        assert_eq!(sequential.coordinate_map, parallel.coordinate_map);
        assert_eq!(sequential.inv_coordinate_map, parallel.inv_coordinate_map);
        assert_eq!(sequential.iteration, parallel.iteration);
        assert_eq!(sequential.next_cell_id, parallel.next_cell_id);
        assert_eq!(sequential.match_secret, parallel.match_secret);
    }

    #[test]
    fn test_reference_wasm_path_is_worker_and_hash_invariant() {
        let Some(canary) = wasm_mind_path("isolation_canary") else {
            eprintln!("Skipping: isolation_canary.wasm not found");
            return;
        };
        let mut sequential = create_reference_test_game(1, 117);
        let mut parallel = create_reference_test_game(4, 117);
        sequential.add_team(TeamId(0), &canary).unwrap();
        parallel.add_team(TeamId(0), &canary).unwrap();

        sequential.step(5, false).unwrap();
        parallel.step(5, false).unwrap();

        assert_eq!(
            sequential.reference_state_hash(),
            parallel.reference_state_hash()
        );
        assert!(sequential.reference_state_hash().is_some());
        assert!(sequential.world == parallel.world);
        assert_eq!(sequential.cells, parallel.cells);
        assert_eq!(sequential.coordinate_map, parallel.coordinate_map);
        assert_eq!(sequential.inv_coordinate_map, parallel.inv_coordinate_map);
        assert_eq!(sequential.iteration, parallel.iteration);
        assert!(sequential.cells.values().all(|cell| !cell.defending));
        assert_game_matches_reference_engine(&sequential);
        assert_game_matches_reference_engine(&parallel);
    }

    #[test]
    fn test_reference_native_wasm_is_pristine_memory_explicit_and_worker_invariant() {
        let Some(canary) = wasm_mind_path("isolation_canary") else {
            eprintln!("Skipping: isolation_canary.wasm not found");
            return;
        };
        let mut sequential = create_reference_test_game(1, 118);
        let mut parallel = create_reference_test_game(4, 118);
        sequential.add_team(TeamId(0), &canary).unwrap();
        parallel.add_team(TeamId(0), &canary).unwrap();

        sequential.step(3, false).unwrap();
        parallel.step(3, false).unwrap();

        assert_eq!(
            sequential.reference_state_hash(),
            parallel.reference_state_hash()
        );
        assert_eq!(sequential.cells, parallel.cells);
        assert!(sequential.cells.values().all(|cell| cell.memory[0] == 3));
        assert!(sequential.cells.values().all(|cell| !cell.defending));
    }

    #[test]
    fn test_reference_native_wasm_requires_the_explicit_export() {
        let Some(invalid_mind) = wasm_mind_path("invalid_mind_canary") else {
            eprintln!("Skipping: invalid_mind_canary.wasm not found");
            return;
        };
        let mut game = create_reference_test_game(1, 120);
        let error = game.add_team(TeamId(0), &invalid_mind).unwrap_err();
        assert!(error.contains("reference_mind_function"));
        assert!(game.teams.is_empty());
        assert!(game.cells.is_empty());
        assert!(game.coordinate_map.is_empty());
        assert!(game.reference_state_hash().is_none());
    }

    #[test]
    fn test_maintained_minds_execute_through_the_reference_abi() {
        for (index, name) in [
            "simple_mind",
            "aggressive_mind",
            "defensive_mind",
            "explorer_mind",
        ]
        .into_iter()
        .enumerate()
        {
            let Some(mind) = wasm_mind_path(name) else {
                eprintln!("Skipping: {name}.wasm not found");
                return;
            };
            let mut game = create_reference_test_game(2, 130 + index as u64);
            game.add_team(TeamId(0), &mind).unwrap();

            let tick = game.tick_reference_for_verification(false).unwrap();
            assert_eq!(
                tick.reference_commitments.len(),
                game.cell_config.starting_cells_per_team,
                "{name} did not return one canonical decision per ready cell"
            );
            assert!(game.reference_state_hash().is_some());
        }
    }

    #[test]
    fn test_reference_checkpoint_restore_preserves_explicit_memory() {
        let Some(canary) = wasm_mind_path("isolation_canary") else {
            eprintln!("Skipping: isolation_canary.wasm not found");
            return;
        };
        let mut uninterrupted = create_reference_test_game(2, 122);
        uninterrupted.add_team(TeamId(0), &canary).unwrap();
        uninterrupted.tick(false).unwrap();
        let path = std::env::temp_dir().join(format!(
            "blob-reference-native-checkpoint-{}-122.bin",
            std::process::id()
        ));
        uninterrupted
            .save_state(path.to_str().unwrap(), false)
            .unwrap();
        let state = Game::load_state(path.to_str().unwrap(), false).unwrap();
        std::fs::remove_file(&path).unwrap();
        let mut resumed =
            Game::from_state_with_teams(state, &[(TeamId(0), canary)], Some(122)).unwrap();

        uninterrupted.step(2, false).unwrap();
        resumed.step(2, false).unwrap();
        assert_eq!(
            uninterrupted.reference_state_hash(),
            resumed.reference_state_hash()
        );
        assert_eq!(uninterrupted.cells, resumed.cells);
        assert!(resumed.cells.values().all(|cell| cell.memory[0] == 3));
    }

    #[test]
    fn test_game_checkpoint_rejects_previous_versions() {
        let Some(mind) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };
        let mut game = create_reference_test_game(1, 123);
        game.add_team(TeamId(0), &mind).unwrap();
        let path = std::env::temp_dir().join(format!(
            "blob-forward-only-checkpoint-{}-123.bin",
            std::process::id()
        ));
        game.save_state(path.to_str().unwrap(), false).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        let version_start = GAME_STATE_MAGIC.len();
        bytes[version_start..version_start + 2].copy_from_slice(&10_u16.to_le_bytes());
        fs::write(&path, bytes).unwrap();
        let error = Game::load_state(path.to_str().unwrap(), false)
            .err()
            .expect("v10 checkpoints must not be accepted")
            .to_string();
        fs::remove_file(&path).unwrap();

        assert!(error.contains("unsupported game state version 10; expected 11"));
    }

    #[test]
    fn test_reference_wasm_path_exposes_server_verification_tick() {
        let Some(canary) = wasm_mind_path("isolation_canary") else {
            eprintln!("Skipping: isolation_canary.wasm not found");
            return;
        };
        let mut game = create_reference_test_game(2, 119);
        game.add_team(TeamId(0), &canary).unwrap();

        let tick = game.tick_reference_for_verification(false).unwrap();
        assert!(tick.reference_batch.is_some());
        assert_eq!(
            tick.reference_commitments.len(),
            game.cell_config.starting_cells_per_team
        );
        assert!(
            tick.reference_commitments
                .iter()
                .all(|commitment| commitment.receipt.actor == commitment.actor)
        );
    }

    #[test]
    fn test_on_demand_wasm_allocator_remains_available() {
        let Some(mind) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };
        let mut game = create_reference_test_game(2, 120);
        game.memory_config.use_pooling_allocator = false;
        game.add_team(TeamId(0), &mind).unwrap();

        let tick = game.tick_reference_for_verification(false).unwrap();
        assert_eq!(
            tick.reference_commitments.len(),
            game.cell_config.starting_cells_per_team
        );
    }

    #[test]
    fn test_extism_compat_executor_matches_extism_and_keeps_pristine_guests() {
        let Some(canary) = wasm_mind_path("isolation_canary") else {
            eprintln!("Skipping: isolation_canary.wasm not found");
            return;
        };
        let mut extism = create_reference_test_game(4, 131);
        let mut compat = create_reference_test_game(4, 131);
        compat.memory_config.wasm_executor = crate::config::WasmExecutorKind::ExtismCompat;
        extism.add_team(TeamId(0), &canary).unwrap();
        compat.add_team(TeamId(0), &canary).unwrap();

        for _ in 0..5 {
            let extism_tick = extism.tick_reference_for_verification(false).unwrap();
            let compat_tick = compat.tick_reference_for_verification(false).unwrap();
            assert_eq!(
                extism_tick.reference_commitments,
                compat_tick.reference_commitments
            );
            assert_eq!(extism.reference_state_hash(), compat.reference_state_hash());
        }
        assert!(compat.cells.values().all(|cell| !cell.defending));
        assert!(extism.world == compat.world);
        assert_eq!(extism.cells, compat.cells);
    }

    #[test]
    fn test_all_maintained_minds_match_the_extism_compat_executor() {
        for (index, name) in [
            "simple_mind",
            "aggressive_mind",
            "defensive_mind",
            "explorer_mind",
        ]
        .into_iter()
        .enumerate()
        {
            let Some(mind) = wasm_mind_path(name) else {
                eprintln!("Skipping: {name}.wasm not found");
                return;
            };
            let seed = 140 + index as u64;
            let mut extism = create_reference_test_game(2, seed);
            let mut compat = create_reference_test_game(2, seed);
            compat.memory_config.wasm_executor = crate::config::WasmExecutorKind::ExtismCompat;
            extism.add_team(TeamId(0), &mind).unwrap();
            compat.add_team(TeamId(0), &mind).unwrap();

            for _ in 0..3 {
                let extism_tick = extism.tick_reference_for_verification(false).unwrap();
                let compat_tick = compat.tick_reference_for_verification(false).unwrap();
                assert_eq!(
                    extism_tick.reference_commitments, compat_tick.reference_commitments,
                    "executor mismatch for {name}"
                );
                assert_eq!(
                    extism.reference_state_hash(),
                    compat.reference_state_hash(),
                    "state mismatch for {name}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires separately built AssemblyScript and Go conformance artifacts"]
    fn test_language_pdk_minds_match_both_executors() {
        for (index, (language, name)) in [
            ("assemblyscript_wait_mind", "assemblyscript_wait_mind"),
            ("go_wait_mind", "go_wait_mind"),
        ]
        .into_iter()
        .enumerate()
        {
            let mind = language_mind_path(language, name);
            assert!(
                mind.is_file(),
                "required {language} conformance artifact is unavailable at {}",
                mind.display()
            );
            let seed = 170 + index as u64;
            let mut extism = create_reference_test_game(2, seed);
            let mut compat = create_reference_test_game(2, seed);
            compat.memory_config.wasm_executor = crate::config::WasmExecutorKind::ExtismCompat;
            extism.add_team(TeamId(0), &mind).unwrap();
            compat.add_team(TeamId(0), &mind).unwrap();

            for _ in 0..3 {
                let extism_tick = extism.tick_reference_for_verification(false).unwrap();
                let compat_tick = compat.tick_reference_for_verification(false).unwrap();
                assert!(
                    extism_tick.reference_commitments.iter().all(|commitment| {
                        commitment.request == ActionRequest::Wait
                            && commitment.memory_update == ReferenceMemoryUpdate::Retain
                    }),
                    "{language} canary must return Wait with retained memory"
                );
                assert_eq!(
                    extism_tick.reference_commitments, compat_tick.reference_commitments,
                    "executor mismatch for {language}"
                );
                assert_eq!(
                    extism.reference_state_hash(),
                    compat.reference_state_hash(),
                    "state mismatch for {language}"
                );
            }
        }
    }

    #[test]
    fn test_extism_compat_is_worker_invariant_and_rejects_extra_import_surfaces() {
        let Some(canary) = wasm_mind_path("isolation_canary") else {
            eprintln!("Skipping: isolation_canary.wasm not found");
            return;
        };
        let mut sequential = create_reference_test_game(1, 151);
        let mut parallel = create_reference_test_game(4, 151);
        sequential.memory_config.wasm_executor = crate::config::WasmExecutorKind::ExtismCompat;
        parallel.memory_config.wasm_executor = crate::config::WasmExecutorKind::ExtismCompat;
        sequential.add_team(TeamId(0), &canary).unwrap();
        parallel.add_team(TeamId(0), &canary).unwrap();
        sequential.step(5, false).unwrap();
        parallel.step(5, false).unwrap();

        assert_eq!(
            sequential.reference_state_hash(),
            parallel.reference_state_hash()
        );
        assert_eq!(sequential.cells, parallel.cells);
        assert!(parallel.cells.values().all(|cell| !cell.defending));

        let Some(invalid) = wasm_mind_path("invalid_mind_canary") else {
            return;
        };
        let mut rejected = create_reference_test_game(1, 152);
        rejected.memory_config.wasm_executor = crate::config::WasmExecutorKind::ExtismCompat;
        let error = rejected.add_team(TeamId(0), &invalid).unwrap_err();
        assert!(error.contains("does not export reference_mind_function"));
    }

    #[test]
    #[ignore = "release-only capacity benchmark; run with --ignored --nocapture"]
    fn benchmark_reference_wasm_verification_capacity() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };

        println!("Pristine-instance WASM authoritative verification capacity");
        for executor_kind in [
            crate::config::WasmExecutorKind::Extism,
            crate::config::WasmExecutorKind::ExtismCompat,
        ] {
            for use_pooling_allocator in [true, false] {
                for (board, population, workers) in [
                    (64, 100, 1),
                    (64, 100, 16),
                    (64, 1_000, 1),
                    (64, 1_000, 16),
                    (128, 5_000, 16),
                ] {
                    let mut memory_config = MemoryConfig::default();
                    memory_config.plugin_pool_size = workers;
                    memory_config.use_pooling_allocator = use_pooling_allocator;
                    memory_config.wasm_executor = executor_kind;
                    let cell_config = CellConfig {
                        starting_cells_per_team: population,
                        ..CellConfig::default()
                    };
                    let mut game =
                        Game::new(board, board, 100, cell_config, Some(123), memory_config);
                    game.add_team(TeamId(0), &mind_path).unwrap();
                    let actual_population = game.cells.len();
                    game.initialize_reference_for_verification().unwrap();

                    // Exclude compilation and initial lazy runtime setup.
                    game.tick_reference_for_verification(false).unwrap();
                    let started = std::time::Instant::now();
                    let mut actions = 0_usize;
                    let batches = 20;
                    for _ in 0..batches {
                        actions += game
                            .tick_reference_for_verification(false)
                            .unwrap()
                            .reference_commitments
                            .len();
                    }
                    let elapsed = started.elapsed().as_secs_f64();
                    let allocator = if use_pooling_allocator {
                        "pooling"
                    } else {
                        "on-demand"
                    };
                    let executor = match executor_kind {
                        crate::config::WasmExecutorKind::Extism => "extism",
                        crate::config::WasmExecutorKind::ExtismCompat => "compat",
                    };
                    println!(
                        "{board:>4}x{board:<4} {actual_population:>7} cells {workers:>2} workers \
                         {executor:>6}/{allocator:<9} {actions:>7} actions/{batches} batches  \
                         {elapsed:>7.3}s  {:>9.0} actions/s  {:>7.2} ms/batch",
                        actions as f64 / elapsed,
                        elapsed * 1_000.0 / f64::from(batches),
                    );
                }
            }
        }
    }

    #[test]
    fn test_server_verifier_reexecutes_the_submitted_wasm() {
        use blob_engine::resolution::{
            MatchVerificationLimits, MatchVerificationManifest, MindArtifactBinding,
            MindRuntimeProfile, ReplaySegmentDescriptor,
        };
        use blob_engine::server_verification::ServerMatchVerifier;

        let Some(canary) = wasm_mind_path("isolation_canary") else {
            eprintln!("Skipping: isolation_canary.wasm not found");
            return;
        };
        let stream_config = ReplayStreamConfig {
            max_events_per_segment: 2,
            max_event_bytes_per_segment: usize::MAX / 2,
            segment_limits: ReplaySegmentLimits {
                max_segment_bytes: usize::MAX,
                ..ReplaySegmentLimits::default()
            },
            ..ReplayStreamConfig::default()
        };

        let mut client = create_reference_test_game(2, 121);
        client.add_team(TeamId(0), &canary).unwrap();
        client
            .start_reference_replay_streaming(stream_config)
            .unwrap();
        let initial_runtime_hash = client.reference_state_hash().unwrap();
        client.tick(false).unwrap();
        client.tick(false).unwrap();
        let segment = client.take_reference_replay_segment().unwrap();
        let replay_manifest = client.reference_replay_stream_manifest().unwrap();
        assert_eq!(
            replay_manifest.segment(0),
            Some(&ReplaySegmentDescriptor::from_segment(&segment))
        );

        let artifact_bytes = fs::read(&canary).unwrap();
        let artifacts = vec![MindArtifactBinding {
            slot: 0,
            artifact_hash: blob_engine::resolution::mind_artifact_hash(&artifact_bytes),
        }];
        let verification_manifest = MatchVerificationManifest::from_replay(
            CanonicalHash::from_bytes([0x64; 32]),
            CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            initial_runtime_hash,
            artifacts.clone(),
            &replay_manifest,
            MatchVerificationLimits::default(),
        )
        .unwrap();

        let mut server = create_reference_test_game(2, 121);
        server.add_team(TeamId(0), &canary).unwrap();
        server.initialize_reference_for_verification().unwrap();
        let mut verifier = ServerMatchVerifier::start(
            verification_manifest,
            replay_manifest,
            CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            server.reference_state_hash().unwrap(),
            &artifacts,
            segment,
            server.reference_compiled_ruleset_hash().unwrap(),
            server.reference_canonical_state_hash().unwrap(),
        )
        .unwrap();
        for _ in 0..2 {
            let tick = server.tick_reference_for_verification(false).unwrap();
            verifier
                .verify_tick(&tick, ReplayLimits::default())
                .unwrap();
        }
        verifier.finish().unwrap();
    }

    #[test]
    fn test_reference_game_checkpoint_resumes_with_identical_hashes() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };
        let mut uninterrupted = create_reference_test_game(2, 117);
        uninterrupted.add_team(TeamId(0), &mind_path).unwrap();
        // Force at least two action-duration classes: this cell can only guard
        // while its heavier neighbors remain able to move or consume.
        uninterrupted.cells.get_mut(&CellId(1)).unwrap().energy = 20;
        uninterrupted.start_reference_replay_recording(3).unwrap();
        uninterrupted.tick(false).unwrap();
        assert!(
            uninterrupted
                .reference_engine
                .as_ref()
                .unwrap()
                .reference_simulation()
                .unwrap()
                .cells()
                .values()
                .any(|cell| cell.pending_action.is_some())
        );

        let saved_hash = uninterrupted.reference_state_hash().unwrap();
        let path = std::env::temp_dir().join(format!(
            "blob-reference-checkpoint-{}-117.bin",
            std::process::id()
        ));
        uninterrupted
            .save_state(path.to_str().unwrap(), false)
            .unwrap();
        let mut tampered = Game::load_state(path.to_str().unwrap(), false).unwrap();
        tampered.cells.get_mut(&CellId(0)).unwrap().random_lineage += 1;
        let error =
            Game::from_state_with_teams(tampered, &[(TeamId(0), mind_path.clone())], Some(117))
                .err()
                .expect("tampered private host state must be rejected");
        assert!(error.contains("runtime hash mismatch"));

        let state = Game::load_state(path.to_str().unwrap(), false).unwrap();
        std::fs::remove_file(&path).unwrap();
        let mut resumed =
            Game::from_state_with_teams(state, &[(TeamId(0), mind_path)], Some(117)).unwrap();

        assert_eq!(resumed.reference_state_hash(), Some(saved_hash));
        uninterrupted.step(8, false).unwrap();
        resumed.step(8, false).unwrap();
        assert_eq!(
            uninterrupted.reference_state_hash(),
            resumed.reference_state_hash()
        );
        assert!(uninterrupted.world == resumed.world);
        assert_eq!(uninterrupted.cells, resumed.cells);
        assert_eq!(uninterrupted.coordinate_map, resumed.coordinate_map);
        assert_eq!(uninterrupted.inv_coordinate_map, resumed.inv_coordinate_map);
        assert_eq!(uninterrupted.iteration, resumed.iteration);
        let uninterrupted_replay = uninterrupted.export_reference_replay_bundle().unwrap();
        let resumed_replay = resumed.export_reference_replay_bundle().unwrap();
        assert_eq!(uninterrupted_replay.to_bytes(), resumed_replay.to_bytes());
        assert_eq!(uninterrupted_replay.archive().event_count(), 9);
        assert_eq!(uninterrupted_replay.seek(8).unwrap().events, 6..8);
    }

    #[test]
    fn test_reference_replay_stream_checkpoint_resumes_exactly() {
        let Some(mind_path) = wasm_mind_path("simple_mind") else {
            eprintln!("Skipping: simple_mind.wasm not found");
            return;
        };
        let mut uninterrupted = create_reference_test_game(2, 211);
        uninterrupted.add_team(TeamId(0), &mind_path).unwrap();
        uninterrupted
            .start_reference_replay_streaming(ReplayStreamConfig {
                max_events_per_segment: 2,
                ..ReplayStreamConfig::default()
            })
            .unwrap();
        uninterrupted.tick(false).unwrap();
        assert_eq!(
            uninterrupted
                .reference_replay_stream_manifest()
                .unwrap()
                .event_count(),
            0
        );

        let path = std::env::temp_dir().join(format!(
            "blob-reference-stream-checkpoint-{}-211.bin",
            std::process::id()
        ));
        uninterrupted
            .save_state(path.to_str().unwrap(), false)
            .unwrap();
        let state = Game::load_state(path.to_str().unwrap(), false).unwrap();
        std::fs::remove_file(&path).unwrap();
        let mut resumed =
            Game::from_state_with_teams(state, &[(TeamId(0), mind_path)], Some(211)).unwrap();
        assert_eq!(
            uninterrupted.reference_state_hash(),
            resumed.reference_state_hash()
        );

        uninterrupted.tick(false).unwrap();
        resumed.tick(false).unwrap();
        let uninterrupted_manifest = uninterrupted.reference_replay_stream_manifest().unwrap();
        let resumed_manifest = resumed.reference_replay_stream_manifest().unwrap();
        assert!(uninterrupted_manifest.to_bytes() == resumed_manifest.to_bytes());
        assert_eq!(uninterrupted_manifest.event_count(), 2);

        let uninterrupted_hash = uninterrupted.reference_state_hash();
        let resumed_hash = resumed.reference_state_hash();
        assert!(uninterrupted.tick(false).is_err());
        assert!(resumed.tick(false).is_err());
        assert_eq!(uninterrupted.reference_state_hash(), uninterrupted_hash);
        assert_eq!(resumed.reference_state_hash(), resumed_hash);

        let uninterrupted_first = uninterrupted.take_reference_replay_segment().unwrap();
        let resumed_first = resumed.take_reference_replay_segment().unwrap();
        assert!(uninterrupted_first.to_bytes() == resumed_first.to_bytes());
        uninterrupted_manifest
            .verify_segment(0, &uninterrupted_first)
            .unwrap();

        uninterrupted.tick(false).unwrap();
        resumed.tick(false).unwrap();
        assert!(uninterrupted.flush_reference_replay_stream().unwrap());
        assert!(resumed.flush_reference_replay_stream().unwrap());
        let uninterrupted_tail = uninterrupted.take_reference_replay_segment().unwrap();
        let resumed_tail = resumed.take_reference_replay_segment().unwrap();
        assert!(uninterrupted_tail.to_bytes() == resumed_tail.to_bytes());
        let final_manifest = uninterrupted.reference_replay_stream_manifest().unwrap();
        assert_eq!(final_manifest.event_count(), 3);
        assert_eq!(final_manifest.segment_count(), 2);
        final_manifest
            .verify_segment(1, &uninterrupted_tail)
            .unwrap();
        assert!(
            final_manifest.to_bytes()
                == resumed
                    .reference_replay_stream_manifest()
                    .unwrap()
                    .to_bytes()
        );
    }
}
