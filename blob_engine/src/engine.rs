//! Canonical simulation engine for reference Minds.

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

use blob_interface::cell::Cell;
use blob_interface::randomness::{match_secret_from_seed, PrivateRandom, PrivateRandomDeriver};
use blob_interface::reference_mind::{
    ReferenceMemoryUpdate, ReferenceMind, ReferenceMindDecision, ReferenceMindFactory,
    ReferenceMindInput,
};
use blob_interface::types::{CellId, Coordinate, TeamId};
use blob_interface::world::{EnergySource, World};

use crate::resolution::{
    verify_replay_from_cursor, ActionRequest, BatchReport, BoundaryRule, CellColdState, CellKey,
    CellState as ReferenceCellState, DecisionCommitment, IntegrityMode, NeighborhoodSpec,
    ReferenceCheckpoint, ReferenceObservationBatch, ReferenceRuleset, ReferenceSimulation,
    ReplayArchive, ReplayBatchEvent, ReplayBundle, ReplayBundleLimits, ReplayChainCursor,
    ReplayCommitment, ReplayLimits, ReplayManifest, ReplayManifestLimits, ReplayRecorder,
    ReplaySegment, ReplaySegmentDescriptor, ReplaySegmentLimits, SimTime, SimulationState,
    TileIndex, TileState,
};

/// Configuration for cell behavior
#[derive(Debug, Clone)]
pub struct CellConfig {
    pub min_energy: u32,
    pub initial_energy: u32,
    pub starting_cells_per_team: usize,
    pub max_energy: u32,
    pub min_attack_power: u32,
    pub max_attack_power: u32,
    pub max_energy_for_attack_scaling: u32,
}

impl Default for CellConfig {
    fn default() -> Self {
        CellConfig {
            min_energy: 10,
            initial_energy: 100,
            starting_cells_per_team: 12,
            max_energy: 500,
            min_attack_power: 5,
            max_attack_power: 50,
            max_energy_for_attack_scaling: 200,
        }
    }
}

/// Complete canonical resolver checkpoint and its host continuation state.
#[derive(Debug, Clone)]
pub struct ReferenceRuntimeCheckpoint {
    pub canonical: ReferenceCheckpoint,
    pub iteration: u64,
    pub replay: Option<(u64, ReplayBundle)>,
    pub replay_stream: Option<ReferenceReplayStreamCheckpoint>,
}

#[derive(Debug, Clone)]
struct LiveReplayRecording {
    checkpoint_interval: u64,
    compiled_ruleset_hash: crate::resolution::CanonicalHash,
    initial_state_hash: crate::resolution::CanonicalHash,
    recorder: ReplayRecorder,
    events: Vec<ReplayBatchEvent>,
    checkpoints: Vec<(u64, ReferenceCheckpoint)>,
}

/// Rotation and allocation policy for online replay streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayStreamConfig {
    pub max_events_per_segment: usize,
    pub max_event_bytes_per_segment: usize,
    pub segment_limits: ReplaySegmentLimits,
    pub manifest_limits: ReplayManifestLimits,
}

impl Default for ReplayStreamConfig {
    fn default() -> Self {
        Self {
            max_events_per_segment: 4_096,
            max_event_bytes_per_segment: 8 * 1024 * 1024,
            segment_limits: ReplaySegmentLimits::default(),
            manifest_limits: ReplayManifestLimits::default(),
        }
    }
}

/// Serializable-in-principle stream continuation carried by host checkpoints.
/// Sealed segment bytes may already live outside the engine; the manifest keeps
/// their authenticated descriptors while only the active suffix is retained.
#[derive(Debug, Clone)]
pub struct ReferenceReplayStreamCheckpoint {
    pub config: ReplayStreamConfig,
    pub manifest: ReplayManifest,
    pub active_start_cursor: ReplayChainCursor,
    pub active_checkpoint: ReferenceCheckpoint,
    pub active_events: Vec<ReplayBatchEvent>,
    pub active_event_bytes: usize,
    pub pending_segment: Option<ReplaySegment>,
}

#[derive(Debug, Clone)]
struct LiveReplayStream {
    config: ReplayStreamConfig,
    compiled_ruleset_hash: crate::resolution::CanonicalHash,
    initial_state_hash: crate::resolution::CanonicalHash,
    recorder: ReplayRecorder,
    active_start_cursor: ReplayChainCursor,
    active_checkpoint: ReferenceCheckpoint,
    active_events: Vec<ReplayBatchEvent>,
    active_event_bytes: usize,
    pending_segment: Option<ReplaySegment>,
    descriptors: Vec<ReplaySegmentDescriptor>,
}

/// Result of a single game step, useful for RL training
#[derive(Debug, Clone)]
pub struct StepResult {
    pub iteration: u64,
    pub cells_alive: HashMap<TeamId, usize>,
    pub total_energy: HashMap<TeamId, u32>,
    pub done: bool,
}

/// Events from a single tick, used for reward attribution.
#[derive(Debug, Clone, Default)]
pub struct TickEvents {
    /// (attacker_id, victim_id, victim_team_id)
    pub kills: Vec<(CellId, CellId, TeamId)>,
    /// (parent_id, child_id)
    pub splits: Vec<(CellId, CellId)>,
    /// Canonical report from the authoritative resolver.
    pub reference_batch: Option<BatchReport>,
    /// Exact normalized actions installed at this decision boundary. This is
    /// server-side verification data and is never exposed to another Mind.
    pub reference_commitments: Vec<ReplayCommitment>,
    /// Non-authoritative host-view invalidations. These include passive
    /// physics changes that precede the canonical action delta.
    pub reference_projection_cells: Vec<CellId>,
    pub reference_projection_tiles: Vec<usize>,
}

/// Non-authoritative wall-clock diagnostics for work around one canonical
/// resolver batch. These values never enter hashes, checkpoints, or replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReferenceHostPhaseTimings {
    pub ready_frontier_ns: u64,
    pub observation_index_ns: u64,
    pub observation_projection_ns: u64,
    pub randomness_derivation_ns: u64,
    pub mind_execution_ns: u64,
    pub action_commit_ns: u64,
    pub passive_invalidation_ns: u64,
    pub replay_recording_ns: u64,
    pub host_projection_ns: u64,
    pub total_ns: u64,
    pub parallel_observation: bool,
}

/// Controls whether canonical batches are copied into the renderer-shaped
/// host world after resolution. This is host policy only: both modes execute
/// identical Minds and canonical physics and produce identical verification
/// commitments and hashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceHostMode {
    /// Maintain the legacy `World`/`Cell` view for GUIs and state inspection.
    #[default]
    Projected,
    /// Retain only private dispatch metadata required for future Mind calls.
    /// Intended for trusted servers and throughput-oriented verification.
    MetadataOnly,
}

/// Core simulation engine for exact reference-Mind decisions.
pub struct Engine {
    pub world: World,
    reference_minds: HashMap<TeamId, Vec<Box<dyn ReferenceMind>>>,
    pub cells: HashMap<CellId, Cell>,
    pub coordinate_map: HashMap<Coordinate, CellId>,
    pub inv_coordinate_map: HashMap<CellId, Coordinate>,
    pub iteration: u64,
    pub max_iterations: u64,
    pub cell_config: CellConfig,
    rng: StdRng,
    next_cell_id: usize,
    match_secret: [u8; 32],
    rules: ReferenceRuleset,
    /// Host execution policy only; absent from canonical state and replay.
    reference_parallel_threshold: Option<usize>,
    /// Host Mind/projection policy only; absent from canonical state and replay.
    reference_mind_parallel_threshold: Option<usize>,
    /// Host integrity policy only; absent from canonical state and rulesets.
    reference_integrity_mode: IntegrityMode,
    /// Host projection policy only; absent from canonical state and rulesets.
    reference_host_mode: ReferenceHostMode,
    reference_simulation: Option<ReferenceSimulation>,
    replay_recording: Option<LiveReplayRecording>,
    replay_stream: Option<LiveReplayStream>,
    last_reference_host_timings: ReferenceHostPhaseTimings,
}

impl Engine {
    pub fn new(
        world_width: usize,
        world_height: usize,
        max_iterations: u64,
        cell_config: CellConfig,
        seed: Option<u64>,
        rules: ReferenceRuleset,
    ) -> Self {
        let actual_seed = seed.unwrap_or(0);
        Self::new_with_match_secret(
            world_width,
            world_height,
            max_iterations,
            cell_config,
            Some(actual_seed),
            match_secret_from_seed(actual_seed),
            rules,
        )
    }

    /// Creates an engine with randomness rooted in an independently supplied
    /// secret. Authoritative online hosts should use this constructor and must
    /// not publish the secret while a match is accepting decisions.
    pub fn new_with_match_secret(
        world_width: usize,
        world_height: usize,
        max_iterations: u64,
        cell_config: CellConfig,
        seed: Option<u64>,
        match_secret: [u8; 32],
        mut rules: ReferenceRuleset,
    ) -> Self {
        let actual_seed = seed.unwrap_or(0);
        rules.neighborhood = NeighborhoodSpec::moore_8(BoundaryRule::Wrap);
        rules.child_core_mass = u64::from(cell_config.min_energy);
        rules.minimum_survival_energy = 1;
        Engine {
            world: World::new(world_width, world_height),
            reference_minds: HashMap::new(),
            cells: HashMap::new(),
            coordinate_map: HashMap::new(),
            inv_coordinate_map: HashMap::new(),
            iteration: 0,
            max_iterations,
            cell_config,
            rng: StdRng::seed_from_u64(actual_seed),
            next_cell_id: 0,
            match_secret,
            rules,
            reference_parallel_threshold: Some(2_048),
            reference_mind_parallel_threshold: Some(256),
            reference_integrity_mode: IntegrityMode::Verified,
            reference_host_mode: ReferenceHostMode::Projected,
            reference_simulation: None,
            replay_recording: None,
            replay_stream: None,
            last_reference_host_timings: ReferenceHostPhaseTimings::default(),
        }
    }

    pub fn reference_simulation(&self) -> Option<&ReferenceSimulation> {
        self.reference_simulation.as_ref()
    }

    pub const fn last_reference_host_timings(&self) -> ReferenceHostPhaseTimings {
        self.last_reference_host_timings
    }

    /// Enables ordered parallel intent validation for batches at or above the
    /// threshold. `None` forces the serial semantic oracle.
    pub fn set_reference_parallel_threshold(&mut self, threshold: Option<usize>) {
        self.reference_parallel_threshold = threshold.map(|value| value.max(2));
    }

    /// Enables parallel local projection and native Mind execution when a team
    /// has multiple isolated instances and at least this many ready cells.
    /// This is host policy only and cannot affect canonical ordering.
    pub fn set_reference_mind_parallel_threshold(&mut self, threshold: Option<usize>) {
        self.reference_mind_parallel_threshold = threshold.map(|value| value.max(2));
    }

    /// Selects whether each resolver batch emits authenticated state
    /// boundaries. Replay recording requires verified mode. An on-demand
    /// engine can still compute a current authoritative hash explicitly.
    pub fn set_reference_integrity_mode(&mut self, mode: IntegrityMode) -> Result<(), String> {
        if mode == IntegrityMode::OnDemand
            && (self.replay_recording.is_some() || self.replay_stream.is_some())
        {
            return Err("cannot disable per-batch hashing while replay recording is active".into());
        }
        self.reference_integrity_mode = mode;
        Ok(())
    }

    pub const fn reference_integrity_mode(&self) -> IntegrityMode {
        self.reference_integrity_mode
    }

    /// Selects the host projection policy. It must be selected before the
    /// canonical state is initialized so a running match cannot silently
    /// change the meaning of its public host view.
    pub fn set_reference_host_mode(&mut self, mode: ReferenceHostMode) -> Result<(), String> {
        if self.reference_simulation.is_some() && self.reference_host_mode != mode {
            return Err("reference host mode must be selected before initialization".into());
        }
        self.reference_host_mode = mode;
        Ok(())
    }

    pub const fn reference_host_mode(&self) -> ReferenceHostMode {
        self.reference_host_mode
    }

    /// Materializes the initial canonical state without committing an action.
    pub fn initialize_reference_state(&mut self) -> Result<(), String> {
        self.ensure_reference_simulation()
    }

    pub fn export_reference_checkpoint(&self) -> Result<ReferenceRuntimeCheckpoint, String> {
        let simulation = self
            .reference_simulation
            .as_ref()
            .ok_or("reference state has not been initialized")?;
        let replay = self
            .replay_recording
            .as_ref()
            .map(|recording| {
                self.build_replay_bundle(recording, ReplayBundleLimits::default())
                    .map(|bundle| (recording.checkpoint_interval, bundle))
            })
            .transpose()?;
        let replay_stream = self
            .replay_stream
            .as_ref()
            .map(
                |stream| -> Result<ReferenceReplayStreamCheckpoint, String> {
                    Ok(ReferenceReplayStreamCheckpoint {
                        config: stream.config,
                        manifest: self.build_replay_manifest(stream)?,
                        active_start_cursor: stream.active_start_cursor,
                        active_checkpoint: stream.active_checkpoint.clone(),
                        active_events: stream.active_events.clone(),
                        active_event_bytes: stream.active_event_bytes,
                        pending_segment: stream.pending_segment.clone(),
                    })
                },
            )
            .transpose()?;
        Ok(ReferenceRuntimeCheckpoint {
            canonical: ReferenceCheckpoint::from_simulation(simulation),
            iteration: self.iteration,
            replay,
            replay_stream,
        })
    }

    /// Installs a validated canonical checkpoint over a previously supplied
    /// host projection. Core-derived fields are rebuilt; only private host
    /// metadata such as team, age, inbox, and random sequence is preserved.
    pub fn restore_reference_checkpoint(
        &mut self,
        checkpoint: ReferenceRuntimeCheckpoint,
    ) -> Result<(), String> {
        let ReferenceRuntimeCheckpoint {
            canonical,
            iteration,
            replay,
            replay_stream,
        } = checkpoint;
        if self.reference_integrity_mode != IntegrityMode::Verified
            && (replay.is_some() || replay_stream.is_some())
        {
            return Err("restoring replay state requires verified integrity mode".into());
        }
        if canonical.width() != self.world.dimensions.0
            || canonical.height() != self.world.dimensions.1
        {
            return Err("checkpoint dimensions do not match the host projection".into());
        }
        let tile_count = canonical
            .width()
            .checked_mul(canonical.height())
            .ok_or("checkpoint tile count overflow")?;
        if self.world.pheromone.len() != tile_count {
            return Err("host pheromone projection has the wrong tile count".into());
        }
        let rules = canonical.rules().clone();
        let next_cell_id = usize::try_from(canonical.state().next_cell_key)
            .map_err(|_| "checkpoint next cell key does not fit CellId")?;
        let simulation = canonical
            .into_simulation()
            .map_err(|error| format!("invalid reference checkpoint: {error}"))?;
        self.rules = rules;
        self.reference_simulation = Some(simulation);
        self.iteration = iteration;
        self.next_cell_id = next_cell_id;
        self.replay_recording = None;
        self.replay_stream = None;
        if self.reference_host_mode == ReferenceHostMode::Projected {
            self.refresh_reference_projection()?;
        } else {
            self.retain_reference_metadata_only()?;
        }
        if let Some((checkpoint_interval, bundle)) = replay {
            self.resume_reference_replay_recording(bundle, checkpoint_interval)?;
        }
        if let Some(stream) = replay_stream {
            self.resume_reference_replay_stream(stream)?;
        }
        Ok(())
    }

    /// Starts optional live recording from the current canonical state. The
    /// interval counts completed replay events and must be positive.
    pub fn start_reference_replay_recording(
        &mut self,
        checkpoint_interval: u64,
    ) -> Result<(), String> {
        if self.reference_integrity_mode != IntegrityMode::Verified {
            return Err("reference replay recording requires verified integrity mode".into());
        }
        if checkpoint_interval == 0 {
            return Err("replay checkpoint interval must be positive".into());
        }
        if self.replay_recording.is_some() {
            return Err("reference replay recording is already active".into());
        }
        if self.replay_stream.is_some() {
            return Err("reference replay streaming is already active".into());
        }
        self.ensure_reference_simulation()?;
        let simulation = self
            .reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?;
        let compiled_ruleset_hash = simulation.compiled_ruleset_hash();
        let initial_state_hash = simulation.state_hash();
        self.replay_recording = Some(LiveReplayRecording {
            checkpoint_interval,
            compiled_ruleset_hash,
            initial_state_hash,
            recorder: ReplayRecorder::new(compiled_ruleset_hash, initial_state_hash),
            events: Vec::new(),
            checkpoints: vec![(0, ReferenceCheckpoint::from_simulation(simulation))],
        });
        Ok(())
    }

    pub fn replay_event_count(&self) -> Option<usize> {
        self.replay_recording.as_ref().map_or_else(
            || {
                self.replay_stream
                    .as_ref()
                    .and_then(|stream| usize::try_from(stream.recorder.cursor().next_sequence).ok())
            },
            |recording| Some(recording.events.len()),
        )
    }

    pub fn export_reference_replay_bundle(
        &self,
        limits: ReplayBundleLimits,
    ) -> Result<ReplayBundle, String> {
        let recording = self
            .replay_recording
            .as_ref()
            .ok_or("reference replay recording is not active")?;
        self.build_replay_bundle(recording, limits)
    }

    fn build_replay_bundle(
        &self,
        recording: &LiveReplayRecording,
        limits: ReplayBundleLimits,
    ) -> Result<ReplayBundle, String> {
        let archive = ReplayArchive::from_events(
            &recording.events,
            recording.compiled_ruleset_hash,
            recording.initial_state_hash,
            limits.archive,
        )
        .map_err(|error| format!("could not build replay archive: {error}"))?;
        let mut checkpoints = recording.checkpoints.clone();
        let final_sequence = recording.events.len() as u64;
        if checkpoints.last().map(|entry| entry.0) != Some(final_sequence) {
            let simulation = self
                .reference_simulation
                .as_ref()
                .ok_or("reference simulation was not initialized")?;
            checkpoints.push((
                final_sequence,
                ReferenceCheckpoint::from_simulation(simulation),
            ));
        }
        ReplayBundle::from_parts(archive, checkpoints, limits)
            .map_err(|error| format!("could not build replay bundle: {error}"))
    }

    fn resume_reference_replay_recording(
        &mut self,
        bundle: ReplayBundle,
        checkpoint_interval: u64,
    ) -> Result<(), String> {
        if self.reference_integrity_mode != IntegrityMode::Verified {
            return Err("reference replay recording requires verified integrity mode".into());
        }
        if checkpoint_interval == 0 {
            return Err("replay checkpoint interval must be positive".into());
        }
        let simulation = self
            .reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?;
        if bundle.archive().compiled_ruleset_hash() != simulation.compiled_ruleset_hash() {
            return Err("replay bundle ruleset does not match restored simulation".into());
        }
        let event_count = bundle.archive().event_count();
        let expected_final_state = if event_count == 0 {
            bundle.archive().initial_state_hash()
        } else {
            bundle
                .archive()
                .event(event_count - 1, ReplayLimits::default())
                .map_err(|error| format!("invalid replay event: {error}"))?
                .post_state_hash
        };
        if expected_final_state != simulation.state_hash() {
            return Err("replay bundle does not end at the restored simulation state".into());
        }
        let mut events = Vec::with_capacity(event_count);
        for index in 0..event_count {
            events.push(
                bundle
                    .archive()
                    .event(index, ReplayLimits::default())
                    .map_err(|error| format!("invalid replay event: {error}"))?,
            );
        }
        let compiled_ruleset_hash = bundle.archive().compiled_ruleset_hash();
        let initial_state_hash = bundle.archive().initial_state_hash();
        let recorder = ReplayRecorder::from_events(
            &events,
            compiled_ruleset_hash,
            initial_state_hash,
            ReplayLimits::default(),
        )
        .map_err(|error| format!("could not resume replay recorder: {error}"))?;
        let checkpoints = bundle
            .checkpoints()
            // Export adds a tail checkpoint so a standalone bundle can verify
            // and resume its final state. That tail is an artifact of when the
            // export happened, not part of the configured periodic schedule.
            // Discarding non-boundary checkpoints makes a resumed recording
            // byte-identical to an uninterrupted recording.
            .filter(|(sequence, _)| *sequence == 0 || sequence.is_multiple_of(checkpoint_interval))
            .map(|(sequence, checkpoint)| (sequence, checkpoint.clone()))
            .collect();
        self.replay_recording = Some(LiveReplayRecording {
            checkpoint_interval,
            compiled_ruleset_hash,
            initial_state_hash,
            recorder,
            events,
            checkpoints,
        });
        Ok(())
    }

    fn record_reference_batch(
        &mut self,
        report: &BatchReport,
        commitments: &[ReplayCommitment],
    ) -> Result<(), String> {
        let Some(recording) = self.replay_recording.as_mut() else {
            return Ok(());
        };
        let event = recording
            .recorder
            .record_with_commitments(report, commitments.to_vec())
            .map_err(|error| format!("could not record reference replay event: {error}"))?;
        let applied_events = event
            .sequence
            .checked_add(1)
            .ok_or("replay sequence overflow")?;
        recording.events.push(event);
        if applied_events.is_multiple_of(recording.checkpoint_interval) {
            let simulation = self
                .reference_simulation
                .as_ref()
                .ok_or("reference simulation was not initialized")?;
            recording.checkpoints.push((
                applied_events,
                ReferenceCheckpoint::from_simulation(simulation),
            ));
        }
        Ok(())
    }

    /// Starts bounded online recording. A completed segment must be drained
    /// before another tick; this applies backpressure before any mind decision
    /// or authoritative state mutation can occur.
    pub fn start_reference_replay_streaming(
        &mut self,
        config: ReplayStreamConfig,
    ) -> Result<(), String> {
        if self.reference_integrity_mode != IntegrityMode::Verified {
            return Err("reference replay streaming requires verified integrity mode".into());
        }
        Self::validate_replay_stream_config(config)?;
        if self.replay_recording.is_some() {
            return Err("reference replay bundle recording is already active".into());
        }
        if self.replay_stream.is_some() {
            return Err("reference replay streaming is already active".into());
        }
        self.ensure_reference_simulation()?;
        let simulation = self
            .reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?;
        let compiled_ruleset_hash = simulation.compiled_ruleset_hash();
        let initial_state_hash = simulation.state_hash();
        let recorder = ReplayRecorder::new(compiled_ruleset_hash, initial_state_hash);
        self.replay_stream = Some(LiveReplayStream {
            config,
            compiled_ruleset_hash,
            initial_state_hash,
            active_start_cursor: recorder.cursor(),
            active_checkpoint: ReferenceCheckpoint::from_simulation(simulation),
            recorder,
            active_events: Vec::new(),
            active_event_bytes: 0,
            pending_segment: None,
            descriptors: Vec::new(),
        });
        Ok(())
    }

    pub fn reference_replay_stream_manifest(&self) -> Result<ReplayManifest, String> {
        let stream = self
            .replay_stream
            .as_ref()
            .ok_or("reference replay streaming is not active")?;
        self.build_replay_manifest(stream)
    }

    /// Transfers ownership of the one sealed segment awaiting persistence.
    pub fn take_reference_replay_segment(&mut self) -> Option<ReplaySegment> {
        self.replay_stream
            .as_mut()
            .and_then(|stream| stream.pending_segment.take())
    }

    pub fn reference_replay_stream_buffered_bytes(&self) -> Option<usize> {
        self.replay_stream.as_ref().map(|stream| {
            stream.active_event_bytes
                + stream
                    .pending_segment
                    .as_ref()
                    .map_or(0, |segment| segment.to_bytes().len())
        })
    }

    pub fn reference_replay_stream_needs_drain(&self) -> bool {
        self.replay_stream
            .as_ref()
            .is_some_and(|stream| stream.pending_segment.is_some())
    }

    /// Seals a non-empty tail, for example at match completion. The returned
    /// segment is obtained with [`Self::take_reference_replay_segment`].
    pub fn flush_reference_replay_stream(&mut self) -> Result<bool, String> {
        if self.replay_stream.is_none() {
            return Err("reference replay streaming is not active".into());
        }
        if self
            .replay_stream
            .as_ref()
            .is_some_and(|stream| stream.active_events.is_empty())
        {
            return Ok(false);
        }
        self.seal_reference_replay_segment()?;
        Ok(true)
    }

    fn validate_replay_stream_config(config: ReplayStreamConfig) -> Result<(), String> {
        if config.max_events_per_segment == 0
            || config.max_events_per_segment > config.segment_limits.max_events
        {
            return Err("stream segment event threshold is outside its decoder limits".into());
        }
        if config.max_event_bytes_per_segment == 0
            || config.max_event_bytes_per_segment >= config.segment_limits.max_segment_bytes
        {
            return Err("stream segment byte threshold is outside its decoder limits".into());
        }
        // Rotation occurs after accepting the event that crosses a threshold.
        // Reserve enough space for that full frame, the segment checkpoint,
        // and the canonical offset index so a valid configured frame cannot
        // make sealing fail after the authoritative batch has committed.
        let count_payload = config
            .max_events_per_segment
            .saturating_mul(config.segment_limits.frame.max_frame_bytes);
        let byte_payload = config
            .max_event_bytes_per_segment
            .saturating_add(config.segment_limits.frame.max_frame_bytes);
        let maximum_payload = count_payload.min(byte_payload);
        let maximum_index = config.max_events_per_segment.saturating_mul(16);
        let required_capacity = config
            .segment_limits
            .checkpoint
            .max_checkpoint_bytes
            .saturating_add(maximum_payload)
            .saturating_add(maximum_index)
            .saturating_add(512);
        if required_capacity > config.segment_limits.max_segment_bytes {
            return Err(format!(
                "stream segment limit {} cannot contain its configured worst-case {} bytes",
                config.segment_limits.max_segment_bytes, required_capacity
            ));
        }
        Ok(())
    }

    fn build_replay_manifest(&self, stream: &LiveReplayStream) -> Result<ReplayManifest, String> {
        ReplayManifest::from_segments(
            stream.compiled_ruleset_hash,
            stream.initial_state_hash,
            stream.descriptors.clone(),
            stream.config.manifest_limits,
        )
        .map_err(|error| format!("could not build replay manifest: {error}"))
    }

    fn record_reference_stream_batch(
        &mut self,
        report: &BatchReport,
        commitments: &[ReplayCommitment],
    ) -> Result<(), String> {
        let Some(stream) = self.replay_stream.as_mut() else {
            return Ok(());
        };
        let event = stream
            .recorder
            .record_with_commitments(report, commitments.to_vec())
            .map_err(|error| format!("could not record replay stream event: {error}"))?;
        let event_bytes = event.to_bytes().len();
        stream.active_event_bytes = stream
            .active_event_bytes
            .checked_add(event_bytes)
            .ok_or("replay stream byte count overflow")?;
        stream.active_events.push(event);
        let should_rotate = stream.active_events.len() >= stream.config.max_events_per_segment
            || stream.active_event_bytes >= stream.config.max_event_bytes_per_segment;
        if should_rotate {
            self.seal_reference_replay_segment()?;
        }
        Ok(())
    }

    fn seal_reference_replay_segment(&mut self) -> Result<(), String> {
        let stream = self
            .replay_stream
            .as_ref()
            .ok_or("reference replay streaming is not active")?;
        if stream.pending_segment.is_some() {
            return Err("sealed replay segment must be drained before continuing".into());
        }
        if stream.active_events.is_empty() {
            return Ok(());
        }
        let segment = ReplaySegment::from_events(
            stream.compiled_ruleset_hash,
            stream.active_start_cursor,
            stream.active_checkpoint.clone(),
            &stream.active_events,
            stream.config.segment_limits,
        )
        .map_err(|error| format!("could not seal replay segment: {error}"))?;
        let next_checkpoint = ReferenceCheckpoint::from_simulation(
            self.reference_simulation
                .as_ref()
                .ok_or("reference simulation was not initialized")?,
        );
        let stream = self
            .replay_stream
            .as_mut()
            .ok_or("reference replay streaming is not active")?;
        stream
            .descriptors
            .push(ReplaySegmentDescriptor::from_segment(&segment));
        stream.active_start_cursor = segment.end_cursor();
        stream.active_checkpoint = next_checkpoint;
        stream.active_events.clear();
        stream.active_event_bytes = 0;
        stream.pending_segment = Some(segment);
        Ok(())
    }

    fn resume_reference_replay_stream(
        &mut self,
        checkpoint: ReferenceReplayStreamCheckpoint,
    ) -> Result<(), String> {
        if self.reference_integrity_mode != IntegrityMode::Verified {
            return Err("reference replay streaming requires verified integrity mode".into());
        }
        Self::validate_replay_stream_config(checkpoint.config)?;
        let simulation = self
            .reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?;
        if checkpoint.manifest.compiled_ruleset_hash() != simulation.compiled_ruleset_hash() {
            return Err("replay stream manifest ruleset does not match restored simulation".into());
        }
        if checkpoint.manifest.final_cursor() != checkpoint.active_start_cursor {
            return Err("replay stream active suffix does not continue its manifest".into());
        }
        if checkpoint.active_checkpoint.compiled_ruleset_hash()
            != simulation.compiled_ruleset_hash()
            || checkpoint.active_checkpoint.state_hash()
                != checkpoint.active_start_cursor.previous_state_hash
        {
            return Err("replay stream active checkpoint does not match its cursor".into());
        }
        if let Some(pending) = checkpoint.pending_segment.as_ref() {
            let last = checkpoint
                .manifest
                .segment_count()
                .checked_sub(1)
                .ok_or("pending replay segment is absent from its manifest")?;
            checkpoint
                .manifest
                .verify_segment(last, pending)
                .map_err(|error| format!("invalid pending replay segment: {error}"))?;
        }
        let actual_active_bytes = checkpoint
            .active_events
            .iter()
            .try_fold(0_usize, |total, event| {
                total.checked_add(event.to_bytes().len())
            })
            .ok_or("replay stream byte count overflow")?;
        if actual_active_bytes != checkpoint.active_event_bytes {
            return Err("replay stream active byte count is inconsistent".into());
        }
        let frames: Vec<Vec<u8>> = checkpoint
            .active_events
            .iter()
            .map(ReplayBatchEvent::to_bytes)
            .collect();
        let final_cursor = verify_replay_from_cursor(
            frames.iter().map(Vec::as_slice),
            simulation.compiled_ruleset_hash(),
            checkpoint.active_start_cursor,
            checkpoint.config.segment_limits.frame,
        )
        .map_err(|error| format!("invalid active replay suffix: {error}"))?;
        if final_cursor.previous_state_hash != simulation.state_hash() {
            return Err("active replay suffix does not end at restored simulation state".into());
        }
        let descriptors = checkpoint.manifest.segments().copied().collect();
        self.replay_stream = Some(LiveReplayStream {
            config: checkpoint.config,
            compiled_ruleset_hash: simulation.compiled_ruleset_hash(),
            initial_state_hash: checkpoint.manifest.initial_state_hash(),
            recorder: ReplayRecorder::from_cursor(simulation.compiled_ruleset_hash(), final_cursor),
            active_start_cursor: checkpoint.active_start_cursor,
            active_checkpoint: checkpoint.active_checkpoint,
            active_events: checkpoint.active_events,
            active_event_bytes: checkpoint.active_event_bytes,
            pending_segment: checkpoint.pending_segment,
            descriptors,
        });
        Ok(())
    }

    /// Hashes the reference core plus host state that can affect future mind
    /// decisions. Artifact/ABI hashes are match-envelope inputs and must be
    /// bound separately by the server.
    pub fn authoritative_state_hash(&self) -> Option<crate::resolution::CanonicalHash> {
        let simulation = self.reference_simulation.as_ref()?;
        let mut hasher = Sha256::new();
        let domain = b"blob.authoritative.runtime.v2";
        hasher.update((domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        hasher.update(simulation.state_hash().as_bytes());
        hasher.update(self.iteration.to_le_bytes());
        hasher.update(self.max_iterations.to_le_bytes());
        hasher.update(Sha256::digest(self.match_secret));

        let mut cell_ids: Vec<CellId> = self.cells.keys().copied().collect();
        cell_ids.sort_by_key(|id| id.0);
        hasher.update((cell_ids.len() as u64).to_le_bytes());
        for id in cell_ids {
            let cell = &self.cells[&id];
            hasher.update((id.0 as u64).to_le_bytes());
            hasher.update((cell.team_id.0 as u64).to_le_bytes());
            hasher.update(cell.random_lineage.to_le_bytes());
            hasher.update(cell.decision_sequence.to_le_bytes());
        }
        Some(crate::resolution::CanonicalHash::from_bytes(
            hasher.finalize().into(),
        ))
    }

    /// Returns the deterministic decision frontier. Before the first reference
    /// import all configured cells are ready; afterward only cells whose
    /// pending action has completed are included.
    pub fn ready_cell_ids(&self) -> Vec<CellId> {
        let mut ids: Vec<CellId> = if let Some(simulation) = &self.reference_simulation {
            simulation
                .cells()
                .iter()
                .filter_map(|(key, cell)| {
                    (cell.is_ready_at(simulation.now()))
                        .then(|| usize::try_from(key.0).ok().map(CellId))
                        .flatten()
                })
                .collect()
        } else {
            self.cells.keys().copied().collect()
        };
        ids.sort_by_key(|id| id.0);
        ids
    }

    /// Projects the same anonymous local input used by a Mind. The caller
    /// supplies only the invocation's private random bytes;
    /// engine identity remains outside the observation.
    pub fn reference_mind_input_for(
        &self,
        cell_id: CellId,
        randomness: PrivateRandom,
    ) -> Result<ReferenceMindInput, String> {
        let actor =
            CellKey(u64::try_from(cell_id.0).map_err(|_| "cell ID does not fit reference key")?);
        self.reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?
            .reference_mind_input(actor, randomness)
            .map_err(|error| error.to_string())
    }

    /// Replaces the pre-match host projection before the canonical resolver
    /// has imported it. This is used by external Mind hosts such as
    /// `blob_game`; mutation is rejected once authoritative execution starts.
    pub fn replace_setup_projection(
        &mut self,
        world: World,
        cells: HashMap<CellId, Cell>,
        coordinate_map: HashMap<Coordinate, CellId>,
        inv_coordinate_map: HashMap<CellId, Coordinate>,
    ) -> Result<(), String> {
        if self.reference_simulation.is_some() || self.iteration != 0 {
            return Err("cannot replace setup after authoritative execution starts".into());
        }
        let next_cell_id = cells
            .keys()
            .map(|id| id.0)
            .max()
            .map_or(0, |id| id.saturating_add(1));
        self.world = world;
        self.cells = cells;
        self.coordinate_map = coordinate_map;
        self.inv_coordinate_map = inv_coordinate_map;
        self.next_cell_id = next_cell_id;
        Ok(())
    }

    pub fn next_cell_id(&self) -> usize {
        self.next_cell_id
    }

    /// Adds a team that consumes canonical local observations and returns exact
    /// action/private-memory decisions.
    pub fn add_team_with_minds<R: ReferenceMind + 'static>(
        &mut self,
        team_id: TeamId,
        minds: Vec<R>,
    ) -> Result<(), String> {
        if self.reference_simulation.is_some() || self.iteration != 0 {
            return Err("cannot add a team after authoritative execution starts".into());
        }
        if minds.is_empty() {
            return Err("Must provide at least one reference Mind instance".into());
        }
        if self.reference_minds.contains_key(&team_id) {
            return Err("team already has a Mind pool".into());
        }
        self.reference_minds.insert(
            team_id,
            minds
                .into_iter()
                .map(|mind| Box::new(mind) as Box<dyn ReferenceMind>)
                .collect(),
        );
        self.place_team_cluster(team_id)?;
        Ok(())
    }

    pub fn add_team_with_factory<F: ReferenceMindFactory>(
        &mut self,
        team_id: TeamId,
        factory: &F,
        pool_size: usize,
    ) -> Result<(), String>
    where
        F::M: 'static,
    {
        let mut minds = Vec::with_capacity(pool_size.max(1));
        for index in 0..pool_size.max(1) {
            minds.push(factory.create().map_err(|error| {
                format!("Failed to create reference Mind instance {index}: {error}")
            })?);
        }
        self.add_team_with_minds(team_id, minds)
    }

    fn place_team_cluster(&mut self, team_id: TeamId) -> Result<(), String> {
        let (width, height) = self.world.dimensions;
        let num_cells = self.cell_config.starting_cells_per_team;
        let margin = num_cells.min(width / 4).max(1);
        let center_x = self
            .rng
            .random_range(margin..width.saturating_sub(margin).max(margin + 1));
        let center_y = self
            .rng
            .random_range(margin..height.saturating_sub(margin).max(margin + 1));
        let radius = ((num_cells as f64).sqrt() * 1.5).ceil() as usize;

        let mut placed = 0;
        let mut attempts = 0;
        let max_attempts = num_cells * 20;

        while placed < num_cells && attempts < max_attempts {
            let dx = self.rng.random_range(0..=radius * 2) as isize - radius as isize;
            let dy = self.rng.random_range(0..=radius * 2) as isize - radius as isize;
            let x = (center_x as isize + dx).rem_euclid(width as isize) as usize;
            let y = (center_y as isize + dy).rem_euclid(height as isize) as usize;
            let coord = Coordinate { x, y };

            if !self.coordinate_map.contains_key(&coord) {
                let cell_id = CellId(self.next_cell_id);
                self.next_cell_id += 1;
                let cell = Cell::new(
                    cell_id,
                    team_id,
                    self.cell_config.initial_energy,
                    self.cell_config.min_energy,
                );
                self.cells.insert(cell_id, cell);
                self.coordinate_map.insert(coord, cell_id);
                self.inv_coordinate_map.insert(cell_id, coord);
                placed += 1;
            }
            attempts += 1;
        }
        Ok(())
    }

    /// Core tick: gather exact reference decisions and resolve the next event.
    /// Returns events from the tick for reward attribution.
    pub fn tick(&mut self, verbose: bool) -> Result<TickEvents, String> {
        self.tick_reference(&HashMap::new(), verbose)
    }

    /// Entry point for hosts that already invoked a Mind, such as the
    /// pristine-instance WASM host.
    pub fn tick_reference_with_overrides(
        &mut self,
        reference_overrides: &HashMap<CellId, ReferenceMindDecision>,
        verbose: bool,
    ) -> Result<TickEvents, String> {
        self.tick_reference(reference_overrides, verbose)
    }

    fn gather_reference_decisions(
        &mut self,
        eligible: &[CellId],
        overrides: &HashMap<CellId, ReferenceMindDecision>,
    ) -> Result<(Vec<GatheredReferenceDecision>, u64, u64, u64, u64, bool), String> {
        let mut all_decisions = Vec::with_capacity(eligible.len());
        let mut work_by_team: HashMap<TeamId, Vec<(CellId, u64, u64, bool)>> = HashMap::new();
        let mut cell_ids = eligible.to_vec();
        cell_ids.sort_by_key(|id| id.0);

        for cell_id in cell_ids {
            if let Some(decision) = overrides.get(&cell_id) {
                let host_visible_noop = self.cells.get(&cell_id).is_some_and(|cell| {
                    !cell.defending
                        && match &decision.memory_update {
                            ReferenceMemoryUpdate::Retain => true,
                            ReferenceMemoryUpdate::Replace(bytes) => cell.memory == *bytes,
                        }
                        && decision.signal.is_none()
                        && matches!(
                            &decision.action,
                            blob_interface::reference_mind::ReferenceMindAction::Wait
                        )
                });
                all_decisions.push(GatheredReferenceDecision {
                    cell_id,
                    decision: decision.clone(),
                    host_visible_noop,
                });
                continue;
            }
            let (team_id, lineage, sequence, defending) = {
                let cell = self.cells.get(&cell_id).ok_or("cell input disappeared")?;
                (
                    cell.team_id,
                    cell.random_lineage,
                    cell.decision_sequence,
                    cell.defending,
                )
            };
            if !self.reference_minds.contains_key(&team_id) {
                continue;
            }
            self.cells
                .get_mut(&cell_id)
                .ok_or("cell disappeared before random derivation")?
                .decision_sequence = sequence
                .checked_add(1)
                .ok_or("cell decision sequence overflow")?;
            work_by_team
                .entry(team_id)
                .or_default()
                .push((cell_id, lineage, sequence, defending));
        }

        let pipeline_started = HostPhaseTimer::start();
        let mut observation_projection_ns = 0_u64;
        let mut randomness_derivation_ns = 0_u64;
        let mut parallel_observation = false;
        let match_secret = self.match_secret;
        let random_deriver = PrivateRandomDeriver::new(&match_secret);
        let simulation = self
            .reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?;
        let observation_index_started = HostPhaseTimer::start();
        let observation_batch = simulation.observation_batch();
        let observation_index_ns = observation_index_started.elapsed_ns();
        for (team_id, work) in work_by_team {
            let pool = self
                .reference_minds
                .get_mut(&team_id)
                .ok_or("reference Mind pool disappeared")?;
            let chunk_size = work.len().div_ceil(pool.len());
            let run_parallel = pool.len() > 1
                && self
                    .reference_mind_parallel_threshold
                    .is_some_and(|threshold| work.len() >= threshold);
            parallel_observation |= run_parallel;
            let team_started = HostPhaseTimer::start();
            let chunk_results: Vec<Result<ReferenceDecisionChunk, String>> = if run_parallel {
                pool.par_iter_mut()
                    .zip(work.par_chunks(chunk_size))
                    .map(|(mind, chunk)| {
                        execute_reference_mind_chunk(
                            &observation_batch,
                            mind.as_mut(),
                            chunk,
                            &random_deriver,
                        )
                    })
                    .collect()
            } else {
                pool.iter_mut()
                    .zip(work.chunks(chunk_size))
                    .map(|(mind, chunk)| {
                        execute_reference_mind_chunk(
                            &observation_batch,
                            mind.as_mut(),
                            chunk,
                            &random_deriver,
                        )
                    })
                    .collect()
            };
            let team_wall_ns = team_started.elapsed_ns();
            let mut team_projection_work_ns = 0_u128;
            let mut team_randomness_work_ns = 0_u128;
            let mut team_total_work_ns = 0_u128;
            for result in chunk_results {
                let result = result?;
                team_projection_work_ns += u128::from(result.projection_ns);
                team_randomness_work_ns += u128::from(result.randomness_ns);
                team_total_work_ns += u128::from(result.total_ns);
                all_decisions.extend(result.decisions);
            }
            let projected_wall_ns = if team_total_work_ns == 0 {
                0
            } else {
                u64::try_from(
                    u128::from(team_wall_ns) * team_projection_work_ns / team_total_work_ns,
                )
                .unwrap_or(u64::MAX)
            };
            observation_projection_ns = observation_projection_ns.saturating_add(projected_wall_ns);
            let randomness_wall_ns = if team_total_work_ns == 0 {
                0
            } else {
                u64::try_from(
                    u128::from(team_wall_ns) * team_randomness_work_ns / team_total_work_ns,
                )
                .unwrap_or(u64::MAX)
            };
            randomness_derivation_ns = randomness_derivation_ns.saturating_add(randomness_wall_ns);
        }
        let mind_execution_ns = pipeline_started
            .elapsed_ns()
            .saturating_sub(observation_index_ns)
            .saturating_sub(observation_projection_ns);
        let mind_execution_ns = mind_execution_ns.saturating_sub(randomness_derivation_ns);
        all_decisions.sort_by_key(|decision| decision.cell_id.0);
        Ok((
            all_decisions,
            observation_index_ns,
            observation_projection_ns,
            randomness_derivation_ns,
            mind_execution_ns,
            parallel_observation,
        ))
    }

    fn tick_reference(
        &mut self,
        reference_overrides: &HashMap<CellId, ReferenceMindDecision>,
        verbose: bool,
    ) -> Result<TickEvents, String> {
        let total_started = HostPhaseTimer::start();
        if self
            .replay_stream
            .as_ref()
            .is_some_and(|stream| stream.pending_segment.is_some())
        {
            return Err("sealed replay segment must be drained before continuing".into());
        }
        self.ensure_reference_simulation()?;

        let ready_started = HostPhaseTimer::start();
        let ready_cells: Vec<CellId> = {
            let simulation = self
                .reference_simulation
                .as_ref()
                .ok_or("reference simulation was not initialized")?;
            simulation
                .cells()
                .iter()
                .filter_map(|(key, cell)| {
                    cell.is_ready_at(simulation.now())
                        .then(|| usize::try_from(key.0).ok().map(CellId))
                        .flatten()
                })
                .collect()
        };

        let decision_sequences_before: Vec<(CellId, u64)> = ready_cells
            .iter()
            .filter_map(|cell_id| {
                self.cells
                    .get(cell_id)
                    .map(|cell| (*cell_id, cell.decision_sequence))
            })
            .collect();
        let ready_frontier_ns = ready_started.elapsed_ns();
        let (
            reference_decisions,
            observation_index_ns,
            observation_projection_ns,
            randomness_derivation_ns,
            mind_execution_ns,
            parallel_observation,
        ) = match self.gather_reference_decisions(&ready_cells, reference_overrides) {
            Ok(gathered) => gathered,
            Err(error) => {
                for (cell_id, sequence) in decision_sequences_before {
                    if let Some(cell) = self.cells.get_mut(&cell_id) {
                        cell.decision_sequence = sequence;
                    }
                }
                return Err(error);
            }
        };

        let commit_started = HostPhaseTimer::start();
        let mut decision_commitments = Vec::with_capacity(ready_cells.len());
        let mut projection_skips = if self.reference_host_mode == ReferenceHostMode::Projected {
            Vec::with_capacity(ready_cells.len())
        } else {
            Vec::new()
        };
        let started_at = self
            .reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?
            .now();
        {
            let mut provided = reference_decisions.into_iter().peekable();
            for cell_id in ready_cells {
                let actor = CellKey(
                    u64::try_from(cell_id.0).map_err(|_| "cell ID does not fit reference key")?,
                );
                let (request, signal, memory_update, host_visible_noop) = match provided.peek() {
                    Some(peeked) if peeked.cell_id.0 < cell_id.0 => {
                        return Err("reference decision is outside the ready frontier".into());
                    }
                    Some(peeked) if peeked.cell_id == cell_id => {
                        let provided = provided
                            .next()
                            .ok_or("reference decision iterator disappeared")?;
                        (
                            ActionRequest::from(provided.decision.action),
                            provided.decision.signal,
                            provided.decision.memory_update,
                            provided.host_visible_noop,
                        )
                    }
                    Some(_) | None => {
                        let host_visible_noop =
                            self.cells.get(&cell_id).is_some_and(|cell| !cell.defending);
                        (
                            ActionRequest::Wait,
                            None,
                            ReferenceMemoryUpdate::Retain,
                            host_visible_noop,
                        )
                    }
                };
                if host_visible_noop && self.reference_host_mode == ReferenceHostMode::Projected {
                    projection_skips.push(actor);
                }
                decision_commitments.push(DecisionCommitment {
                    actor,
                    request,
                    signal,
                    memory_update,
                });
            }
            if provided.next().is_some() {
                return Err("reference decision is outside the ready frontier".into());
            }
        }
        let receipts = self
            .reference_simulation
            .as_mut()
            .ok_or("reference simulation was not initialized")?
            .commit_decisions_ordered(&decision_commitments)
            .map_err(|error| format!("reference action commit failed: {error}"))?;
        let commitments = decision_commitments
            .into_iter()
            .zip(receipts)
            .map(|(decision, receipt)| ReplayCommitment {
                actor: decision.actor,
                request: decision.request,
                signal: decision.signal,
                memory_update: decision.memory_update,
                started_at,
                receipt,
            })
            .collect::<Vec<_>>();
        let action_commit_ns = commit_started.elapsed_ns();

        let parallel_threshold = self.reference_parallel_threshold;
        let passive_invalidation_started = HostPhaseTimer::start();
        let (passive_tiles, passive_cells) =
            if self.reference_host_mode == ReferenceHostMode::Projected {
                let simulation = self
                    .reference_simulation
                    .as_ref()
                    .ok_or("reference simulation was not initialized")?;
                let completion_time = simulation
                    .next_event_time()
                    .map_err(|error| format!("reference event scheduling failed: {error}"))?
                    .ok_or("reference simulation has no scheduled event")?;
                simulation.passive_projection_changes_at(completion_time)
            } else {
                (Vec::new(), Vec::new())
            };
        let passive_invalidation_ns = passive_invalidation_started.elapsed_ns();
        let simulation = self
            .reference_simulation
            .as_mut()
            .ok_or("reference simulation was not initialized")?;
        let integrity_mode = self.reference_integrity_mode;
        let report = match parallel_threshold {
            Some(threshold) => {
                simulation.resolve_next_batch_parallel_with_integrity(threshold, integrity_mode)
            }
            None => simulation.resolve_next_batch_with_integrity(integrity_mode),
        }
        .map_err(|error| format!("reference batch resolution failed: {error}"))?;

        if verbose {
            println!(
                "Reference batch {:?}: {} outcomes, state {}",
                report.completed_at,
                report.outcomes.len(),
                report
                    .state_hash()
                    .map_or_else(|| "unverified".into(), |hash| hash.to_string())
            );
        }

        let replay_started = HostPhaseTimer::start();
        self.record_reference_batch(&report, &commitments)?;
        self.record_reference_stream_batch(&report, &commitments)?;
        let replay_recording_ns = replay_started.elapsed_ns();
        let projection_started = HostPhaseTimer::start();
        let mut events = match self.reference_host_mode {
            ReferenceHostMode::Projected => self.sync_reference_projection(
                &report,
                &passive_tiles,
                &passive_cells,
                &projection_skips,
            )?,
            ReferenceHostMode::MetadataOnly => self.sync_reference_metadata(&report)?,
        };
        let host_projection_ns = projection_started.elapsed_ns();
        events.reference_commitments = commitments;
        events.reference_batch = Some(report);
        self.iteration = self.iteration.checked_add(1).ok_or("iteration overflow")?;
        self.last_reference_host_timings = ReferenceHostPhaseTimings {
            ready_frontier_ns,
            observation_index_ns,
            observation_projection_ns,
            randomness_derivation_ns,
            mind_execution_ns,
            action_commit_ns,
            passive_invalidation_ns,
            replay_recording_ns,
            host_projection_ns,
            total_ns: total_started.elapsed_ns(),
            parallel_observation,
        };
        Ok(events)
    }

    fn ensure_reference_simulation(&mut self) -> Result<(), String> {
        if self.reference_simulation.is_some() {
            return Ok(());
        }
        let rules = self.rules.clone();
        let (width, height) = self.world.dimensions;
        let tile_count = width
            .checked_mul(height)
            .ok_or("world tile count overflow")?;
        if self.world.elevation.len() != tile_count || self.world.energy.len() != tile_count {
            return Err("host world arrays do not match configured dimensions".into());
        }

        let mut tiles = Vec::with_capacity(tile_count);
        for index in 0..tile_count {
            let elevation = i16::try_from(self.world.elevation[index])
                .map_err(|_| "world elevation does not fit reference representation")?;
            let mut tile = TileState {
                elevation,
                ..TileState::default()
            };
            match self.world.energy[index] {
                Some(EnergySource::Scattered(amount)) => tile.loose_energy = u64::from(amount),
                Some(EnergySource::Plant {
                    rate,
                    current_energy,
                    max_energy,
                }) => {
                    tile.plant_energy = u64::from(current_energy);
                    tile.plant_capacity = u64::from(max_energy.max(current_energy));
                    tile.plant_growth_rate = u64::from(rate);
                }
                None => {}
            }
            tiles.push(tile);
        }

        for (coordinate, cell_id) in &self.coordinate_map {
            if coordinate.x >= width || coordinate.y >= height {
                return Err("cell coordinate is outside the reference world".into());
            }
            let tile = coordinate.y * width + coordinate.x;
            let key = CellKey(
                u64::try_from(cell_id.0).map_err(|_| "cell ID does not fit reference key")?,
            );
            if tiles[tile].occupant.replace(key).is_some() {
                return Err("multiple cells occupy one imported tile".into());
            }
        }

        let mut cell_ids: Vec<CellId> = self.cells.keys().copied().collect();
        cell_ids.sort_by_key(|id| id.0);
        let mut cells = Vec::with_capacity(cell_ids.len());
        for cell_id in cell_ids {
            let cell = self.cells.get(&cell_id).ok_or("cell disappeared")?;
            let coordinate = self
                .inv_coordinate_map
                .get(&cell_id)
                .ok_or("cell has no imported coordinate")?;
            let key = CellKey(
                u64::try_from(cell_id.0).map_err(|_| "cell ID does not fit reference key")?,
            );
            cells.push((
                key,
                ReferenceCellState {
                    position: TileIndex(coordinate.y * width + coordinate.x),
                    core_mass: u64::from(cell.min_energy),
                    assimilated_energy: u64::from(cell.energy),
                    gut_energy: 0,
                    digestion_remainder: 0,
                    metabolism_remainder: 0,
                    carried_material_mass: u64::from(cell.loaded),
                    marker: cell.marker,
                    guarded: cell.defending,
                    ready_at: SimTime(0),
                    pending_action: None,
                    last_outcome: None,
                    cold: Arc::new(CellColdState {
                        private_memory: cell.memory.clone().into(),
                    }),
                },
            ));
        }
        let next_cell_key =
            u64::try_from(self.next_cell_id).map_err(|_| "next cell ID does not fit u64")?;
        let state = SimulationState {
            now: SimTime(0),
            tiles,
            cells,
            next_cell_key,
        };
        self.reference_simulation = Some(
            ReferenceSimulation::from_canonical_state(width, height, rules, state)
                .map_err(|error| format!("reference import failed: {error}"))?,
        );
        if self.reference_host_mode == ReferenceHostMode::MetadataOnly {
            self.retain_reference_metadata_only()?;
        }
        Ok(())
    }

    /// Drops renderer indexes while retaining the private per-cell identity
    /// needed to dispatch future Mind calls. Canonical cell/world state stays
    /// solely in `ReferenceSimulation` in this mode.
    fn retain_reference_metadata_only(&mut self) -> Result<(), String> {
        let simulation = self
            .reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?;
        for (key, _) in simulation.cells().iter() {
            let id = CellId(usize::try_from(key.0).map_err(|_| "cell key does not fit CellId")?);
            if !self.cells.contains_key(&id) {
                return Err("canonical cell is missing private host metadata".into());
            }
        }
        self.cells.retain(|id, _| {
            u64::try_from(id.0)
                .ok()
                .is_some_and(|key| simulation.cell(CellKey(key)).is_some())
        });
        for cell in self.cells.values_mut() {
            cell.energy = 0;
            cell.min_energy = 0;
            cell.marker = 0;
            cell.loaded = false;
            cell.age = 0;
            cell.memory = Vec::new();
            cell.message_queue = Vec::new();
            cell.defending = false;
        }
        self.coordinate_map.clear();
        self.inv_coordinate_map.clear();
        Ok(())
    }

    fn refresh_reference_projection(&mut self) -> Result<(), String> {
        let simulation = self
            .reference_simulation
            .as_ref()
            .ok_or("reference simulation was not initialized")?;
        let state = simulation.canonical_state();
        let tile_count = state.tiles.len();
        if self.world.elevation.len() != tile_count
            || self.world.energy.len() != tile_count
            || self.world.pheromone.len() != tile_count
        {
            return Err("host world arrays do not match checkpoint dimensions".into());
        }

        let mut projected = HashMap::with_capacity(state.cells.len());
        self.coordinate_map.clear();
        self.inv_coordinate_map.clear();
        for (key, reference_cell) in &state.cells {
            let id = CellId(usize::try_from(key.0).map_err(|_| "cell key does not fit CellId")?);
            let mut cell = self
                .cells
                .get(&id)
                .cloned()
                .ok_or("checkpoint cell is missing private host metadata")?;
            cell.id = id;
            cell.energy = u32::try_from(reference_cell.assimilated_energy)
                .map_err(|_| "canonical energy does not fit host projection")?;
            cell.min_energy = u32::try_from(reference_cell.core_mass)
                .map_err(|_| "canonical core mass does not fit host projection")?;
            cell.marker = reference_cell.marker;
            cell.loaded = reference_cell.carried_material_mass > 0;
            cell.memory = reference_cell.private_memory.to_vec();
            cell.defending = reference_cell.guarded;
            projected.insert(id, cell);

            let (x, y) = simulation
                .neighborhood()
                .coordinate(reference_cell.position)
                .ok_or("reference cell position is invalid")?;
            let coordinate = Coordinate { x, y };
            self.coordinate_map.insert(coordinate, id);
            self.inv_coordinate_map.insert(id, coordinate);
        }
        self.cells = projected;

        for (index, tile) in state.tiles.iter().enumerate() {
            self.world.elevation[index] = i32::from(tile.elevation);
            let visible = tile
                .plant_energy
                .checked_add(tile.loose_energy)
                .ok_or("visible tile energy overflow")?;
            let visible = u32::try_from(visible)
                .map_err(|_| "canonical tile energy does not fit host projection")?;
            self.world.energy[index] = if tile.plant_growth_rate > 0 || tile.plant_capacity > 0 {
                Some(EnergySource::Plant {
                    rate: u32::try_from(tile.plant_growth_rate)
                        .map_err(|_| "canonical plant rate does not fit host projection")?,
                    current_energy: visible,
                    max_energy: u32::try_from(tile.plant_capacity)
                        .map_err(|_| "canonical plant capacity does not fit host projection")?
                        .max(visible),
                })
            } else if visible > 0 {
                Some(EnergySource::Scattered(visible))
            } else {
                None
            };
            self.world.pheromone[index] = project_signal(tile.signal_energy);
        }
        Ok(())
    }

    fn sync_reference_projection(
        &mut self,
        report: &BatchReport,
        passive_tiles: &[TileIndex],
        passive_cells: &[CellKey],
        projection_skips: &[CellKey],
    ) -> Result<TickEvents, String> {
        let (width, height) = self.world.dimensions;
        let tile_count = width
            .checked_mul(height)
            .ok_or("world tile count overflow")?;
        let child_teams: HashMap<CellKey, TeamId> = report
            .births
            .iter()
            .map(|(parent, child)| {
                let parent_id = CellId(
                    usize::try_from(parent.0).map_err(|_| "parent key does not fit CellId")?,
                );
                let team = self
                    .cells
                    .get(&parent_id)
                    .ok_or("new reference cell has no parent team")?
                    .team_id;
                Ok((*child, team))
            })
            .collect::<Result<_, String>>()?;
        let death_teams: HashMap<CellKey, TeamId> = report
            .deaths
            .iter()
            .filter_map(|key| {
                let id = usize::try_from(key.0).ok().map(CellId)?;
                self.cells.get(&id).map(|cell| (*key, cell.team_id))
            })
            .collect();
        let death_positions: HashMap<CellKey, TileIndex> = report
            .deaths
            .iter()
            .filter_map(|key| {
                let id = usize::try_from(key.0).ok().map(CellId)?;
                let coordinate = self.inv_coordinate_map.get(&id)?;
                Some((*key, TileIndex(coordinate.y * width + coordinate.x)))
            })
            .collect();
        let (passive_cell_states, passive_tile_states) = {
            let simulation = self
                .reference_simulation
                .as_ref()
                .ok_or("reference simulation was not initialized")?;
            let cells = passive_cells
                .iter()
                .filter(|key| {
                    report
                        .delta
                        .cells
                        .binary_search_by_key(*key, |delta| delta.cell)
                        .is_err()
                })
                .map(|key| (*key, simulation.cell(*key).cloned()))
                .collect::<Vec<_>>();
            let tiles = passive_tiles
                .iter()
                .filter(|tile| {
                    report
                        .delta
                        .tiles
                        .binary_search_by_key(*tile, |delta| delta.tile)
                        .is_err()
                })
                .map(|tile| {
                    let state = simulation
                        .tile_state(*tile)
                        .cloned()
                        .ok_or("passive tile position is invalid")?;
                    Ok((*tile, state))
                })
                .collect::<Result<Vec<_>, String>>()?;
            (cells, tiles)
        };

        // Host age advances once per resolved event, including for cells that
        // did not otherwise change in the canonical state.
        for cell in self.cells.values_mut() {
            cell.age = cell.age.saturating_add(1);
        }

        // Vacate every changed origin before filling destinations. This keeps
        // swaps and chains independent of cell-delta iteration order.
        for delta in &report.delta.cells {
            let Some(before) = delta.before.as_ref() else {
                continue;
            };
            if delta
                .after
                .as_ref()
                .is_some_and(|after| before.position == after.position)
            {
                continue;
            }
            let id =
                CellId(usize::try_from(delta.cell.0).map_err(|_| "cell key does not fit CellId")?);
            let old_coordinate = self.inv_coordinate_map.get(&id).copied();
            let new_coordinate = delta.after.as_ref().and_then(|cell| {
                (cell.position.0 < tile_count).then_some(Coordinate {
                    x: cell.position.0 % width,
                    y: cell.position.0 / width,
                })
            });
            if old_coordinate != new_coordinate {
                if let Some(coordinate) = self.inv_coordinate_map.remove(&id) {
                    if self.coordinate_map.get(&coordinate) == Some(&id) {
                        self.coordinate_map.remove(&coordinate);
                    }
                }
            }
            if delta.after.is_none() {
                self.cells.remove(&id);
            }
        }

        for delta in &report.delta.cells {
            let Some(reference_cell) = delta.after.as_ref() else {
                continue;
            };
            if projection_skips.binary_search(&delta.cell).is_ok()
                && passive_cells.binary_search(&delta.cell).is_err()
                && delta
                    .before
                    .as_ref()
                    .is_some_and(|before| reference_projection_fields_equal(before, reference_cell))
            {
                continue;
            }
            let id =
                CellId(usize::try_from(delta.cell.0).map_err(|_| "cell key does not fit CellId")?);
            let energy = u32::try_from(reference_cell.assimilated_energy)
                .map_err(|_| "canonical energy does not fit host projection")?;
            let min_energy = u32::try_from(reference_cell.core_mass)
                .map_err(|_| "canonical core mass does not fit host projection")?;
            let is_birth = delta.before.is_none();
            if is_birth {
                let team_id = child_teams
                    .get(&delta.cell)
                    .copied()
                    .ok_or("new reference cell has no parent team")?;
                self.cells
                    .insert(id, Cell::new(id, team_id, energy, min_energy));
            }
            let cell = self
                .cells
                .get_mut(&id)
                .ok_or("changed reference cell is missing host metadata")?;
            project_reference_cell(cell, reference_cell, true)?;

            let tile = reference_cell.position.0;
            if tile >= tile_count {
                return Err("reference cell position is invalid".into());
            }
            let coordinate = Coordinate {
                x: tile % width,
                y: tile / width,
            };
            if self.inv_coordinate_map.get(&id) != Some(&coordinate) {
                if self.coordinate_map.insert(coordinate, id).is_some() {
                    return Err("multiple projected cells occupy one tile".into());
                }
                self.inv_coordinate_map.insert(id, coordinate);
            }
        }
        for (key, reference_cell) in &passive_cell_states {
            let id = CellId(usize::try_from(key.0).map_err(|_| "cell key does not fit CellId")?);
            let Some(reference_cell) = reference_cell else {
                if let Some(coordinate) = self.inv_coordinate_map.remove(&id) {
                    self.coordinate_map.remove(&coordinate);
                }
                self.cells.remove(&id);
                continue;
            };
            let cell = self
                .cells
                .get_mut(&id)
                .ok_or("passively changed cell is missing host metadata")?;
            project_reference_cell(cell, reference_cell, false)?;
        }
        self.next_cell_id = usize::try_from(report.delta.after_next_cell_key)
            .map_err(|_| "next key does not fit CellId")?;

        for delta in &report.delta.tiles {
            project_reference_tile(&mut self.world, delta.tile.0, &delta.after)?;
        }
        for (tile, state) in &passive_tile_states {
            project_reference_tile(&mut self.world, tile.0, state)?;
        }

        let mut events = TickEvents::default();
        events.reference_projection_cells = merge_sorted_unique(
            report.delta.cells.iter().map(|delta| delta.cell),
            passive_cells.iter().copied(),
        )
        .into_iter()
        .map(|key| {
            usize::try_from(key.0)
                .map(CellId)
                .map_err(|_| "cell key does not fit CellId".to_string())
        })
        .collect::<Result<_, _>>()?;
        events.reference_projection_tiles = merge_sorted_unique(
            report.delta.tiles.iter().map(|delta| delta.tile.0),
            passive_tiles.iter().map(|tile| tile.0),
        );
        for (parent, child) in &report.births {
            events.splits.push((
                CellId(usize::try_from(parent.0).map_err(|_| "parent key overflow")?),
                CellId(usize::try_from(child.0).map_err(|_| "child key overflow")?),
            ));
        }
        for death in &report.deaths {
            let victim_team = death_teams.get(death).copied().unwrap_or(TeamId(0));
            if let Some(death_position) = death_positions.get(death) {
                for outcome in &report.outcomes {
                    if outcome.action == crate::resolution::ActionKind::Attack
                        && outcome.status == crate::resolution::OutcomeStatus::Success
                        && outcome.target.as_ref() == Some(death_position)
                    {
                        events.kills.push((
                            CellId(
                                usize::try_from(outcome.actor.0)
                                    .map_err(|_| "attacker key overflow")?,
                            ),
                            CellId(usize::try_from(death.0).map_err(|_| "victim key overflow")?),
                            victim_team,
                        ));
                    }
                }
            }
        }
        Ok(events)
    }

    /// Advances only the private host metadata that is not represented in the
    /// canonical simulation. No cell/world fields used by renderers are copied.
    fn sync_reference_metadata(&mut self, report: &BatchReport) -> Result<TickEvents, String> {
        let child_teams: HashMap<CellKey, TeamId> = report
            .births
            .iter()
            .map(|(parent, child)| {
                let parent_id = CellId(
                    usize::try_from(parent.0).map_err(|_| "parent key does not fit CellId")?,
                );
                let team = self
                    .cells
                    .get(&parent_id)
                    .ok_or("new reference cell has no parent team")?
                    .team_id;
                Ok((*child, team))
            })
            .collect::<Result<_, String>>()?;
        let death_teams: HashMap<CellKey, TeamId> = report
            .deaths
            .iter()
            .filter_map(|key| {
                let id = usize::try_from(key.0).ok().map(CellId)?;
                self.cells.get(&id).map(|cell| (*key, cell.team_id))
            })
            .collect();
        let death_positions: HashMap<CellKey, TileIndex> = report
            .delta
            .cells
            .iter()
            .filter_map(|delta| {
                report.deaths.binary_search(&delta.cell).ok().and_then(|_| {
                    delta
                        .before
                        .as_ref()
                        .map(|cell| (delta.cell, cell.position))
                })
            })
            .collect();

        for death in &report.deaths {
            let id =
                CellId(usize::try_from(death.0).map_err(|_| "dead cell key does not fit CellId")?);
            self.cells.remove(&id);
        }
        for (_, child) in &report.births {
            let id = CellId(usize::try_from(child.0).map_err(|_| "child key does not fit CellId")?);
            let team_id = child_teams
                .get(child)
                .copied()
                .ok_or("new reference cell has no parent team")?;
            self.cells.insert(
                id,
                Cell {
                    id,
                    team_id,
                    energy: 0,
                    min_energy: 0,
                    marker: 0,
                    loaded: false,
                    age: 0,
                    memory: Vec::new(),
                    message_queue: Vec::new(),
                    defending: false,
                    random_lineage: id.0 as u64,
                    decision_sequence: 0,
                },
            );
        }
        self.next_cell_id = usize::try_from(report.delta.after_next_cell_key)
            .map_err(|_| "next key does not fit CellId")?;

        let mut events = TickEvents::default();
        for (parent, child) in &report.births {
            events.splits.push((
                CellId(usize::try_from(parent.0).map_err(|_| "parent key overflow")?),
                CellId(usize::try_from(child.0).map_err(|_| "child key overflow")?),
            ));
        }
        for death in &report.deaths {
            let victim_team = death_teams.get(death).copied().unwrap_or(TeamId(0));
            if let Some(death_position) = death_positions.get(death) {
                for outcome in &report.outcomes {
                    if outcome.action == crate::resolution::ActionKind::Attack
                        && outcome.status == crate::resolution::OutcomeStatus::Success
                        && outcome.target.as_ref() == Some(death_position)
                    {
                        events.kills.push((
                            CellId(
                                usize::try_from(outcome.actor.0)
                                    .map_err(|_| "attacker key overflow")?,
                            ),
                            CellId(usize::try_from(death.0).map_err(|_| "victim key overflow")?),
                            victim_team,
                        ));
                    }
                }
            }
        }
        Ok(events)
    }

    /// Run N ticks, return the number actually executed.
    pub fn step(&mut self, n: u64, verbose: bool) -> Result<u64, String> {
        let mut executed = 0;
        for _ in 0..n {
            if self.iteration >= self.max_iterations {
                break;
            }
            self.tick(verbose)?;
            executed += 1;
        }
        Ok(executed)
    }

    /// Get a StepResult summarizing the current state.
    pub fn get_step_result(&self) -> StepResult {
        let mut cells_alive: HashMap<TeamId, usize> = HashMap::new();
        let mut total_energy: HashMap<TeamId, u32> = HashMap::new();
        for cell in self.cells.values() {
            *cells_alive.entry(cell.team_id).or_insert(0) += 1;
            *total_energy.entry(cell.team_id).or_insert(0) += cell.energy;
        }
        StepResult {
            iteration: self.iteration,
            cells_alive,
            total_energy,
            done: self.iteration >= self.max_iterations,
        }
    }
}

struct GatheredReferenceDecision {
    cell_id: CellId,
    decision: ReferenceMindDecision,
    host_visible_noop: bool,
}

struct ReferenceDecisionChunk {
    decisions: Vec<GatheredReferenceDecision>,
    projection_ns: u64,
    randomness_ns: u64,
    total_ns: u64,
}

fn execute_reference_mind_chunk(
    observations: &ReferenceObservationBatch<'_>,
    mind: &mut dyn ReferenceMind,
    work: &[(CellId, u64, u64, bool)],
    random_deriver: &PrivateRandomDeriver,
) -> Result<ReferenceDecisionChunk, String> {
    let total_started = HostPhaseTimer::start();
    let mut projection_ns = 0_u64;
    let mut randomness_ns = 0_u64;
    let mut decisions = Vec::with_capacity(work.len());
    let mut scratch_input = None;
    let mut randomness_samples = 0_usize;
    for (index, (cell_id, lineage, sequence, host_was_defending)) in work.iter().enumerate() {
        let actor =
            CellKey(u64::try_from(cell_id.0).map_err(|_| "cell ID does not fit reference key")?);
        let randomness = if index < 64 {
            let randomness_started = HostPhaseTimer::start();
            let randomness = random_deriver.derive(*lineage, *sequence);
            randomness_ns = randomness_ns.saturating_add(randomness_started.elapsed_ns());
            randomness_samples += 1;
            randomness
        } else {
            random_deriver.derive(*lineage, *sequence)
        };
        let projection_started = HostPhaseTimer::start();
        if let Some(input) = scratch_input.as_mut() {
            observations
                .reference_mind_input_into(actor, randomness, input)
                .map_err(|error| format!("reference Mind projection failed: {error}"))?;
        } else {
            scratch_input = Some(
                observations
                    .reference_mind_input(actor, randomness)
                    .map_err(|error| format!("reference Mind projection failed: {error}"))?,
            );
        }
        projection_ns = projection_ns.saturating_add(projection_started.elapsed_ns());
        mind.reset()
            .map_err(|error| format!("reference Mind reset error: {error}"))?;
        let input = scratch_input
            .as_ref()
            .ok_or("reference Mind scratch input disappeared")?;
        let decision = mind.decide(input);
        let host_visible_noop = !*host_was_defending
            && match &decision.memory_update {
                ReferenceMemoryUpdate::Retain => true,
                ReferenceMemoryUpdate::Replace(bytes) => *bytes == input.private_memory,
            }
            && decision.signal.is_none()
            && matches!(
                &decision.action,
                blob_interface::reference_mind::ReferenceMindAction::Wait
            );
        decisions.push(GatheredReferenceDecision {
            cell_id: *cell_id,
            decision,
            host_visible_noop,
        });
    }
    if randomness_samples > 0 && randomness_samples < work.len() {
        randomness_ns = u64::try_from(
            u128::from(randomness_ns) * work.len() as u128 / randomness_samples as u128,
        )
        .unwrap_or(u64::MAX);
    }
    Ok(ReferenceDecisionChunk {
        decisions,
        projection_ns,
        randomness_ns,
        total_ns: total_started.elapsed_ns(),
    })
}

#[cfg(not(target_arch = "wasm32"))]
struct HostPhaseTimer(std::time::Instant);

#[cfg(not(target_arch = "wasm32"))]
impl HostPhaseTimer {
    fn start() -> Self {
        Self(std::time::Instant::now())
    }

    fn elapsed_ns(&self) -> u64 {
        u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}

#[cfg(target_arch = "wasm32")]
struct HostPhaseTimer;

#[cfg(target_arch = "wasm32")]
impl HostPhaseTimer {
    const fn start() -> Self {
        Self
    }

    const fn elapsed_ns(&self) -> u64 {
        0
    }
}

fn merge_sorted_unique<T, I, J>(left: I, right: J) -> Vec<T>
where
    T: Copy + Ord,
    I: IntoIterator<Item = T>,
    J: IntoIterator<Item = T>,
{
    let mut left = left.into_iter().peekable();
    let mut right = right.into_iter().peekable();
    let mut merged = Vec::with_capacity(left.size_hint().0.saturating_add(right.size_hint().0));
    while left.peek().is_some() || right.peek().is_some() {
        match (left.peek(), right.peek()) {
            (Some(left_value), Some(right_value)) if left_value < right_value => {
                merged.push(left.next().expect("peeked projection key disappeared"));
            }
            (Some(left_value), Some(right_value)) if right_value < left_value => {
                merged.push(right.next().expect("peeked projection key disappeared"));
            }
            (Some(_), Some(_)) => {
                merged.push(left.next().expect("peeked projection key disappeared"));
                right.next();
            }
            (Some(_), None) => {
                merged.push(left.next().expect("peeked projection key disappeared"));
            }
            (None, Some(_)) => {
                merged.push(right.next().expect("peeked projection key disappeared"));
            }
            (None, None) => break,
        }
    }
    merged
}

fn project_reference_cell(
    cell: &mut Cell,
    reference: &ReferenceCellState,
    project_memory: bool,
) -> Result<(), String> {
    cell.energy = u32::try_from(reference.assimilated_energy)
        .map_err(|_| "canonical energy does not fit host projection")?;
    cell.min_energy = u32::try_from(reference.core_mass)
        .map_err(|_| "canonical core mass does not fit host projection")?;
    cell.marker = reference.marker;
    cell.loaded = reference.carried_material_mass > 0;
    if project_memory && cell.memory.as_slice() != reference.private_memory.as_ref() {
        cell.memory.clear();
        cell.memory.extend_from_slice(&reference.private_memory);
    }
    cell.defending = reference.guarded;
    Ok(())
}

fn reference_projection_fields_equal(
    left: &ReferenceCellState,
    right: &ReferenceCellState,
) -> bool {
    left.position == right.position
        && left.core_mass == right.core_mass
        && left.assimilated_energy == right.assimilated_energy
        && left.marker == right.marker
        && (left.carried_material_mass > 0) == (right.carried_material_mass > 0)
        && left.guarded == right.guarded
}

fn project_signal(
    signal: [u64; blob_interface::reference_mind::REFERENCE_SIGNAL_CHANNELS],
) -> Option<u32> {
    let channels = signal.map(|energy| energy.min(u64::from(u8::MAX)) as u8);
    (channels != [0; 4]).then_some(u32::from_le_bytes(channels))
}

fn project_reference_tile(world: &mut World, index: usize, tile: &TileState) -> Result<(), String> {
    if index >= world.elevation.len()
        || index >= world.energy.len()
        || index >= world.pheromone.len()
    {
        return Err("reference tile position is invalid".into());
    }
    world.elevation[index] = i32::from(tile.elevation);
    let visible = tile
        .plant_energy
        .checked_add(tile.loose_energy)
        .ok_or("visible tile energy overflow")?;
    let visible =
        u32::try_from(visible).map_err(|_| "canonical tile energy does not fit host projection")?;
    world.energy[index] = if tile.plant_growth_rate > 0 || tile.plant_capacity > 0 {
        Some(EnergySource::Plant {
            rate: u32::try_from(tile.plant_growth_rate)
                .map_err(|_| "canonical plant rate does not fit host projection")?,
            current_energy: visible,
            max_energy: u32::try_from(tile.plant_capacity)
                .map_err(|_| "canonical plant capacity does not fit host projection")?
                .max(visible),
        })
    } else if visible > 0 {
        Some(EnergySource::Scattered(visible))
    } else {
        None
    };
    world.pheromone[index] = project_signal(tile.signal_energy);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_minds::RandomMind;
    use crate::resolution::{DurationRule, OutcomeStatus, ReferenceReplayDriver};

    fn create_test_engine(seed: u64) -> Engine {
        Engine::new(
            64,
            64,
            1000,
            CellConfig::default(),
            Some(seed),
            ReferenceRuleset::default(),
        )
    }

    struct MemoryReferenceMind;

    impl ReferenceMind for MemoryReferenceMind {
        fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
            let mut memory = input.private_memory.clone();
            memory[0] = memory[0].wrapping_add(1);
            memory[1] = input.randomness.as_bytes()[0];
            ReferenceMindDecision {
                action: blob_interface::reference_mind::ReferenceMindAction::Wait,
                signal: None,
                memory_update: ReferenceMemoryUpdate::Replace(memory),
            }
        }

        fn reset(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    struct FailingReferenceMind;

    impl ReferenceMind for FailingReferenceMind {
        fn decide(&mut self, _input: &ReferenceMindInput) -> ReferenceMindDecision {
            unreachable!("a failed reset must prevent decision execution")
        }

        fn reset(&mut self) -> Result<(), String> {
            Err("intentional reset failure".into())
        }
    }

    #[test]
    fn test_engine_creation() {
        let engine = create_test_engine(42);
        assert_eq!(engine.iteration, 0);
        assert!(engine.cells.is_empty());
        assert!(engine.reference_minds.is_empty());
    }

    #[test]
    fn metadata_only_host_is_canonically_and_authoritatively_equivalent() {
        let mut projected = create_test_engine(142);
        let mut metadata_only = create_test_engine(142);
        metadata_only
            .set_reference_host_mode(ReferenceHostMode::MetadataOnly)
            .unwrap();
        projected
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        metadata_only
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();

        for _ in 0..10 {
            let projected_events = projected.tick(false).unwrap();
            let metadata_events = metadata_only.tick(false).unwrap();
            assert_eq!(
                projected_events.reference_commitments,
                metadata_events.reference_commitments
            );
            assert_eq!(
                projected_events.reference_batch,
                metadata_events.reference_batch
            );
            assert_eq!(
                projected.authoritative_state_hash(),
                metadata_only.authoritative_state_hash()
            );
            assert!(metadata_events.reference_projection_cells.is_empty());
            assert!(metadata_events.reference_projection_tiles.is_empty());
        }

        assert!(metadata_only.coordinate_map.is_empty());
        assert!(metadata_only.inv_coordinate_map.is_empty());
        assert!(metadata_only
            .cells
            .values()
            .all(|cell| cell.memory.is_empty() && cell.message_queue.is_empty()));
    }

    #[test]
    fn plant_state_is_canonical_and_survives_runtime_restore() {
        let mut engine = Engine::new(
            1,
            1,
            10,
            CellConfig::default(),
            Some(42),
            ReferenceRuleset::default(),
        );
        engine.world.energy[0] = Some(EnergySource::Plant {
            rate: 3,
            current_energy: 4,
            max_energy: 10,
        });
        engine.initialize_reference_state().unwrap();
        {
            let simulation = engine.reference_simulation.as_mut().unwrap();
            simulation
                .tile_state_mut(TileIndex(0))
                .unwrap()
                .diffuse_energy = 5;
            simulation.advance_clock_to(SimTime(1024)).unwrap();
        }
        engine.refresh_reference_projection().unwrap();

        let tile = engine
            .reference_simulation()
            .unwrap()
            .tile_state(TileIndex(0))
            .unwrap();
        assert_eq!(tile.plant_energy, 7);
        assert_eq!(tile.diffuse_energy, 2);
        assert_eq!(tile.plant_capacity, 10);
        assert_eq!(tile.plant_growth_rate, 3);

        let checkpoint = engine.export_reference_checkpoint().unwrap();
        let mut restored = Engine::new(
            1,
            1,
            10,
            CellConfig::default(),
            Some(99),
            ReferenceRuleset::default(),
        );
        restored.restore_reference_checkpoint(checkpoint).unwrap();
        assert_eq!(
            restored.reference_simulation().unwrap().canonical_state(),
            engine.reference_simulation().unwrap().canonical_state()
        );
        assert!(
            restored.world.energy[0]
                == Some(EnergySource::Plant {
                    rate: 3,
                    current_energy: 7,
                    max_energy: 10,
                })
        );
        assert_projection_matches_canonical(&restored);
    }

    #[test]
    fn test_add_team_with_minds() {
        let mut engine = create_test_engine(42);
        let minds = vec![RandomMind::new()];
        engine.add_team_with_minds(TeamId(0), minds).unwrap();
        assert_eq!(
            engine.cells.len(),
            engine.cell_config.starting_cells_per_team
        );
        assert_eq!(engine.reference_minds.len(), 1);
    }

    #[test]
    fn duplicate_team_registration_is_rejected_without_placing_more_cells() {
        let mut engine = create_test_engine(42);
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        let cells_before = engine.cells.len();

        let error = engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap_err();

        assert!(error.contains("already has a Mind pool"));
        assert_eq!(engine.cells.len(), cells_before);
        assert_eq!(engine.reference_minds.len(), 1);
    }

    #[test]
    fn test_tick_advances_iteration() {
        let mut engine = create_test_engine(42);
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        engine.tick(false).unwrap();
        assert_eq!(engine.iteration, 1);
        assert_projection_matches_canonical(&engine);
    }

    #[test]
    fn test_step_multiple() {
        let mut engine = create_test_engine(42);
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        let executed = engine.step(10, false).unwrap();
        assert_eq!(executed, 10);
        assert_eq!(engine.iteration, 10);
    }

    #[test]
    fn test_step_respects_max_iterations() {
        let mut engine = Engine::new(
            32,
            32,
            5,
            CellConfig::default(),
            Some(42),
            ReferenceRuleset::default(),
        );
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        let executed = engine.step(100, false).unwrap();
        assert_eq!(executed, 5);
    }

    #[test]
    fn test_two_teams() {
        let mut engine = create_test_engine(42);
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        engine
            .add_team_with_minds(TeamId(1), vec![RandomMind::new()])
            .unwrap();
        assert_eq!(
            engine.cells.len(),
            engine.cell_config.starting_cells_per_team * 2
        );
        engine.step(50, false).unwrap();
        assert_eq!(engine.iteration, 50);
    }

    #[test]
    fn test_cell_aging() {
        let mut engine = create_test_engine(42);
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        for cell in engine.cells.values() {
            assert_eq!(cell.age, 0);
        }
        engine.tick(false).unwrap();
        for cell in engine.cells.values() {
            assert_eq!(cell.age, 1);
        }
    }

    #[test]
    fn test_step_result() {
        let mut engine = create_test_engine(42);
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        engine
            .add_team_with_minds(TeamId(1), vec![RandomMind::new()])
            .unwrap();
        let result = engine.get_step_result();
        assert_eq!(result.iteration, 0);
        assert_eq!(result.cells_alive.len(), 2);
        assert!(!result.done);
    }

    #[test]
    fn private_random_actions_are_worker_assignment_invariant() {
        let mut sequential = Engine::new(
            32,
            32,
            20,
            CellConfig::default(),
            Some(77),
            ReferenceRuleset::default(),
        );
        let mut parallel = Engine::new(
            32,
            32,
            20,
            CellConfig::default(),
            Some(77),
            ReferenceRuleset::default(),
        );
        sequential
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        parallel
            .add_team_with_minds(TeamId(0), (0..4).map(|_| RandomMind::new()).collect())
            .unwrap();
        sequential.tick(false).unwrap();
        parallel.tick(false).unwrap();
        assert!(!parallel.last_reference_host_timings().parallel_observation);
        sequential.step(9, false).unwrap();
        parallel.step(9, false).unwrap();

        assert!(sequential.world == parallel.world);
        assert_eq!(sequential.cells, parallel.cells);
        assert_eq!(sequential.coordinate_map, parallel.coordinate_map);
        assert_eq!(sequential.inv_coordinate_map, parallel.inv_coordinate_map);
        assert_eq!(sequential.iteration, parallel.iteration);
    }

    fn uniform_reference_rules() -> ReferenceRuleset {
        let duration = DurationRule::new(1024, 0, 1);
        ReferenceRuleset {
            wait_duration: duration,
            move_duration: duration,
            attack_duration: duration,
            consume_duration: duration,
            split_duration: duration,
            regurgitate_duration: duration,
            digestion_rate_numerator: 0,
            metabolism_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        }
    }

    fn insert_test_cell(engine: &mut Engine, id: usize, coordinate: Coordinate) {
        let cell_id = CellId(id);
        engine.cells.insert(
            cell_id,
            Cell::new(cell_id, TeamId(0), 100, engine.cell_config.min_energy),
        );
        engine.coordinate_map.insert(coordinate, cell_id);
        engine.inv_coordinate_map.insert(cell_id, coordinate);
        engine.next_cell_id = engine.next_cell_id.max(id + 1);
    }

    fn exact(action: blob_interface::reference_mind::ReferenceMindAction) -> ReferenceMindDecision {
        ReferenceMindDecision {
            action,
            signal: None,
            memory_update: ReferenceMemoryUpdate::Replace(vec![0; 2048]),
        }
    }

    fn assert_projection_matches_canonical(engine: &Engine) {
        let simulation = engine.reference_simulation().unwrap();
        let state = simulation.canonical_state();
        assert_eq!(engine.cells.len(), state.cells.len());
        assert_eq!(engine.coordinate_map.len(), state.cells.len());
        assert_eq!(engine.inv_coordinate_map.len(), state.cells.len());
        for (key, reference_cell) in &state.cells {
            let id = CellId(usize::try_from(key.0).unwrap());
            let cell = &engine.cells[&id];
            assert_eq!(u64::from(cell.energy), reference_cell.assimilated_energy);
            assert_eq!(u64::from(cell.min_energy), reference_cell.core_mass);
            assert_eq!(cell.marker, reference_cell.marker);
            assert_eq!(cell.loaded, reference_cell.carried_material_mass > 0);
            assert_eq!(
                cell.memory.as_slice(),
                reference_cell.private_memory.as_ref()
            );
            assert_eq!(cell.defending, reference_cell.guarded);
            let (x, y) = simulation
                .neighborhood()
                .coordinate(reference_cell.position)
                .unwrap();
            let coordinate = Coordinate { x, y };
            assert_eq!(engine.inv_coordinate_map[&id], coordinate);
            assert_eq!(engine.coordinate_map[&coordinate], id);
        }
        let mut expected_world = World::new(engine.world.dimensions.0, engine.world.dimensions.1);
        for (index, tile) in state.tiles.iter().enumerate() {
            project_reference_tile(&mut expected_world, index, tile).unwrap();
        }
        assert!(engine.world == expected_world);
        assert_eq!(
            engine.next_cell_id,
            usize::try_from(state.next_cell_key).unwrap()
        );
    }

    #[test]
    fn canonical_engine_resolves_a_snapshot_gated_convoy() {
        let config = CellConfig {
            starting_cells_per_team: 0,
            ..CellConfig::default()
        };
        let mut engine = Engine::new(3, 1, 10, config, Some(9), uniform_reference_rules());
        insert_test_cell(&mut engine, 0, Coordinate { x: 0, y: 0 });
        insert_test_cell(&mut engine, 1, Coordinate { x: 1, y: 0 });

        let actions = HashMap::from([
            (
                CellId(0),
                exact(blob_interface::reference_mind::ReferenceMindAction::Move {
                    target_slot: 4,
                    effort: blob_interface::reference_mind::ReferenceEffort::Standard,
                }),
            ),
            (
                CellId(1),
                exact(blob_interface::reference_mind::ReferenceMindAction::Move {
                    target_slot: 4,
                    effort: blob_interface::reference_mind::ReferenceEffort::Standard,
                }),
            ),
        ]);
        let events = engine
            .tick_reference_with_overrides(&actions, false)
            .unwrap();
        let report = events.reference_batch.as_ref().unwrap();
        assert_eq!(report.outcomes.len(), 2);
        assert_eq!(report.outcomes[0].status, OutcomeStatus::Frustrated);
        assert_eq!(report.outcomes[1].status, OutcomeStatus::Success);
        assert_eq!(
            engine.inv_coordinate_map[&CellId(0)],
            Coordinate { x: 0, y: 0 }
        );
        assert_eq!(
            engine.inv_coordinate_map[&CellId(1)],
            Coordinate { x: 2, y: 0 }
        );
        assert_eq!(
            Some(engine.reference_simulation().unwrap().state_hash()),
            report.state_hash()
        );
        assert_projection_matches_canonical(&engine);
        let runtime_hash = engine.authoritative_state_hash().unwrap();
        engine.cells.get_mut(&CellId(0)).unwrap().random_lineage += 1;
        assert_ne!(runtime_hash, engine.authoritative_state_hash().unwrap());
        assert_eq!(
            Some(engine.reference_simulation().unwrap().state_hash()),
            report.state_hash(),
            "host metadata is outside the core hash but inside the runtime hash"
        );
    }

    #[test]
    fn ordered_decision_merge_fills_missing_minds_with_waits() {
        let config = CellConfig {
            starting_cells_per_team: 0,
            ..CellConfig::default()
        };
        let mut engine = Engine::new(3, 1, 10, config, Some(15), uniform_reference_rules());
        for id in 0..3 {
            insert_test_cell(&mut engine, id, Coordinate { x: id, y: 0 });
        }
        let events = engine
            .tick_reference_with_overrides(
                &HashMap::from([(
                    CellId(1),
                    exact(blob_interface::reference_mind::ReferenceMindAction::Guard {
                        effort: blob_interface::reference_mind::ReferenceEffort::Standard,
                    }),
                )]),
                false,
            )
            .unwrap();

        assert_eq!(
            events
                .reference_commitments
                .iter()
                .map(|commitment| commitment.actor)
                .collect::<Vec<_>>(),
            vec![CellKey(0), CellKey(1), CellKey(2)]
        );
        assert!(matches!(
            events.reference_commitments[0].request,
            ActionRequest::Wait
        ));
        assert!(matches!(
            events.reference_commitments[1].request,
            ActionRequest::Guard { .. }
        ));
        assert!(matches!(
            events.reference_commitments[2].request,
            ActionRequest::Wait
        ));
        assert_projection_matches_canonical(&engine);
    }

    #[test]
    fn delta_projection_tracks_births_without_rebuilding_survivors() {
        let config = CellConfig {
            starting_cells_per_team: 0,
            ..CellConfig::default()
        };
        let mut engine = Engine::new(3, 1, 10, config, Some(10), uniform_reference_rules());
        insert_test_cell(&mut engine, 0, Coordinate { x: 0, y: 0 });
        engine.cells.get_mut(&CellId(0)).unwrap().team_id = TeamId(7);

        let events = engine
            .tick_reference_with_overrides(
                &HashMap::from([(
                    CellId(0),
                    exact(blob_interface::reference_mind::ReferenceMindAction::Split {
                        target_slot: 4,
                        child_allocation: 30,
                        marker: 77,
                        private_memory: vec![9; 32],
                    }),
                )]),
                false,
            )
            .unwrap();

        assert_eq!(events.splits, vec![(CellId(0), CellId(1))]);
        assert_eq!(engine.cells[&CellId(0)].age, 1);
        assert_eq!(engine.cells[&CellId(1)].age, 0);
        assert_eq!(engine.cells[&CellId(1)].team_id, TeamId(7));
        assert_eq!(engine.cells[&CellId(1)].marker, 77);
        assert_projection_matches_canonical(&engine);
    }

    #[test]
    fn metadata_only_birth_inherits_private_dispatch_identity() {
        let config = CellConfig {
            starting_cells_per_team: 0,
            ..CellConfig::default()
        };
        let mut engine = Engine::new(3, 1, 10, config, Some(10), uniform_reference_rules());
        engine
            .set_reference_host_mode(ReferenceHostMode::MetadataOnly)
            .unwrap();
        insert_test_cell(&mut engine, 0, Coordinate { x: 0, y: 0 });
        engine.cells.get_mut(&CellId(0)).unwrap().team_id = TeamId(7);

        let events = engine
            .tick_reference_with_overrides(
                &HashMap::from([(
                    CellId(0),
                    exact(blob_interface::reference_mind::ReferenceMindAction::Split {
                        target_slot: 4,
                        child_allocation: 30,
                        marker: 77,
                        private_memory: vec![9; 32],
                    }),
                )]),
                false,
            )
            .unwrap();

        assert_eq!(events.splits, vec![(CellId(0), CellId(1))]);
        assert_eq!(engine.cells[&CellId(1)].team_id, TeamId(7));
        assert_eq!(engine.cells[&CellId(1)].random_lineage, 1);
        assert_eq!(engine.cells[&CellId(1)].decision_sequence, 0);
        assert!(engine.cells[&CellId(1)].memory.is_empty());
        assert!(engine.coordinate_map.is_empty());
        assert!(engine.inv_coordinate_map.is_empty());
    }

    #[test]
    fn canonical_engine_attributes_an_attack_that_completes_in_a_later_driver_tick() {
        let config = CellConfig {
            starting_cells_per_team: 0,
            ..CellConfig::default()
        };
        let mut rules = uniform_reference_rules();
        rules.attack_duration = DurationRule::new(4096, 0, 1);
        let mut engine = Engine::new(3, 1, 10, config, Some(13), rules);
        insert_test_cell(&mut engine, 0, Coordinate { x: 0, y: 0 });
        insert_test_cell(&mut engine, 1, Coordinate { x: 1, y: 0 });
        let victim = engine.cells.get_mut(&CellId(1)).unwrap();
        victim.team_id = TeamId(1);
        victim.energy = 10;

        let first = HashMap::from([
            (
                CellId(0),
                exact(
                    blob_interface::reference_mind::ReferenceMindAction::Attack {
                        target_slot: 4,
                        effort: blob_interface::reference_mind::ReferenceEffort::Standard,
                        payload: 20,
                    },
                ),
            ),
            (
                CellId(1),
                exact(blob_interface::reference_mind::ReferenceMindAction::Guard {
                    effort: blob_interface::reference_mind::ReferenceEffort::Standard,
                }),
            ),
        ]);
        assert!(engine
            .tick_reference_with_overrides(&first, false)
            .unwrap()
            .kills
            .is_empty());

        let mut kills = Vec::new();
        for _ in 0..4 {
            kills.extend(engine.tick(false).unwrap().kills);
            if !kills.is_empty() {
                break;
            }
        }
        assert_eq!(kills, vec![(CellId(0), CellId(1), TeamId(1))]);
        assert_projection_matches_canonical(&engine);
    }

    #[test]
    fn host_visible_wait_noop_does_not_hide_same_batch_attack_damage() {
        let config = CellConfig {
            starting_cells_per_team: 0,
            ..CellConfig::default()
        };
        let mut engine = Engine::new(3, 1, 2, config, Some(29), uniform_reference_rules());
        insert_test_cell(&mut engine, 0, Coordinate { x: 0, y: 0 });
        insert_test_cell(&mut engine, 1, Coordinate { x: 1, y: 0 });
        let before = engine.cells[&CellId(1)].energy;
        let actions = HashMap::from([
            (
                CellId(0),
                exact(
                    blob_interface::reference_mind::ReferenceMindAction::Attack {
                        target_slot: 4,
                        effort: blob_interface::reference_mind::ReferenceEffort::Standard,
                        payload: 20,
                    },
                ),
            ),
            (
                CellId(1),
                exact(blob_interface::reference_mind::ReferenceMindAction::Wait),
            ),
        ]);

        engine
            .tick_reference_with_overrides(&actions, false)
            .unwrap();

        assert!(engine.cells[&CellId(1)].energy < before);
        assert_projection_matches_canonical(&engine);
    }

    #[test]
    fn canonical_engine_is_worker_assignment_invariant() {
        let mut sequential = Engine::new(
            32,
            32,
            20,
            CellConfig::default(),
            Some(77),
            ReferenceRuleset::default(),
        );
        let mut parallel = Engine::new(
            32,
            32,
            20,
            CellConfig::default(),
            Some(77),
            ReferenceRuleset::default(),
        );
        sequential
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        parallel
            .add_team_with_minds(TeamId(0), (0..4).map(|_| RandomMind::new()).collect())
            .unwrap();
        sequential.start_reference_replay_recording(3).unwrap();
        parallel.start_reference_replay_recording(3).unwrap();

        sequential.step(10, false).unwrap();
        parallel.step(10, false).unwrap();

        assert_eq!(
            sequential.reference_simulation().unwrap().state_hash(),
            parallel.reference_simulation().unwrap().state_hash()
        );
        assert!(sequential.world == parallel.world);
        assert_eq!(sequential.cells, parallel.cells);
        assert_eq!(sequential.coordinate_map, parallel.coordinate_map);
        assert_eq!(sequential.inv_coordinate_map, parallel.inv_coordinate_map);
        let sequential_replay = sequential
            .export_reference_replay_bundle(ReplayBundleLimits::default())
            .unwrap();
        let parallel_replay = parallel
            .export_reference_replay_bundle(ReplayBundleLimits::default())
            .unwrap();
        assert_eq!(sequential_replay.to_bytes(), parallel_replay.to_bytes());
        assert_eq!(sequential_replay.archive().event_count(), 10);
        assert_eq!(sequential_replay.seek(7).unwrap().events, 6..7);
    }

    #[test]
    fn reference_replay_stream_rotates_with_pre_mutation_backpressure() {
        let mut engine = Engine::new(
            32,
            32,
            20,
            CellConfig::default(),
            Some(91),
            ReferenceRuleset::default(),
        );
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        engine
            .start_reference_replay_streaming(ReplayStreamConfig {
                max_events_per_segment: 2,
                max_event_bytes_per_segment: usize::MAX / 2,
                segment_limits: ReplaySegmentLimits {
                    max_segment_bytes: usize::MAX,
                    ..ReplaySegmentLimits::default()
                },
                ..ReplayStreamConfig::default()
            })
            .unwrap();

        engine.step(2, false).unwrap();
        let manifest = engine.reference_replay_stream_manifest().unwrap();
        assert_eq!(manifest.segment_count(), 1);
        assert_eq!(manifest.event_count(), 2);
        let hash_before_backpressure = engine.authoritative_state_hash();
        let iteration_before_backpressure = engine.iteration;
        assert!(engine.tick(false).is_err());
        assert_eq!(engine.authoritative_state_hash(), hash_before_backpressure);
        assert_eq!(engine.iteration, iteration_before_backpressure);

        let first = engine.take_reference_replay_segment().unwrap();
        manifest.verify_segment(0, &first).unwrap();
        assert!(!first
            .event(0, ReplayLimits::default())
            .unwrap()
            .commitments
            .is_empty());
        let mut replay = ReferenceReplayDriver::from_segment(&first).unwrap();
        replay
            .apply_segment(&first, ReplayLimits::default())
            .unwrap();
        assert_eq!(replay.cursor(), first.end_cursor());
        assert_eq!(
            replay.simulation().state_hash(),
            first.end_cursor().previous_state_hash
        );
        engine.step(1, false).unwrap();
        assert!(engine.flush_reference_replay_stream().unwrap());
        let second = engine.take_reference_replay_segment().unwrap();
        let manifest = engine.reference_replay_stream_manifest().unwrap();
        assert_eq!(manifest.segment_count(), 2);
        assert_eq!(manifest.event_count(), 3);
        manifest.verify_segment(1, &second).unwrap();
        assert!(!engine.flush_reference_replay_stream().unwrap());
    }

    #[test]
    fn canonical_minds_are_worker_invariant_and_persist_explicit_memory() {
        let config = CellConfig {
            starting_cells_per_team: 4,
            ..CellConfig::default()
        };
        let mut sequential = Engine::new(
            24,
            24,
            10,
            config.clone(),
            Some(123),
            ReferenceRuleset::default(),
        );
        let mut pooled = Engine::new(24, 24, 10, config, Some(123), ReferenceRuleset::default());
        sequential
            .add_team_with_minds(TeamId(0), vec![MemoryReferenceMind])
            .unwrap();
        pooled
            .add_team_with_minds(TeamId(0), (0..4).map(|_| MemoryReferenceMind).collect())
            .unwrap();
        pooled.set_reference_mind_parallel_threshold(Some(2));

        for _ in 0..3 {
            let sequential_tick = sequential.tick(false).unwrap();
            let pooled_tick = pooled.tick(false).unwrap();
            assert!(pooled.last_reference_host_timings().parallel_observation);
            assert!(sequential_tick
                .reference_commitments
                .iter()
                .all(|commitment| matches!(
                    &commitment.memory_update,
                    ReferenceMemoryUpdate::Replace(bytes)
                        if bytes[0] == sequential.iteration as u8
                )));
            assert_eq!(
                sequential_tick.reference_commitments,
                pooled_tick.reference_commitments
            );
        }
        assert_eq!(
            sequential.reference_simulation().unwrap().state_hash(),
            pooled.reference_simulation().unwrap().state_hash()
        );
        assert!(sequential
            .reference_simulation()
            .unwrap()
            .cells()
            .values()
            .all(|cell| cell.private_memory[0] == 3));
    }

    #[test]
    fn failed_reference_mind_invocation_does_not_advance_private_randomness() {
        let config = CellConfig {
            starting_cells_per_team: 4,
            ..CellConfig::default()
        };
        let mut engine = Engine::new(24, 24, 10, config, Some(124), ReferenceRuleset::default());
        engine
            .add_team_with_minds(TeamId(0), (0..4).map(|_| FailingReferenceMind).collect())
            .unwrap();
        engine.set_reference_mind_parallel_threshold(Some(2));
        let sequences_before: Vec<_> = engine
            .cells
            .iter()
            .map(|(cell_id, cell)| (*cell_id, cell.decision_sequence))
            .collect();

        let error = engine.tick(false).unwrap_err();

        assert!(error.contains("intentional reset failure"));
        for (cell_id, sequence) in sequences_before {
            assert_eq!(engine.cells[&cell_id].decision_sequence, sequence);
        }
    }
}
