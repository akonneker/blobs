//! Browser-facing replay and attestation API.

use std::collections::BTreeSet;
use std::fmt::Write;

use blob_engine::resolution::{
    ActionKind, CanonicalHash, MatchAttestation, MatchVerificationLimits, OutcomeStatus,
    ReferenceReplayDriver, ReplayLimits, ReplayManifest, ReplayManifestLimits, ReplaySegment,
    ReplaySegmentLimits, TileIndex,
};
use wasm_bindgen::prelude::*;

pub const BROWSER_API_VERSION: u16 = 1;
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_MANIFEST_SEGMENTS: usize = 100_000;
const MAX_SEGMENT_BYTES: usize = 64 * 1024 * 1024;
const MAX_SEGMENT_EVENTS: usize = 4_096;
const MAX_ATTESTATION_BYTES: usize = 2 * 1024 * 1024;
const NO_OCCUPANT: u64 = u64::MAX;

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn browser_manifest_limits() -> ReplayManifestLimits {
    ReplayManifestLimits {
        max_manifest_bytes: MAX_MANIFEST_BYTES,
        max_segments: MAX_MANIFEST_SEGMENTS,
    }
}

fn browser_segment_limits() -> ReplaySegmentLimits {
    ReplaySegmentLimits {
        max_segment_bytes: MAX_SEGMENT_BYTES,
        max_events: MAX_SEGMENT_EVENTS,
        ..ReplaySegmentLimits::default()
    }
}

#[wasm_bindgen]
pub fn browser_api_version() -> u16 {
    BROWSER_API_VERSION
}

/// Indexed, checkpoint-backed replay session. Loading a segment seeks directly
/// to its checkpoint and re-executes only the requested bounded prefix.
#[wasm_bindgen]
pub struct BrowserReplay {
    manifest: ReplayManifest,
    segment: Option<ReplaySegment>,
    driver: Option<ReferenceReplayDriver>,
}

#[wasm_bindgen]
impl BrowserReplay {
    #[wasm_bindgen(constructor)]
    pub fn new(manifest_bytes: &[u8]) -> Result<BrowserReplay, JsValue> {
        let manifest =
            ReplayManifest::from_bytes_with_limits(manifest_bytes, browser_manifest_limits())
                .map_err(js_error)?;
        Ok(Self {
            manifest,
            segment: None,
            driver: None,
        })
    }

    #[wasm_bindgen(getter, js_name = manifestHash)]
    pub fn manifest_hash(&self) -> String {
        self.manifest.manifest_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = compiledRulesetHash)]
    pub fn compiled_ruleset_hash(&self) -> String {
        self.manifest.compiled_ruleset_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = initialStateHash)]
    pub fn initial_state_hash(&self) -> String {
        self.manifest.initial_state_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = segmentCount)]
    pub fn segment_count(&self) -> u32 {
        self.manifest.segment_count().try_into().unwrap_or(u32::MAX)
    }

    #[wasm_bindgen(getter, js_name = eventCount)]
    pub fn event_count(&self) -> u64 {
        self.manifest.event_count()
    }

    /// Resolves a global event sequence to the independently fetchable segment
    /// that contains it.
    #[wasm_bindgen(js_name = segmentIndexForEvent)]
    pub fn segment_index_for_event(&self, sequence: u64) -> Result<u32, JsValue> {
        self.manifest
            .segment_for_event(sequence)
            .map_err(js_error)?
            .try_into()
            .map_err(js_error)
    }

    /// Returns the content-address and event range needed to fetch one segment.
    /// Integer fields are strings to preserve their full range in JavaScript.
    #[wasm_bindgen(js_name = segmentDescriptorJson)]
    pub fn segment_descriptor_json(&self, segment_index: u32) -> Result<String, JsValue> {
        let index = usize::try_from(segment_index).map_err(js_error)?;
        let descriptor = self
            .manifest
            .segment(index)
            .ok_or_else(|| js_error(format!("invalid replay segment index {segment_index}")))?;
        Ok(format!(
            "{{\"index\":{},\"startEvent\":\"{}\",\"endEvent\":\"{}\",\"eventCount\":\"{}\",\"checkpointHash\":\"{}\",\"segmentHash\":\"{}\",\"byteLength\":\"{}\"}}",
            segment_index,
            descriptor.start_cursor.next_sequence,
            descriptor.end_cursor.next_sequence,
            descriptor.event_count(),
            descriptor.checkpoint_hash.to_hex(),
            descriptor.segment_hash.to_hex(),
            descriptor.byte_length,
        ))
    }

    /// Loads one content-addressed segment and seeks to the state after
    /// `applied_events` global events. Work is capped by `max_prefix_events`.
    #[wasm_bindgen(js_name = loadSegment)]
    pub fn load_segment(
        &mut self,
        segment_index: u32,
        segment_bytes: &[u8],
        applied_events: u64,
        max_prefix_events: u32,
    ) -> Result<(), JsValue> {
        let segment_index = usize::try_from(segment_index).map_err(js_error)?;
        let segment =
            ReplaySegment::from_bytes_with_limits(segment_bytes, browser_segment_limits())
                .map_err(js_error)?;
        self.manifest
            .verify_segment(segment_index, &segment)
            .map_err(js_error)?;
        let start = segment.start_cursor().next_sequence;
        let end = segment.end_cursor().next_sequence;
        if applied_events < start || applied_events > end {
            return Err(js_error(format!(
                "applied event count {applied_events} is outside segment range {start}..={end}"
            )));
        }
        let prefix_u64 = applied_events - start;
        if prefix_u64 > u64::from(max_prefix_events) {
            return Err(js_error(format!(
                "seek prefix {prefix_u64} exceeds browser work limit {max_prefix_events}"
            )));
        }
        let prefix = usize::try_from(prefix_u64).map_err(js_error)?;
        let mut driver = ReferenceReplayDriver::from_segment(&segment).map_err(js_error)?;
        driver
            .apply_segment_prefix(&segment, prefix, ReplayLimits::default())
            .map_err(js_error)?;
        self.segment = Some(segment);
        self.driver = Some(driver);
        Ok(())
    }

    #[wasm_bindgen(js_name = step)]
    pub fn step(&mut self) -> Result<BrowserReplayEvent, JsValue> {
        let segment = self
            .segment
            .as_ref()
            .ok_or_else(|| js_error("no replay segment is loaded"))?;
        let driver = self
            .driver
            .as_mut()
            .ok_or_else(|| js_error("no replay driver is loaded"))?;
        let sequence = driver.cursor().next_sequence;
        let relative = sequence
            .checked_sub(segment.start_cursor().next_sequence)
            .ok_or_else(|| js_error("replay cursor precedes the loaded segment"))?;
        let relative = usize::try_from(relative).map_err(js_error)?;
        if relative >= segment.event_count() {
            return Err(js_error("loaded replay segment is exhausted"));
        }
        let event = segment
            .event(relative, ReplayLimits::default())
            .map_err(js_error)?;
        let mut changed_tiles: BTreeSet<u32> = event
            .delta
            .tiles
            .iter()
            .map(|delta| delta.tile.0.try_into().unwrap_or(u32::MAX))
            .collect();
        let mut changed_cells: BTreeSet<u64> =
            event.delta.cells.iter().map(|delta| delta.cell.0).collect();
        if event.completed_at > driver.simulation().now() {
            changed_tiles.extend(
                driver
                    .simulation()
                    .growing_plant_tiles()
                    .iter()
                    .filter(|tile| {
                        let state = driver
                            .simulation()
                            .tile_state(**tile)
                            .expect("derived plant tile index");
                        (state.diffuse_energy > 0 && state.plant_energy < state.plant_capacity)
                            || state.plant_growth_remainder > 0
                    })
                    .map(|tile| tile.0.try_into().unwrap_or(u32::MAX)),
            );
            changed_tiles.extend(
                driver
                    .simulation()
                    .field_changed_tiles_at(event.completed_at)
                    .into_iter()
                    .map(|tile| tile.0.try_into().unwrap_or(u32::MAX)),
            );
            changed_cells.extend(
                driver
                    .simulation()
                    .cells()
                    .iter()
                    .filter(|(_, cell)| cell.gut_energy != 0)
                    .map(|(key, _)| key.0),
            );
            if driver.simulation().rules().metabolism_rate_numerator > 0 {
                changed_cells.extend(
                    driver
                        .simulation()
                        .cells()
                        .iter()
                        .filter(|(_, cell)| cell.assimilated_energy > 0)
                        .map(|(key, _)| key.0),
                );
                changed_tiles.extend(
                    driver
                        .simulation()
                        .cells()
                        .values()
                        .filter(|cell| cell.assimilated_energy > 0)
                        .map(|cell| cell.position.0.try_into().unwrap_or(u32::MAX)),
                );
            }
        }
        let deaths = event.deaths.iter().map(|key| key.0).collect();
        let births = event
            .births
            .iter()
            .flat_map(|(parent, child)| [parent.0, child.0])
            .collect();
        driver
            .apply_event(&event, ReplayLimits::default())
            .map_err(js_error)?;
        Ok(BrowserReplayEvent {
            sequence: event.sequence,
            completed_at: event.completed_at.0,
            commitment_count: event.commitments.len().try_into().unwrap_or(u32::MAX),
            outcome_count: event.outcomes.len().try_into().unwrap_or(u32::MAX),
            death_count: event.deaths.len().try_into().unwrap_or(u32::MAX),
            birth_count: event.births.len().try_into().unwrap_or(u32::MAX),
            state_hash: event.post_state_hash,
            changed_tiles: changed_tiles.into_iter().collect(),
            changed_cells: changed_cells.into_iter().collect(),
            deaths,
            births,
        })
    }

    #[wasm_bindgen(getter, js_name = currentSequence)]
    pub fn current_sequence(&self) -> Option<u64> {
        self.driver
            .as_ref()
            .map(|driver| driver.cursor().next_sequence)
    }

    #[wasm_bindgen(getter, js_name = currentTime)]
    pub fn current_time(&self) -> Option<u64> {
        self.driver
            .as_ref()
            .map(|driver| driver.simulation().now().0)
    }

    #[wasm_bindgen(getter, js_name = stateHash)]
    pub fn state_hash(&self) -> Option<String> {
        self.driver
            .as_ref()
            .map(|driver| driver.simulation().state_hash().to_hex())
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> Option<u32> {
        self.driver
            .as_ref()
            .and_then(|driver| driver.simulation().neighborhood().width().try_into().ok())
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> Option<u32> {
        self.driver
            .as_ref()
            .and_then(|driver| driver.simulation().neighborhood().height().try_into().ok())
    }

    #[wasm_bindgen(js_name = tileElevations)]
    pub fn tile_elevations(&self) -> Result<Vec<i16>, JsValue> {
        let simulation = self.simulation()?;
        Ok((0..simulation.neighborhood().tile_count())
            .map(|index| {
                simulation
                    .tile_state(TileIndex(index))
                    .expect("compiled tile index")
                    .elevation
            })
            .collect())
    }

    /// Returns occupant keys with `2^64-1` representing an empty tile.
    #[wasm_bindgen(js_name = tileOccupants)]
    pub fn tile_occupants(&self) -> Result<Vec<u64>, JsValue> {
        let simulation = self.simulation()?;
        Ok((0..simulation.neighborhood().tile_count())
            .map(|index| {
                simulation
                    .tile_state(TileIndex(index))
                    .expect("compiled tile index")
                    .occupant
                    .map_or(NO_OCCUPANT, |key| key.0)
            })
            .collect())
    }

    #[wasm_bindgen(js_name = tilePlantEnergy)]
    pub fn tile_plant_energy(&self) -> Result<Vec<u64>, JsValue> {
        self.tile_energy(|tile| tile.plant_energy)
    }

    #[wasm_bindgen(js_name = tilePlantCapacity)]
    pub fn tile_plant_capacity(&self) -> Result<Vec<u64>, JsValue> {
        self.tile_energy(|tile| tile.plant_capacity)
    }

    #[wasm_bindgen(js_name = tilePlantGrowthRate)]
    pub fn tile_plant_growth_rate(&self) -> Result<Vec<u64>, JsValue> {
        self.tile_energy(|tile| tile.plant_growth_rate)
    }

    #[wasm_bindgen(js_name = tilePlantGrowthRemainder)]
    pub fn tile_plant_growth_remainder(&self) -> Result<Vec<u64>, JsValue> {
        self.tile_energy(|tile| tile.plant_growth_remainder)
    }

    #[wasm_bindgen(js_name = tileLooseEnergy)]
    pub fn tile_loose_energy(&self) -> Result<Vec<u64>, JsValue> {
        self.tile_energy(|tile| tile.loose_energy)
    }

    #[wasm_bindgen(js_name = tileDiffuseEnergy)]
    pub fn tile_diffuse_energy(&self) -> Result<Vec<u64>, JsValue> {
        self.tile_energy(|tile| tile.diffuse_energy)
    }

    #[wasm_bindgen(js_name = tileDiffusionRemainder)]
    pub fn tile_diffusion_remainder(&self) -> Result<Vec<u64>, JsValue> {
        self.tile_energy(|tile| tile.diffusion_remainder)
    }

    #[wasm_bindgen(js_name = tileSignalEnergy)]
    pub fn tile_signal_energy(&self, channel: u8) -> Result<Vec<u64>, JsValue> {
        let channel = usize::from(channel);
        if channel >= blob_engine::resolution::SIGNAL_CHANNELS {
            return Err(js_error("invalid signal channel"));
        }
        self.tile_energy(|tile| tile.signal_energy[channel])
    }

    /// Compact JSON metadata for variable-sized cell state. All u64 values are
    /// decimal strings so JavaScript does not lose integer precision.
    #[wasm_bindgen(js_name = cellsJson)]
    pub fn cells_json(&self) -> Result<String, JsValue> {
        let simulation = self.simulation()?;
        let width = simulation.neighborhood().width();
        let mut json = String::from("[");
        for (index, (key, cell)) in simulation.cells().iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write_cell_json(&mut json, *key, cell, width);
        }
        json.push(']');
        Ok(json)
    }

    /// Returns only selected current tile states, in request order.
    #[wasm_bindgen(js_name = tilesJson)]
    pub fn tiles_json(&self, indices: &[u32]) -> Result<String, JsValue> {
        let simulation = self.simulation()?;
        let mut json = String::from("[");
        for (position, index) in indices.iter().copied().enumerate() {
            let index = usize::try_from(index).map_err(js_error)?;
            let tile = simulation
                .tile_state(TileIndex(index))
                .ok_or_else(|| js_error(format!("invalid tile index {index}")))?;
            if position != 0 {
                json.push(',');
            }
            write!(
                json,
                "{{\"tile\":{},\"elevation\":{},\"occupant\":{},\"plantEnergy\":\"{}\",\"plantCapacity\":\"{}\",\"plantGrowthRate\":\"{}\",\"plantGrowthRemainder\":\"{}\",\"looseEnergy\":\"{}\",\"diffuseEnergy\":\"{}\",\"diffusionRemainder\":\"{}\",\"signalEnergy\":{},\"signalDecayRemainder\":{}}}",
                index,
                tile.elevation,
                tile.occupant.map_or_else(|| "null".to_owned(), |key| format!("\"{}\"", key.0)),
                tile.plant_energy,
                tile.plant_capacity,
                tile.plant_growth_rate,
                tile.plant_growth_remainder,
                tile.loose_energy,
                tile.diffuse_energy,
                tile.diffusion_remainder,
                u64_array_json(&tile.signal_energy),
                u64_array_json(&tile.signal_decay_remainder),
            )
            .expect("writing JSON to String cannot fail");
        }
        json.push(']');
        Ok(json)
    }

    /// Returns selected cells as `{ key, state }` records. `state` is null for
    /// a key that died in the preceding event.
    #[wasm_bindgen(js_name = cellsByKeyJson)]
    pub fn cells_by_key_json(&self, keys: &[u64]) -> Result<String, JsValue> {
        let simulation = self.simulation()?;
        let width = simulation.neighborhood().width();
        let mut json = String::from("[");
        for (index, key) in keys.iter().copied().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write!(json, "{{\"key\":\"{key}\",\"state\":")
                .expect("writing JSON to String cannot fail");
            if let Some(cell) = simulation.cell(blob_engine::resolution::CellKey(key)) {
                write_cell_json(
                    &mut json,
                    blob_engine::resolution::CellKey(key),
                    cell,
                    width,
                );
            } else {
                json.push_str("null");
            }
            json.push('}');
        }
        json.push(']');
        Ok(json)
    }
}

impl BrowserReplay {
    fn simulation(&self) -> Result<&blob_engine::resolution::ReferenceSimulation, JsValue> {
        self.driver
            .as_ref()
            .map(ReferenceReplayDriver::simulation)
            .ok_or_else(|| js_error("no replay segment is loaded"))
    }

    fn tile_energy(
        &self,
        value: impl Fn(&blob_engine::resolution::TileState) -> u64,
    ) -> Result<Vec<u64>, JsValue> {
        let simulation = self.simulation()?;
        Ok((0..simulation.neighborhood().tile_count())
            .map(|index| {
                value(
                    simulation
                        .tile_state(TileIndex(index))
                        .expect("compiled tile index"),
                )
            })
            .collect())
    }
}

#[wasm_bindgen]
pub struct BrowserReplayEvent {
    sequence: u64,
    completed_at: u64,
    commitment_count: u32,
    outcome_count: u32,
    death_count: u32,
    birth_count: u32,
    state_hash: CanonicalHash,
    changed_tiles: Vec<u32>,
    changed_cells: Vec<u64>,
    deaths: Vec<u64>,
    births: Vec<u64>,
}

#[wasm_bindgen]
impl BrowserReplayEvent {
    #[wasm_bindgen(getter)]
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    #[wasm_bindgen(getter, js_name = completedAt)]
    pub fn completed_at(&self) -> u64 {
        self.completed_at
    }

    #[wasm_bindgen(getter, js_name = commitmentCount)]
    pub fn commitment_count(&self) -> u32 {
        self.commitment_count
    }

    #[wasm_bindgen(getter, js_name = outcomeCount)]
    pub fn outcome_count(&self) -> u32 {
        self.outcome_count
    }

    #[wasm_bindgen(getter, js_name = deathCount)]
    pub fn death_count(&self) -> u32 {
        self.death_count
    }

    #[wasm_bindgen(getter, js_name = birthCount)]
    pub fn birth_count(&self) -> u32 {
        self.birth_count
    }

    #[wasm_bindgen(getter, js_name = stateHash)]
    pub fn state_hash(&self) -> String {
        self.state_hash.to_hex()
    }

    /// Canonical tile indices changed by this event. A renderer can refresh
    /// these after taking one full snapshot at segment load.
    #[wasm_bindgen(js_name = changedTiles)]
    pub fn changed_tiles(&self) -> Vec<u32> {
        self.changed_tiles.clone()
    }

    #[wasm_bindgen(js_name = changedCells)]
    pub fn changed_cells(&self) -> Vec<u64> {
        self.changed_cells.clone()
    }

    pub fn deaths(&self) -> Vec<u64> {
        self.deaths.clone()
    }

    /// Alternating parent and child keys: `[parent0, child0, ...]`.
    pub fn births(&self) -> Vec<u64> {
        self.births.clone()
    }
}

/// Portable attestation parser. JavaScript verifies the returned signature and
/// signing message with WebCrypto after the key ID check succeeds.
#[wasm_bindgen]
pub struct BrowserAttestation {
    inner: MatchAttestation,
}

#[wasm_bindgen]
impl BrowserAttestation {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<BrowserAttestation, JsValue> {
        let inner = MatchAttestation::from_bytes(
            bytes,
            MAX_ATTESTATION_BYTES,
            MatchVerificationLimits::default(),
        )
        .map_err(js_error)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter, js_name = keyId)]
    pub fn key_id(&self) -> String {
        self.inner.key_id().to_hex()
    }

    #[wasm_bindgen(js_name = matchesPublicKey)]
    pub fn matches_public_key(&self, public_key: &[u8]) -> bool {
        self.inner.verify_key_id(public_key).is_ok()
    }

    #[wasm_bindgen(js_name = signingMessage)]
    pub fn signing_message(&self) -> Vec<u8> {
        self.inner.signing_message()
    }

    pub fn signature(&self) -> Vec<u8> {
        self.inner.signature().to_vec()
    }

    #[wasm_bindgen(getter, js_name = manifestHash)]
    pub fn manifest_hash(&self) -> String {
        self.inner.manifest().manifest_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = matchIdHash)]
    pub fn match_id_hash(&self) -> String {
        self.inner.manifest().match_id_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = mindAbiHash)]
    pub fn mind_abi_hash(&self) -> String {
        self.inner.manifest().mind_abi_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = mindRuntimeProfile)]
    pub fn mind_runtime_profile(&self) -> String {
        self.inner.manifest().mind_runtime_profile().name().into()
    }

    #[wasm_bindgen(getter, js_name = compiledRulesetHash)]
    pub fn compiled_ruleset_hash(&self) -> String {
        self.inner.manifest().compiled_ruleset_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = initialStateHash)]
    pub fn initial_state_hash(&self) -> String {
        self.inner.manifest().initial_state_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = initialRuntimeHash)]
    pub fn initial_runtime_hash(&self) -> String {
        self.inner.manifest().initial_runtime_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = replayManifestHash)]
    pub fn replay_manifest_hash(&self) -> String {
        self.inner.manifest().replay_manifest_hash().to_hex()
    }

    #[wasm_bindgen(getter, js_name = finalEventCount)]
    pub fn final_event_count(&self) -> u64 {
        self.inner.manifest().final_cursor().next_sequence
    }

    #[wasm_bindgen(js_name = artifactsJson)]
    pub fn artifacts_json(&self) -> String {
        let mut json = String::from("[");
        for (index, binding) in self.inner.manifest().mind_artifacts().iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write!(
                json,
                "{{\"slot\":\"{}\",\"artifactHash\":\"{}\"}}",
                binding.slot,
                binding.artifact_hash.to_hex(),
            )
            .expect("writing JSON to String cannot fail");
        }
        json.push(']');
        json
    }
}

fn action_kind_name(kind: ActionKind) -> &'static str {
    match kind {
        ActionKind::Wait => "wait",
        ActionKind::Move => "move",
        ActionKind::Attack => "attack",
        ActionKind::Guard => "guard",
        ActionKind::Consume => "consume",
        ActionKind::Split => "split",
        ActionKind::Regurgitate => "regurgitate",
        ActionKind::Excavate => "excavate",
        ActionKind::DepositTerrain => "depositTerrain",
    }
}

fn outcome_name(outcome: OutcomeStatus) -> &'static str {
    match outcome {
        OutcomeStatus::Rejected(_) => "rejected",
        OutcomeStatus::Success => "success",
        OutcomeStatus::Frustrated => "frustrated",
        OutcomeStatus::Contested => "contested",
        OutcomeStatus::Interrupted => "interrupted",
    }
}

fn write_cell_json(
    json: &mut String,
    key: blob_engine::resolution::CellKey,
    cell: &blob_engine::resolution::CellState,
    width: usize,
) {
    let x = cell.position.0 % width;
    let y = cell.position.0 / width;
    let pending = cell
        .pending_action
        .as_ref()
        .map_or("none", |action| action_kind_name(action.request.kind()));
    let outcome = cell.last_outcome.map_or("none", outcome_name);
    write!(
        json,
        "{{\"key\":\"{}\",\"tile\":{},\"x\":{},\"y\":{},\"coreMass\":\"{}\",\"assimilatedEnergy\":\"{}\",\"gutEnergy\":\"{}\",\"metabolismRemainder\":\"{}\",\"carriedMass\":\"{}\",\"marker\":{},\"guarded\":{},\"readyAt\":\"{}\",\"pending\":\"{}\",\"lastOutcome\":\"{}\",\"privateMemoryBytes\":{}}}",
        key.0,
        cell.position.0,
        x,
        y,
        cell.core_mass,
        cell.assimilated_energy,
        cell.gut_energy,
        cell.metabolism_remainder,
        cell.carried_material_mass,
        cell.marker,
        cell.guarded,
        cell.ready_at.0,
        pending,
        outcome,
        cell.private_memory.len(),
    )
    .expect("writing JSON to String cannot fail");
}

fn u64_array_json<const N: usize>(values: &[u64; N]) -> String {
    let mut json = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "\"{value}\"").expect("writing JSON to String cannot fail");
    }
    json.push(']');
    json
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::resolution::{
        attestation_key_id, mind_artifact_hash, ActionRequest, MatchVerificationManifest,
        MindArtifactBinding, MindRuntimeProfile, ReferenceCheckpoint, ReferenceRuleset,
        ReferenceSimulation, ReplayCommitment, ReplayManifest, ReplayRecorder,
        ReplaySegmentDescriptor,
    };
    use blob_interface::reference_mind::ReferenceMemoryUpdate;

    struct ReplayFixture {
        manifest: ReplayManifest,
        segment: ReplaySegment,
        event_hash: CanonicalHash,
    }

    fn replay_fixture() -> ReplayFixture {
        let mut simulation = ReferenceSimulation::new(2, 1, ReferenceRuleset::default()).unwrap();
        {
            let plant = simulation
                .tile_state_mut(simulation.tile(1, 0).unwrap())
                .unwrap();
            plant.plant_energy = 1;
            plant.plant_capacity = 10;
            plant.plant_growth_rate = 2;
            plant.diffuse_energy = 4;
        }
        let actor = simulation
            .add_cell(simulation.tile(0, 0).unwrap(), 20, 100, 7)
            .unwrap();
        let compiled_ruleset_hash = simulation.compiled_ruleset_hash();
        let initial_state_hash = simulation.state_hash();
        let checkpoint = ReferenceCheckpoint::from_simulation(&simulation);
        let mut recorder = ReplayRecorder::new(compiled_ruleset_hash, initial_state_hash);
        let start_cursor = recorder.cursor();
        let request = ActionRequest::Wait;
        let started_at = simulation.now();
        let next_private_memory = vec![1, 2, 3];
        let receipt = simulation
            .commit_decision(actor, request.clone(), next_private_memory.clone())
            .unwrap();
        let report = simulation.resolve_next_batch().unwrap();
        let event = recorder
            .record_with_commitments(
                &report,
                vec![ReplayCommitment {
                    actor,
                    request,
                    signal: None,
                    memory_update: ReferenceMemoryUpdate::Replace(next_private_memory),
                    started_at,
                    receipt,
                }],
            )
            .unwrap();
        let event_hash = event.post_state_hash;
        let segment = ReplaySegment::from_events(
            compiled_ruleset_hash,
            start_cursor,
            checkpoint,
            &[event],
            ReplaySegmentLimits::default(),
        )
        .unwrap();
        let manifest = ReplayManifest::from_segments(
            compiled_ruleset_hash,
            initial_state_hash,
            vec![ReplaySegmentDescriptor::from_segment(&segment)],
            ReplayManifestLimits::default(),
        )
        .unwrap();
        ReplayFixture {
            manifest,
            segment,
            event_hash,
        }
    }

    #[test]
    fn browser_replay_loads_a_checkpoint_and_steps_authoritatively() {
        let fixture = replay_fixture();
        let mut replay = BrowserReplay::new(fixture.manifest.to_bytes()).unwrap();
        assert_eq!(replay.segment_count(), 1);
        assert_eq!(replay.event_count(), 1);
        assert_eq!(replay.segment_index_for_event(0).unwrap(), 0);
        assert!(replay
            .segment_descriptor_json(0)
            .unwrap()
            .contains(&fixture.segment.segment_hash().to_hex()));
        assert_eq!(
            replay.manifest_hash(),
            fixture.manifest.manifest_hash().to_hex()
        );

        replay
            .load_segment(0, fixture.segment.to_bytes(), 0, 1)
            .unwrap();
        assert_eq!(replay.width(), Some(2));
        assert_eq!(replay.height(), Some(1));
        assert_eq!(replay.current_sequence(), Some(0));
        assert_eq!(replay.tile_occupants().unwrap().len(), 2);
        assert!(replay.cells_json().unwrap().contains("\"key\":\"0\""));

        let event = replay.step().unwrap();
        assert_eq!(event.sequence(), 0);
        assert_eq!(event.commitment_count(), 1);
        assert_eq!(event.outcome_count(), 1);
        assert_eq!(event.changed_cells(), vec![0]);
        assert_eq!(event.changed_tiles(), vec![0, 1]);
        assert!(replay
            .cells_by_key_json(&event.changed_cells())
            .unwrap()
            .contains("\"lastOutcome\":\"success\""));
        assert!(replay
            .cells_by_key_json(&event.changed_cells())
            .unwrap()
            .contains("\"metabolismRemainder\":\"0\""));
        assert!(replay
            .tiles_json(&event.changed_tiles())
            .unwrap()
            .contains("\"plantEnergy\":\"3\""));
        assert_eq!(event.state_hash(), fixture.event_hash.to_hex());
        assert_eq!(replay.state_hash(), Some(fixture.event_hash.to_hex()));

        let mut seeked = BrowserReplay::new(fixture.manifest.to_bytes()).unwrap();
        seeked
            .load_segment(0, fixture.segment.to_bytes(), 1, 1)
            .unwrap();
        assert_eq!(seeked.current_sequence(), Some(1));
        assert_eq!(seeked.state_hash(), Some(fixture.event_hash.to_hex()));
    }

    #[test]
    fn browser_attestation_exposes_exact_webcrypto_inputs() {
        let fixture = replay_fixture();
        let artifact = MindArtifactBinding {
            slot: 3,
            artifact_hash: mind_artifact_hash(b"fixture-mind.wasm"),
        };
        let manifest = MatchVerificationManifest::from_replay(
            CanonicalHash::from_bytes([0x11; 32]),
            CanonicalHash::from_bytes([0x22; 32]),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            CanonicalHash::from_bytes([0x33; 32]),
            vec![artifact],
            &fixture.manifest,
            MatchVerificationLimits::default(),
        )
        .unwrap();
        let public_key = [0x44; 32];
        let signature = [0x55; 64];
        let portable = MatchAttestation::from_signature(
            manifest.clone(),
            attestation_key_id(&public_key),
            signature,
        );
        let browser = BrowserAttestation::new(portable.to_bytes()).unwrap();

        assert!(browser.matches_public_key(&public_key));
        assert!(!browser.matches_public_key(&[0x45; 32]));
        assert_eq!(browser.signature(), signature);
        assert_eq!(browser.signing_message(), portable.signing_message());
        assert_eq!(browser.manifest_hash(), manifest.manifest_hash().to_hex());
        assert_eq!(
            browser.replay_manifest_hash(),
            fixture.manifest.manifest_hash().to_hex()
        );
        assert_eq!(browser.final_event_count(), 1);
        assert!(browser
            .artifacts_json()
            .contains(&artifact.artifact_hash.to_hex()));
        assert_eq!(
            browser.mind_runtime_profile(),
            "extism_pdk_deterministic_v1"
        );
    }
}
