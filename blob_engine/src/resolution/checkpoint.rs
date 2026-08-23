//! Canonical, bounded checkpoints for the complete reference simulation.

use std::error::Error;
use std::fmt::{Display, Formatter};

use sha2::{Digest, Sha256};

use super::hashing::CanonicalHash;
use super::neighborhood::{
    BoundaryRule, DiagonalCornerRule, LocalOffset, NeighborhoodSpec, ObservationMasks, SlotMask,
    TargetingAction, MAX_LOCAL_SLOTS,
};
use super::reference::{
    DurationRule, EffortProfile, ReferenceRuleset, ReferenceSimulation, ResolutionError, SimTime,
    SimulationState, TimeConfig,
};
use super::replay::{
    decode_cell_state, decode_tile_state, decode_vec, encode_cell_state, encode_tile_state, Reader,
    ReplayError, Writer,
};
use super::CellKey;

pub const CHECKPOINT_FORMAT_VERSION: u16 = 5;
const CHECKPOINT_MAGIC: &[u8; 8] = b"BLBCHK05";
const CHECKPOINT_DOMAIN: &[u8] = b"blob.simulation.checkpoint";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointLimits {
    pub max_checkpoint_bytes: usize,
    pub max_tiles: usize,
    pub max_cells: usize,
    pub max_private_memory_bytes: usize,
}

impl Default for CheckpointLimits {
    fn default() -> Self {
        Self {
            max_checkpoint_bytes: 64 * 1024 * 1024,
            max_tiles: 4_000_000,
            max_cells: 1_000_000,
            max_private_memory_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointError {
    TooLarge {
        actual: usize,
        limit: usize,
    },
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u16),
    IntegrityHashMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    SemanticRulesetHashMismatch,
    CompiledRulesetHashMismatch,
    StateHashMismatch,
    InvalidField(&'static str),
    TrailingBytes(usize),
    Replay(ReplayError),
    Resolution(ResolutionError),
}

impl Display for CheckpointError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { actual, limit } => {
                write!(formatter, "checkpoint is {actual} bytes; limit is {limit}")
            }
            Self::Truncated => write!(formatter, "checkpoint is truncated"),
            Self::InvalidMagic => write!(formatter, "invalid checkpoint magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported checkpoint format version {version}")
            }
            Self::IntegrityHashMismatch { expected, actual } => write!(
                formatter,
                "checkpoint integrity hash mismatch: expected {expected}, got {actual}"
            ),
            Self::SemanticRulesetHashMismatch => {
                write!(formatter, "checkpoint semantic ruleset hash mismatch")
            }
            Self::CompiledRulesetHashMismatch => {
                write!(formatter, "checkpoint compiled ruleset hash mismatch")
            }
            Self::StateHashMismatch => write!(formatter, "checkpoint state hash mismatch"),
            Self::InvalidField(field) => write!(formatter, "invalid checkpoint field: {field}"),
            Self::TrailingBytes(count) => {
                write!(formatter, "checkpoint has {count} trailing body bytes")
            }
            Self::Replay(error) => Display::fmt(error, formatter),
            Self::Resolution(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for CheckpointError {}

impl From<ReplayError> for CheckpointError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<ResolutionError> for CheckpointError {
    fn from(value: ResolutionError) -> Self {
        Self::Resolution(value)
    }
}

/// A validated snapshot of the complete reference state and its exact rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceCheckpoint {
    width: usize,
    height: usize,
    rules: ReferenceRuleset,
    state: SimulationState,
    semantic_ruleset_hash: CanonicalHash,
    compiled_ruleset_hash: CanonicalHash,
    state_hash: CanonicalHash,
    checkpoint_hash: CanonicalHash,
}

impl ReferenceCheckpoint {
    pub fn from_simulation(simulation: &ReferenceSimulation) -> Self {
        let width = simulation.neighborhood().width();
        let height = simulation.neighborhood().height();
        let rules = simulation.rules().clone();
        let state = simulation.canonical_state();
        let semantic_ruleset_hash = simulation.semantic_ruleset_hash();
        let compiled_ruleset_hash = simulation.compiled_ruleset_hash();
        let state_hash = simulation.state_hash();
        let mut checkpoint = Self {
            width,
            height,
            rules,
            state,
            semantic_ruleset_hash,
            compiled_ruleset_hash,
            state_hash,
            checkpoint_hash: CanonicalHash::from_bytes([0; 32]),
        };
        checkpoint.checkpoint_hash = hash_checkpoint_body(&checkpoint.encode_body());
        checkpoint
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CheckpointError> {
        Self::from_bytes_with_limits(bytes, CheckpointLimits::default())
    }

    pub fn from_bytes_with_limits(
        bytes: &[u8],
        limits: CheckpointLimits,
    ) -> Result<Self, CheckpointError> {
        if bytes.len() > limits.max_checkpoint_bytes {
            return Err(CheckpointError::TooLarge {
                actual: bytes.len(),
                limit: limits.max_checkpoint_bytes,
            });
        }
        if bytes.len() < 32 {
            return Err(CheckpointError::Truncated);
        }
        let body_length = bytes.len() - 32;
        let (body, encoded_hash) = bytes.split_at(body_length);
        let expected = CanonicalHash::from_bytes(
            encoded_hash
                .try_into()
                .map_err(|_| CheckpointError::Truncated)?,
        );
        let actual = hash_checkpoint_body(body);
        if actual != expected {
            return Err(CheckpointError::IntegrityHashMismatch { expected, actual });
        }

        let mut reader = Reader::new(body);
        if reader.take(CHECKPOINT_MAGIC.len())? != CHECKPOINT_MAGIC {
            return Err(CheckpointError::InvalidMagic);
        }
        let version = reader.u16()?;
        if version != CHECKPOINT_FORMAT_VERSION {
            return Err(CheckpointError::UnsupportedVersion(version));
        }
        let width = read_usize(&mut reader, "width")?;
        let height = read_usize(&mut reader, "height")?;
        let semantic_ruleset_hash = reader.hash()?;
        let compiled_ruleset_hash = reader.hash()?;
        let state_hash = reader.hash()?;
        let rules = decode_ruleset(&mut reader)?;
        let state = decode_state(&mut reader, limits)?;
        if reader.remaining() != 0 {
            return Err(CheckpointError::TrailingBytes(reader.remaining()));
        }

        let simulation =
            ReferenceSimulation::from_canonical_state(width, height, rules.clone(), state.clone())?;
        if simulation.semantic_ruleset_hash() != semantic_ruleset_hash {
            return Err(CheckpointError::SemanticRulesetHashMismatch);
        }
        if simulation.compiled_ruleset_hash() != compiled_ruleset_hash {
            return Err(CheckpointError::CompiledRulesetHashMismatch);
        }
        if simulation.state_hash() != state_hash {
            return Err(CheckpointError::StateHashMismatch);
        }

        Ok(Self {
            width,
            height,
            rules,
            state,
            semantic_ruleset_hash,
            compiled_ruleset_hash,
            state_hash,
            checkpoint_hash: expected,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.encode_body();
        bytes.extend_from_slice(self.checkpoint_hash.as_bytes());
        bytes
    }

    pub fn into_simulation(self) -> Result<ReferenceSimulation, ResolutionError> {
        ReferenceSimulation::from_canonical_state(self.width, self.height, self.rules, self.state)
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn rules(&self) -> &ReferenceRuleset {
        &self.rules
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    pub const fn semantic_ruleset_hash(&self) -> CanonicalHash {
        self.semantic_ruleset_hash
    }

    pub const fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.compiled_ruleset_hash
    }

    pub const fn state_hash(&self) -> CanonicalHash {
        self.state_hash
    }

    pub const fn checkpoint_hash(&self) -> CanonicalHash {
        self.checkpoint_hash
    }

    fn encode_body(&self) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.raw(CHECKPOINT_MAGIC);
        writer.u16(CHECKPOINT_FORMAT_VERSION);
        writer.u64(self.width as u64);
        writer.u64(self.height as u64);
        writer.hash(self.semantic_ruleset_hash);
        writer.hash(self.compiled_ruleset_hash);
        writer.hash(self.state_hash);
        encode_ruleset(&mut writer, &self.rules);
        encode_state(&mut writer, &self.state);
        writer.finish()
    }
}

fn hash_checkpoint_body(body: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((CHECKPOINT_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(CHECKPOINT_DOMAIN);
    hasher.update(CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    hasher.update((body.len() as u64).to_le_bytes());
    hasher.update(body);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

fn encode_ruleset(writer: &mut Writer, rules: &ReferenceRuleset) {
    encode_neighborhood(writer, &rules.neighborhood);
    writer.u64(rules.time.arithmetic_quanta_per_unit);
    writer.u64(rules.time.completion_bucket);
    writer.u64(rules.time.decision_interval_floor);
    writer.u64(rules.effort_profiles.len() as u64);
    for profile in rules.effort_profiles {
        writer.u32(profile.cost_numerator);
        writer.u32(profile.cost_denominator);
        writer.u32(profile.duration_numerator);
        writer.u32(profile.duration_denominator);
    }
    for duration in [
        rules.wait_duration,
        rules.move_duration,
        rules.attack_duration,
        rules.consume_duration,
        rules.split_duration,
        rules.regurgitate_duration,
        rules.excavate_duration,
        rules.deposit_terrain_duration,
    ] {
        writer.u64(duration.base_quanta);
        writer.u64(duration.mass_quanta_numerator);
        writer.u64(duration.mass_units_denominator);
    }
    for value in [
        rules.move_effort_base,
        rules.move_mass_units_per_effort,
        rules.attack_effort_base,
        rules.guard_effort_base,
        rules.consume_effort_base,
        rules.split_effort_base,
        rules.regurgitate_effort_base,
        rules.excavate_effort_base,
        rules.deposit_terrain_effort_base,
        rules.bite_capacity,
        rules.gut_capacity,
        rules.digestion_rate_numerator,
        rules.digestion_rate_denominator,
        rules.metabolism_rate_numerator,
        rules.metabolism_rate_denominator,
        rules.signal_emission_cost,
        rules.signal_decay_rate_numerator,
        rules.signal_decay_rate_denominator,
        rules.diffusion_interval_quanta,
        rules.diffusion_rate_numerator,
        rules.diffusion_rate_denominator,
        rules.terrain_mass_per_elevation,
    ] {
        writer.u64(value);
    }
    writer.u32(rules.diffusion_targets.bits());
    writer.u64(rules.minimum_survival_energy);
    writer.u64(rules.child_core_mass);
    writer.i16(rules.maximum_elevation_delta);
    for value in [
        rules.guard_damage_numerator,
        rules.guard_damage_denominator,
        rules.attack_payload_damage_numerator,
        rules.attack_payload_damage_denominator,
        rules.attack_mass_damage_numerator,
        rules.attack_mass_damage_denominator,
        rules.apparent_mass_bucket_width,
    ] {
        writer.u64(value);
    }
    writer.u64(rules.max_private_memory_bytes as u64);
}

fn decode_ruleset(reader: &mut Reader<'_>) -> Result<ReferenceRuleset, CheckpointError> {
    let neighborhood = decode_neighborhood(reader)?;
    let time = TimeConfig {
        arithmetic_quanta_per_unit: reader.u64()?,
        completion_bucket: reader.u64()?,
        decision_interval_floor: reader.u64()?,
    };
    if reader.u64()? != 3 {
        return Err(CheckpointError::InvalidField("effort profile count"));
    }
    let mut effort_profiles = [EffortProfile::new(0, 1, 0, 1); 3];
    for profile in &mut effort_profiles {
        *profile = EffortProfile::new(reader.u32()?, reader.u32()?, reader.u32()?, reader.u32()?);
    }
    let mut duration = || -> Result<DurationRule, CheckpointError> {
        Ok(DurationRule::new(
            reader.u64()?,
            reader.u64()?,
            reader.u64()?,
        ))
    };
    let wait_duration = duration()?;
    let move_duration = duration()?;
    let attack_duration = duration()?;
    let consume_duration = duration()?;
    let split_duration = duration()?;
    let regurgitate_duration = duration()?;
    let excavate_duration = duration()?;
    let deposit_terrain_duration = duration()?;

    Ok(ReferenceRuleset {
        neighborhood,
        time,
        effort_profiles,
        wait_duration,
        move_duration,
        attack_duration,
        consume_duration,
        split_duration,
        regurgitate_duration,
        excavate_duration,
        deposit_terrain_duration,
        move_effort_base: reader.u64()?,
        move_mass_units_per_effort: reader.u64()?,
        attack_effort_base: reader.u64()?,
        guard_effort_base: reader.u64()?,
        consume_effort_base: reader.u64()?,
        split_effort_base: reader.u64()?,
        regurgitate_effort_base: reader.u64()?,
        excavate_effort_base: reader.u64()?,
        deposit_terrain_effort_base: reader.u64()?,
        bite_capacity: reader.u64()?,
        gut_capacity: reader.u64()?,
        digestion_rate_numerator: reader.u64()?,
        digestion_rate_denominator: reader.u64()?,
        metabolism_rate_numerator: reader.u64()?,
        metabolism_rate_denominator: reader.u64()?,
        signal_emission_cost: reader.u64()?,
        signal_decay_rate_numerator: reader.u64()?,
        signal_decay_rate_denominator: reader.u64()?,
        diffusion_interval_quanta: reader.u64()?,
        diffusion_rate_numerator: reader.u64()?,
        diffusion_rate_denominator: reader.u64()?,
        terrain_mass_per_elevation: reader.u64()?,
        diffusion_targets: slot_mask_from_bits(reader.u32()?),
        minimum_survival_energy: reader.u64()?,
        child_core_mass: reader.u64()?,
        maximum_elevation_delta: reader.i16()?,
        guard_damage_numerator: reader.u64()?,
        guard_damage_denominator: reader.u64()?,
        attack_payload_damage_numerator: reader.u64()?,
        attack_payload_damage_denominator: reader.u64()?,
        attack_mass_damage_numerator: reader.u64()?,
        attack_mass_damage_denominator: reader.u64()?,
        apparent_mass_bucket_width: reader.u64()?,
        max_private_memory_bytes: read_usize(reader, "maximum private memory")?,
    })
}

fn encode_neighborhood(writer: &mut Writer, neighborhood: &NeighborhoodSpec) {
    writer.u64(neighborhood.slots.len() as u64);
    for offset in &neighborhood.slots {
        writer.i8(offset.dx);
        writer.i8(offset.dy);
        writer.u16(offset.distance_cost_q10);
    }
    for mask in [
        neighborhood.observations.occupancy,
        neighborhood.observations.marker,
        neighborhood.observations.apparent_mass,
        neighborhood.observations.activity,
        neighborhood.observations.terrain,
        neighborhood.observations.energy,
        neighborhood.observations.signal,
        neighborhood.target_mask(TargetingAction::Move),
        neighborhood.target_mask(TargetingAction::Attack),
        neighborhood.target_mask(TargetingAction::Split),
        neighborhood.target_mask(TargetingAction::Regurgitate),
    ] {
        writer.u32(mask.bits());
    }
    writer.u8(match neighborhood.diagonal_corner_rule {
        DiagonalCornerRule::Allow => 0,
        DiagonalCornerRule::BlockIfEitherOrthogonalOccupied => 1,
        DiagonalCornerRule::BlockIfBothOrthogonalsOccupied => 2,
    });
    writer.u8(match neighborhood.boundary_rule {
        BoundaryRule::Bounded => 0,
        BoundaryRule::Wrap => 1,
    });
    writer.u8(neighborhood.max_radius);
}

fn decode_neighborhood(reader: &mut Reader<'_>) -> Result<NeighborhoodSpec, CheckpointError> {
    let slot_count = reader.u64()?;
    if slot_count > MAX_LOCAL_SLOTS as u64 {
        return Err(CheckpointError::InvalidField("neighborhood slot count"));
    }
    let mut slots = Vec::with_capacity(slot_count as usize);
    for _ in 0..slot_count {
        slots.push(LocalOffset::new(reader.i8()?, reader.i8()?, reader.u16()?));
    }
    let masks: Vec<SlotMask> = (0..11)
        .map(|_| reader.u32().map(slot_mask_from_bits))
        .collect::<Result<_, _>>()?;
    let diagonal_corner_rule = match reader.u8()? {
        0 => DiagonalCornerRule::Allow,
        1 => DiagonalCornerRule::BlockIfEitherOrthogonalOccupied,
        2 => DiagonalCornerRule::BlockIfBothOrthogonalsOccupied,
        _ => return Err(CheckpointError::InvalidField("diagonal corner rule")),
    };
    let boundary_rule = match reader.u8()? {
        0 => BoundaryRule::Bounded,
        1 => BoundaryRule::Wrap,
        _ => return Err(CheckpointError::InvalidField("boundary rule")),
    };
    Ok(NeighborhoodSpec::new(
        slots,
        ObservationMasks {
            occupancy: masks[0],
            marker: masks[1],
            apparent_mass: masks[2],
            activity: masks[3],
            terrain: masks[4],
            energy: masks[5],
            signal: masks[6],
        },
        [masks[7], masks[8], masks[9], masks[10]],
        diagonal_corner_rule,
        boundary_rule,
        reader.u8()?,
    ))
}

fn slot_mask_from_bits(bits: u32) -> SlotMask {
    SlotMask::from_slots(
        (0..MAX_LOCAL_SLOTS)
            .filter(|slot| bits & (1_u32 << slot) != 0)
            .map(|slot| super::LocalSlot(slot as u8)),
    )
}

fn encode_state(writer: &mut Writer, state: &SimulationState) {
    writer.u64(state.now.0);
    writer.u64(state.next_cell_key);
    writer.vec(&state.tiles, encode_tile_state);
    writer.vec(&state.cells, |writer, (key, cell)| {
        writer.u64(key.0);
        encode_cell_state(writer, cell);
    });
}

fn decode_state(
    reader: &mut Reader<'_>,
    limits: CheckpointLimits,
) -> Result<SimulationState, CheckpointError> {
    let now = SimTime(reader.u64()?);
    let next_cell_key = reader.u64()?;
    let tiles = decode_vec(
        reader,
        "checkpoint tiles",
        limits.max_tiles,
        decode_tile_state,
    )?;
    let cells = decode_vec(reader, "checkpoint cells", limits.max_cells, |reader| {
        Ok((
            CellKey(reader.u64()?),
            decode_cell_state(reader, limits.max_private_memory_bytes)?,
        ))
    })?;
    Ok(SimulationState {
        now,
        tiles,
        cells,
        next_cell_key,
    })
}

fn read_usize(reader: &mut Reader<'_>, field: &'static str) -> Result<usize, CheckpointError> {
    usize::try_from(reader.u64()?).map_err(|_| CheckpointError::InvalidField(field))
}
