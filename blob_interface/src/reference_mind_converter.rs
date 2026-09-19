//! Bounded Cap'n Proto conversion for the canonical Mind ABI.

use capnp::message::ReaderOptions;
use capnp::serialize;

use crate::randomness::{PRIVATE_RANDOM_BYTES, PrivateRandom};
use crate::reference_mind::{
    CurrentTileObservation, EFFORT_BURST_BIT, EFFORT_GENTLE_BIT, EFFORT_STANDARD_BIT,
    LOCAL_DIFFUSE_ENERGY_BIT, LOCAL_ELEVATION_BIT, LOCAL_LOOSE_ENERGY_BIT,
    LOCAL_NEIGHBOR_ACTIVITY_BIT, LOCAL_NEIGHBOR_BIT, LOCAL_NEIGHBOR_MARKER_BIT,
    LOCAL_NEIGHBOR_MASS_BIT, LOCAL_NEIGHBOR_PROGRESS_BIT, LOCAL_PLANT_CAPACITY_BIT,
    LOCAL_PLANT_ENERGY_BIT, LOCAL_PLANT_GROWTH_RATE_BIT, LOCAL_SIGNAL_ENERGY_BIT,
    LOCAL_VISIBILITY_BITS, LocalObservation, REFERENCE_MAX_LOCAL_SLOTS, REFERENCE_SIGNAL_CHANNELS,
    ReferenceActionSpace, ReferenceActivity, ReferenceEffort, ReferenceMemoryUpdate,
    ReferenceMindAction, ReferenceMindDecision, ReferenceMindInput, ReferenceNeighbor,
    ReferenceOutcome, ReferenceOutcomeStatus, ReferenceProgress, ReferenceRejectReason,
    ReferenceSelfState, ReferenceSignalEmission,
};
use crate::reference_mind_capnp as wire;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceMindLimits {
    pub max_input_bytes: usize,
    pub max_action_bytes: usize,
    pub max_slots: usize,
    pub max_private_memory_bytes: usize,
}

impl Default for ReferenceMindLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 4 * 1024 * 1024,
            max_action_bytes: 1024 * 1024,
            max_slots: REFERENCE_MAX_LOCAL_SLOTS,
            max_private_memory_bytes: 64 * 1024,
        }
    }
}

pub fn reference_mind_input_to_capnp(
    input: &ReferenceMindInput,
    limits: ReferenceMindLimits,
) -> capnp::Result<Vec<u8>> {
    validate_reference_mind_input(input, limits)?;
    let mut message = capnp::message::Builder::new_default();
    let mut root = message.init_root::<wire::reference_mind_input::Builder>();
    {
        let state = &input.self_state;
        let mut builder = root.reborrow().init_self_state();
        builder.set_core_mass(state.core_mass);
        builder.set_assimilated_energy(state.assimilated_energy);
        builder.set_gut_energy(state.gut_energy);
        builder.set_metabolism_remainder(state.metabolism_remainder);
        builder.set_carried_material_mass(state.carried_material_mass);
        builder.set_marker(state.marker);
        builder.set_guarded(state.guarded);
        set_optional_outcome(builder.init_last_outcome(), state.last_outcome);
    }
    {
        let tile = input.current_tile;
        let mut builder = root.reborrow().init_current_tile();
        builder.set_elevation(tile.elevation);
        builder.set_plant_energy(tile.plant_energy);
        builder.set_plant_capacity(tile.plant_capacity);
        builder.set_plant_growth_rate(tile.plant_growth_rate);
        builder.set_loose_energy(tile.loose_energy);
        builder.set_diffuse_energy(tile.diffuse_energy);
        set_signal_energy(
            builder.init_signal_energy(REFERENCE_SIGNAL_CHANNELS as u32),
            tile.signal_energy,
        );
    }
    {
        let mut slots = root.reborrow().init_slots(input.slots.len() as u32);
        for (index, observation) in input.slots.iter().enumerate() {
            set_local_observation(slots.reborrow().get(index as u32), observation);
        }
    }
    {
        let space = input.action_space;
        let mut builder = root.reborrow().init_action_space();
        builder.set_wait_enabled(space.wait_enabled);
        builder.set_guard_enabled(space.guard_enabled);
        builder.set_consume_enabled(space.consume_enabled);
        builder.set_move_targets(space.move_targets);
        builder.set_attack_targets(space.attack_targets);
        builder.set_split_targets(space.split_targets);
        builder.set_regurgitate_targets(space.regurgitate_targets);
        builder.set_effort_mask(space.effort_mask);
        builder.set_max_consume_amount(space.max_consume_amount);
        builder.set_gut_capacity(space.gut_capacity);
        builder.set_max_private_memory_bytes(space.max_private_memory_bytes);
        builder.set_minimum_survival_energy(space.minimum_survival_energy);
        builder.set_child_core_mass(space.child_core_mass);
        builder.set_metabolism_rate_numerator(space.metabolism_rate_numerator);
        builder.set_metabolism_rate_denominator(space.metabolism_rate_denominator);
        builder.set_excavate_enabled(space.excavate_enabled);
        builder.set_deposit_terrain_enabled(space.deposit_terrain_enabled);
        builder.set_signal_enabled(space.signal_enabled);
        builder.set_terrain_mass_per_elevation(space.terrain_mass_per_elevation);
        builder.set_signal_emission_cost(space.signal_emission_cost);
        {
            let mut values = builder.reborrow().init_effort_cost_numerators(3);
            for (index, value) in space.effort_cost_numerators.into_iter().enumerate() {
                values.set(index as u32, value);
            }
        }
        {
            let mut values = builder.reborrow().init_effort_cost_denominators(3);
            for (index, value) in space.effort_cost_denominators.into_iter().enumerate() {
                values.set(index as u32, value);
            }
        }
        builder.set_move_effort_base(space.move_effort_base);
        builder.set_move_mass_units_per_effort(space.move_mass_units_per_effort);
        builder.set_attack_effort_base(space.attack_effort_base);
        builder.set_guard_effort_base(space.guard_effort_base);
        builder.set_consume_effort_base(space.consume_effort_base);
        builder.set_split_effort_base(space.split_effort_base);
        builder.set_regurgitate_effort_base(space.regurgitate_effort_base);
        builder.set_excavate_effort_base(space.excavate_effort_base);
        builder.set_deposit_terrain_effort_base(space.deposit_terrain_effort_base);
    }
    root.set_private_memory(&input.private_memory);
    root.set_randomness(input.randomness.as_bytes());

    // Flatten once into its exact final capacity instead of repeatedly growing
    // a fresh Vec through the generic Write implementation.
    let output = serialize::write_message_to_words(&message);
    if output.len() > limits.max_input_bytes {
        return Err(failed(format!(
            "reference Mind input is {} bytes; limit is {}",
            output.len(),
            limits.max_input_bytes
        )));
    }
    Ok(output)
}

fn set_local_observation(
    mut builder: wire::local_observation::Builder,
    observation: &LocalObservation,
) {
    builder.set_slot(observation.slot);
    builder.set_dx(observation.dx);
    builder.set_dy(observation.dy);
    builder.set_distance_cost_q10(observation.distance_cost_q10);
    builder.set_reachable(observation.reachable);
    let mut visibility = 0;
    if let Some(value) = observation.elevation {
        visibility |= LOCAL_ELEVATION_BIT;
        builder.set_elevation(value);
    }
    if let Some(value) = observation.plant_energy {
        visibility |= LOCAL_PLANT_ENERGY_BIT;
        builder.set_plant_energy(value);
    }
    if let Some(value) = observation.plant_capacity {
        visibility |= LOCAL_PLANT_CAPACITY_BIT;
        builder.set_plant_capacity(value);
    }
    if let Some(value) = observation.plant_growth_rate {
        visibility |= LOCAL_PLANT_GROWTH_RATE_BIT;
        builder.set_plant_growth_rate(value);
    }
    if let Some(value) = observation.loose_energy {
        visibility |= LOCAL_LOOSE_ENERGY_BIT;
        builder.set_loose_energy(value);
    }
    if let Some(value) = observation.diffuse_energy {
        visibility |= LOCAL_DIFFUSE_ENERGY_BIT;
        builder.set_diffuse_energy(value);
    }
    if let Some(value) = observation.signal_energy {
        visibility |= LOCAL_SIGNAL_ENERGY_BIT;
        builder.set_signal_energy0(value[0]);
        builder.set_signal_energy1(value[1]);
        builder.set_signal_energy2(value[2]);
        builder.set_signal_energy3(value[3]);
    }
    if let Some(neighbor) = observation.neighbor {
        visibility |= LOCAL_NEIGHBOR_BIT;
        if let Some(value) = neighbor.marker {
            visibility |= LOCAL_NEIGHBOR_MARKER_BIT;
            builder.set_neighbor_marker(value);
        }
        if let Some(value) = neighbor.apparent_mass_bucket {
            visibility |= LOCAL_NEIGHBOR_MASS_BIT;
            builder.set_neighbor_apparent_mass_bucket(value);
        }
        if let Some(value) = neighbor.activity {
            visibility |= LOCAL_NEIGHBOR_ACTIVITY_BIT;
            builder.set_neighbor_activity(to_wire_activity(value));
        }
        if let Some(value) = neighbor.progress {
            visibility |= LOCAL_NEIGHBOR_PROGRESS_BIT;
            builder.set_neighbor_progress(to_wire_progress(value));
        }
    }
    builder.set_visibility(visibility);
}

pub fn capnp_to_reference_mind_input(
    bytes: &[u8],
    limits: ReferenceMindLimits,
) -> capnp::Result<ReferenceMindInput> {
    check_bytes(bytes.len(), limits.max_input_bytes, "reference Mind input")?;
    let message = serialize::read_message(bytes, reader_options(limits.max_input_bytes))?;
    let root = message.get_root::<wire::reference_mind_input::Reader>()?;
    let state = root.get_self_state()?;
    let self_state = ReferenceSelfState {
        core_mass: state.get_core_mass(),
        assimilated_energy: state.get_assimilated_energy(),
        gut_energy: state.get_gut_energy(),
        metabolism_remainder: state.get_metabolism_remainder(),
        carried_material_mass: state.get_carried_material_mass(),
        marker: state.get_marker(),
        guarded: state.get_guarded(),
        last_outcome: read_optional_outcome(state.get_last_outcome()?)?,
    };
    let tile = root.get_current_tile()?;
    let current_tile = CurrentTileObservation {
        elevation: tile.get_elevation(),
        plant_energy: tile.get_plant_energy(),
        plant_capacity: tile.get_plant_capacity(),
        plant_growth_rate: tile.get_plant_growth_rate(),
        loose_energy: tile.get_loose_energy(),
        diffuse_energy: tile.get_diffuse_energy(),
        signal_energy: read_signal_energy(tile.get_signal_energy()?, false)?
            .ok_or_else(|| failed("reference Mind current signal field is missing"))?,
    };

    let slot_readers = root.get_slots()?;
    if slot_readers.len() as usize > limits.max_slots {
        return Err(failed("reference Mind input has too many local slots"));
    }
    let mut slots = Vec::with_capacity(slot_readers.len() as usize);
    for index in 0..slot_readers.len() {
        let slot = read_local_observation(slot_readers.get(index))?;
        if slot.slot != index as u8 {
            return Err(failed("reference Mind slots are not canonically ordered"));
        }
        slots.push(slot);
    }

    let action = root.get_action_space()?;
    let effort_numerator_reader = action.get_effort_cost_numerators()?;
    let effort_denominator_reader = action.get_effort_cost_denominators()?;
    if effort_numerator_reader.len() != 3 || effort_denominator_reader.len() != 3 {
        return Err(failed(
            "reference Mind effort cost profiles have the wrong length",
        ));
    }
    let effort_cost_numerators =
        std::array::from_fn(|index| effort_numerator_reader.get(index as u32));
    let effort_cost_denominators =
        std::array::from_fn(|index| effort_denominator_reader.get(index as u32));
    let action_space = ReferenceActionSpace {
        wait_enabled: action.get_wait_enabled(),
        guard_enabled: action.get_guard_enabled(),
        consume_enabled: action.get_consume_enabled(),
        move_targets: action.get_move_targets(),
        attack_targets: action.get_attack_targets(),
        split_targets: action.get_split_targets(),
        regurgitate_targets: action.get_regurgitate_targets(),
        effort_mask: action.get_effort_mask(),
        max_consume_amount: action.get_max_consume_amount(),
        gut_capacity: action.get_gut_capacity(),
        max_private_memory_bytes: action.get_max_private_memory_bytes(),
        minimum_survival_energy: action.get_minimum_survival_energy(),
        child_core_mass: action.get_child_core_mass(),
        metabolism_rate_numerator: action.get_metabolism_rate_numerator(),
        metabolism_rate_denominator: action.get_metabolism_rate_denominator(),
        excavate_enabled: action.get_excavate_enabled(),
        deposit_terrain_enabled: action.get_deposit_terrain_enabled(),
        signal_enabled: action.get_signal_enabled(),
        terrain_mass_per_elevation: action.get_terrain_mass_per_elevation(),
        signal_emission_cost: action.get_signal_emission_cost(),
        effort_cost_numerators,
        effort_cost_denominators,
        move_effort_base: action.get_move_effort_base(),
        move_mass_units_per_effort: action.get_move_mass_units_per_effort(),
        attack_effort_base: action.get_attack_effort_base(),
        guard_effort_base: action.get_guard_effort_base(),
        consume_effort_base: action.get_consume_effort_base(),
        split_effort_base: action.get_split_effort_base(),
        regurgitate_effort_base: action.get_regurgitate_effort_base(),
        excavate_effort_base: action.get_excavate_effort_base(),
        deposit_terrain_effort_base: action.get_deposit_terrain_effort_base(),
    };
    let private_memory = root.get_private_memory()?.to_vec();
    if private_memory.len() > limits.max_private_memory_bytes {
        return Err(failed("reference Mind private memory exceeds its limit"));
    }
    let randomness_reader = root.get_randomness()?;
    if randomness_reader.len() != PRIVATE_RANDOM_BYTES {
        return Err(failed(format!(
            "reference Mind randomness must be {PRIVATE_RANDOM_BYTES} bytes"
        )));
    }
    let mut randomness = [0_u8; PRIVATE_RANDOM_BYTES];
    randomness.copy_from_slice(randomness_reader);
    let input = ReferenceMindInput {
        self_state,
        current_tile,
        slots,
        action_space,
        private_memory,
        randomness: PrivateRandom::from_bytes(randomness),
    };
    validate_reference_mind_input(&input, limits)?;
    Ok(input)
}

fn read_local_observation(
    slot: wire::local_observation::Reader,
) -> capnp::Result<LocalObservation> {
    let visibility = slot.get_visibility();
    if visibility & !LOCAL_VISIBILITY_BITS != 0 {
        return Err(failed(
            "reference Mind local observation has unknown visibility bits",
        ));
    }
    let neighbor_details = LOCAL_NEIGHBOR_MARKER_BIT
        | LOCAL_NEIGHBOR_MASS_BIT
        | LOCAL_NEIGHBOR_ACTIVITY_BIT
        | LOCAL_NEIGHBOR_PROGRESS_BIT;
    if visibility & LOCAL_NEIGHBOR_BIT == 0 && visibility & neighbor_details != 0 {
        return Err(failed(
            "reference Mind hidden neighbor has visible detail fields",
        ));
    }
    if visibility & LOCAL_NEIGHBOR_PROGRESS_BIT != 0
        && visibility & LOCAL_NEIGHBOR_ACTIVITY_BIT == 0
    {
        return Err(failed("neighbor progress requires an activity cue"));
    }
    let signal_energy = [
        slot.get_signal_energy0(),
        slot.get_signal_energy1(),
        slot.get_signal_energy2(),
        slot.get_signal_energy3(),
    ];
    let activity = from_wire_activity(
        slot.get_neighbor_activity()
            .map_err(|_| failed("unknown activity value"))?,
    );
    let progress = from_wire_progress(
        slot.get_neighbor_progress()
            .map_err(|_| failed("unknown progress value"))?,
    );
    let hidden_nonzero = (visibility & LOCAL_ELEVATION_BIT == 0 && slot.get_elevation() != 0)
        || (visibility & LOCAL_PLANT_ENERGY_BIT == 0 && slot.get_plant_energy() != 0)
        || (visibility & LOCAL_PLANT_CAPACITY_BIT == 0 && slot.get_plant_capacity() != 0)
        || (visibility & LOCAL_PLANT_GROWTH_RATE_BIT == 0 && slot.get_plant_growth_rate() != 0)
        || (visibility & LOCAL_LOOSE_ENERGY_BIT == 0 && slot.get_loose_energy() != 0)
        || (visibility & LOCAL_DIFFUSE_ENERGY_BIT == 0 && slot.get_diffuse_energy() != 0)
        || (visibility & LOCAL_SIGNAL_ENERGY_BIT == 0 && signal_energy != [0; 4])
        || (visibility & LOCAL_NEIGHBOR_MARKER_BIT == 0 && slot.get_neighbor_marker() != 0)
        || (visibility & LOCAL_NEIGHBOR_MASS_BIT == 0
            && slot.get_neighbor_apparent_mass_bucket() != 0)
        || (visibility & LOCAL_NEIGHBOR_ACTIVITY_BIT == 0 && activity != ReferenceActivity::Ready)
        || (visibility & LOCAL_NEIGHBOR_PROGRESS_BIT == 0 && progress != ReferenceProgress::Early);
    if hidden_nonzero {
        return Err(failed(
            "reference Mind hidden local observation value is nonzero",
        ));
    }
    Ok(LocalObservation {
        slot: slot.get_slot(),
        dx: slot.get_dx(),
        dy: slot.get_dy(),
        distance_cost_q10: slot.get_distance_cost_q10(),
        reachable: slot.get_reachable(),
        elevation: visible_value(visibility, LOCAL_ELEVATION_BIT, slot.get_elevation()),
        plant_energy: visible_value(visibility, LOCAL_PLANT_ENERGY_BIT, slot.get_plant_energy()),
        plant_capacity: visible_value(
            visibility,
            LOCAL_PLANT_CAPACITY_BIT,
            slot.get_plant_capacity(),
        ),
        plant_growth_rate: visible_value(
            visibility,
            LOCAL_PLANT_GROWTH_RATE_BIT,
            slot.get_plant_growth_rate(),
        ),
        loose_energy: visible_value(visibility, LOCAL_LOOSE_ENERGY_BIT, slot.get_loose_energy()),
        diffuse_energy: visible_value(
            visibility,
            LOCAL_DIFFUSE_ENERGY_BIT,
            slot.get_diffuse_energy(),
        ),
        signal_energy: visible_value(visibility, LOCAL_SIGNAL_ENERGY_BIT, signal_energy),
        neighbor: (visibility & LOCAL_NEIGHBOR_BIT != 0).then_some(ReferenceNeighbor {
            marker: visible_value(
                visibility,
                LOCAL_NEIGHBOR_MARKER_BIT,
                slot.get_neighbor_marker(),
            ),
            apparent_mass_bucket: visible_value(
                visibility,
                LOCAL_NEIGHBOR_MASS_BIT,
                slot.get_neighbor_apparent_mass_bucket(),
            ),
            activity: visible_value(visibility, LOCAL_NEIGHBOR_ACTIVITY_BIT, activity),
            progress: visible_value(visibility, LOCAL_NEIGHBOR_PROGRESS_BIT, progress),
        }),
    })
}

fn visible_value<T>(visibility: u16, bit: u16, value: T) -> Option<T> {
    (visibility & bit != 0).then_some(value)
}

pub fn reference_mind_decision_to_capnp(
    decision: &ReferenceMindDecision,
    limits: ReferenceMindLimits,
) -> capnp::Result<Vec<u8>> {
    if let ReferenceMemoryUpdate::Replace(bytes) = &decision.memory_update
        && bytes.len() > limits.max_private_memory_bytes
    {
        return Err(failed(
            "reference Mind next private memory exceeds its limit",
        ));
    }
    if let ReferenceMindAction::Split { private_memory, .. } = &decision.action
        && private_memory.len() > limits.max_private_memory_bytes
    {
        return Err(failed(
            "reference Mind action private memory exceeds its limit",
        ));
    }
    if decision
        .signal
        .is_some_and(|signal| usize::from(signal.channel) >= REFERENCE_SIGNAL_CHANNELS)
    {
        return Err(failed("reference Mind signal channel is outside its range"));
    }
    if decision.signal.is_some_and(|signal| signal.amount == 0) {
        return Err(failed("reference Mind signal amount must be positive"));
    }
    if matches!(decision.action, ReferenceMindAction::Signal { .. }) && decision.signal.is_some() {
        return Err(failed(
            "reference Mind explicit signal action cannot include a sidecar signal",
        ));
    }
    if matches!(
        &decision.action,
        ReferenceMindAction::Signal { amounts } if amounts.iter().all(|amount| *amount == 0)
    ) {
        return Err(failed(
            "reference Mind explicit signal action must deposit energy",
        ));
    }
    let mut message = capnp::message::Builder::new_default();
    let mut root = message.init_root::<wire::reference_mind_decision::Builder>();
    set_reference_action(root.reborrow().init_action(), &decision.action);
    match &decision.memory_update {
        ReferenceMemoryUpdate::Retain => {
            root.set_retain_private_memory(true);
            root.set_next_private_memory(&[]);
        }
        ReferenceMemoryUpdate::Replace(bytes) => {
            root.set_retain_private_memory(false);
            root.set_next_private_memory(bytes);
        }
    }
    {
        let mut signal = root.reborrow().init_signal();
        match decision.signal {
            Some(value) => {
                let mut emission = signal.init_some();
                emission.set_channel(value.channel);
                emission.set_amount(value.amount);
            }
            None => signal.set_none(()),
        }
    }
    let output = serialize::write_message_to_words(&message);
    check_bytes(
        output.len(),
        limits.max_action_bytes,
        "reference Mind decision",
    )?;
    Ok(output)
}

fn set_reference_action(mut root: wire::reference_action::Builder, action: &ReferenceMindAction) {
    match action {
        ReferenceMindAction::Wait => root.set_wait(()),
        ReferenceMindAction::Move {
            target_slot,
            effort,
        } => {
            let mut value = root.init_move();
            value.set_target_slot(*target_slot);
            value.set_effort(to_wire_effort(*effort));
        }
        ReferenceMindAction::Attack {
            target_slot,
            effort,
            payload,
        } => {
            let mut value = root.init_attack();
            value.set_target_slot(*target_slot);
            value.set_effort(to_wire_effort(*effort));
            value.set_payload(*payload);
        }
        ReferenceMindAction::Guard { effort } => root.set_guard(to_wire_effort(*effort)),
        ReferenceMindAction::Consume { amount } => root.set_consume(*amount),
        ReferenceMindAction::Split {
            target_slot,
            child_allocation,
            marker,
            private_memory,
        } => {
            let mut value = root.init_split();
            value.set_target_slot(*target_slot);
            value.set_child_allocation(*child_allocation);
            value.set_marker(*marker);
            value.set_private_memory(private_memory);
        }
        ReferenceMindAction::Regurgitate {
            target_slot,
            amount,
        } => {
            let mut value = root.init_regurgitate();
            value.set_target_slot(*target_slot);
            value.set_amount(*amount);
        }
        ReferenceMindAction::Signal { amounts } => {
            let mut value = root.init_signal();
            value.set_amount0(amounts[0]);
            value.set_amount1(amounts[1]);
            value.set_amount2(amounts[2]);
            value.set_amount3(amounts[3]);
        }
        ReferenceMindAction::Excavate => root.set_excavate(()),
        ReferenceMindAction::DepositTerrain => root.set_deposit_terrain(()),
    }
}

pub fn capnp_to_reference_mind_decision(
    bytes: &[u8],
    limits: ReferenceMindLimits,
) -> capnp::Result<ReferenceMindDecision> {
    check_bytes(
        bytes.len(),
        limits.max_action_bytes,
        "reference Mind decision",
    )?;
    let message = serialize::read_message(bytes, reader_options(limits.max_action_bytes))?;
    let root = message.get_root::<wire::reference_mind_decision::Reader>()?;
    let action = read_reference_action(root.get_action()?, limits)?;
    let signal = match root
        .get_signal()?
        .which()
        .map_err(|_| failed("unknown optional signal emission variant"))?
    {
        wire::optional_signal_emission::Which::None(()) => None,
        wire::optional_signal_emission::Which::Some(value) => {
            let value = value?;
            Some(ReferenceSignalEmission {
                channel: value.get_channel(),
                amount: value.get_amount(),
            })
        }
    };
    if signal.is_some_and(|emission| {
        usize::from(emission.channel) >= REFERENCE_SIGNAL_CHANNELS || emission.amount == 0
    }) {
        return Err(failed("reference Mind signal emission is invalid"));
    }
    if matches!(&action, ReferenceMindAction::Signal { amounts } if amounts.iter().all(|amount| *amount == 0))
        || matches!(&action, ReferenceMindAction::Signal { .. }) && signal.is_some()
    {
        return Err(failed("reference Mind explicit signal action is invalid"));
    }
    let next_private_memory = root.get_next_private_memory()?;
    let memory_update = if root.get_retain_private_memory() {
        if !next_private_memory.is_empty() {
            return Err(failed(
                "retained private memory must not include replacement bytes",
            ));
        }
        ReferenceMemoryUpdate::Retain
    } else {
        ReferenceMemoryUpdate::Replace(next_private_memory.to_vec())
    };
    if memory_update
        .replacement()
        .is_some_and(|bytes| bytes.len() > limits.max_private_memory_bytes)
    {
        return Err(failed(
            "reference Mind next private memory exceeds its limit",
        ));
    }
    Ok(ReferenceMindDecision {
        action,
        signal,
        memory_update,
    })
}

fn read_reference_action(
    root: wire::reference_action::Reader,
    limits: ReferenceMindLimits,
) -> capnp::Result<ReferenceMindAction> {
    use wire::reference_action::Which;
    match root
        .which()
        .map_err(|_| failed("unknown reference Mind action variant"))?
    {
        Which::Wait(()) => Ok(ReferenceMindAction::Wait),
        Which::Move(value) => {
            let value = value?;
            Ok(ReferenceMindAction::Move {
                target_slot: value.get_target_slot(),
                effort: from_wire_effort(value.get_effort()?)?,
            })
        }
        Which::Attack(value) => {
            let value = value?;
            Ok(ReferenceMindAction::Attack {
                target_slot: value.get_target_slot(),
                effort: from_wire_effort(value.get_effort()?)?,
                payload: value.get_payload(),
            })
        }
        Which::Guard(effort) => Ok(ReferenceMindAction::Guard {
            effort: from_wire_effort(
                effort.map_err(|_| failed("unknown reference Mind effort tier"))?,
            )?,
        }),
        Which::Consume(amount) => Ok(ReferenceMindAction::Consume { amount }),
        Which::Split(value) => {
            let value = value?;
            let private_memory = value.get_private_memory()?.to_vec();
            if private_memory.len() > limits.max_private_memory_bytes {
                return Err(failed(
                    "reference Mind action private memory exceeds its limit",
                ));
            }
            Ok(ReferenceMindAction::Split {
                target_slot: value.get_target_slot(),
                child_allocation: value.get_child_allocation(),
                marker: value.get_marker(),
                private_memory,
            })
        }
        Which::Regurgitate(value) => {
            let value = value?;
            Ok(ReferenceMindAction::Regurgitate {
                target_slot: value.get_target_slot(),
                amount: value.get_amount(),
            })
        }
        Which::Excavate(()) => Ok(ReferenceMindAction::Excavate),
        Which::DepositTerrain(()) => Ok(ReferenceMindAction::DepositTerrain),
        Which::Signal(value) => {
            let value = value?;
            Ok(ReferenceMindAction::Signal {
                amounts: [
                    value.get_amount0(),
                    value.get_amount1(),
                    value.get_amount2(),
                    value.get_amount3(),
                ],
            })
        }
    }
}

fn set_signal_energy(
    mut builder: capnp::primitive_list::Builder<'_, u64>,
    values: [u64; REFERENCE_SIGNAL_CHANNELS],
) {
    for (index, value) in values.into_iter().enumerate() {
        builder.set(index as u32, value);
    }
}

fn read_signal_energy(
    reader: capnp::primitive_list::Reader<'_, u64>,
    optional: bool,
) -> capnp::Result<Option<[u64; REFERENCE_SIGNAL_CHANNELS]>> {
    if optional && reader.is_empty() {
        return Ok(None);
    }
    if reader.len() as usize != REFERENCE_SIGNAL_CHANNELS {
        return Err(failed(
            "reference Mind signal field has the wrong channel count",
        ));
    }
    let mut values = [0; REFERENCE_SIGNAL_CHANNELS];
    for (index, value) in values.iter_mut().enumerate() {
        *value = reader.get(index as u32);
    }
    Ok(Some(values))
}

/// Validate an in-memory input with the same bounds and scalar rules as the wire ABI.
pub fn validate_reference_mind_input(
    input: &ReferenceMindInput,
    limits: ReferenceMindLimits,
) -> capnp::Result<()> {
    if input.slots.len() > limits.max_slots || input.slots.len() > REFERENCE_MAX_LOCAL_SLOTS {
        return Err(failed("reference Mind input has too many local slots"));
    }
    if !input
        .slots
        .iter()
        .enumerate()
        .all(|(index, slot)| usize::from(slot.slot) == index)
    {
        return Err(failed("reference Mind slots are not canonically ordered"));
    }
    if let Some(outcome) = input.self_state.last_outcome
        && (outcome.status == ReferenceOutcomeStatus::Rejected) != outcome.rejected_reason.is_some()
    {
        return Err(failed("rejected outcome and rejection reason disagree"));
    }
    if input
        .slots
        .iter()
        .filter_map(|slot| slot.neighbor)
        .any(|neighbor| neighbor.progress.is_some() && neighbor.activity.is_none())
    {
        return Err(failed("neighbor progress requires an activity cue"));
    }
    validate_input_scalars(
        &input.self_state,
        input.slots.len(),
        &input.action_space,
        input.private_memory.len(),
        limits,
    )
}

fn validate_input_scalars(
    self_state: &ReferenceSelfState,
    slot_count: usize,
    action_space: &ReferenceActionSpace,
    private_memory_len: usize,
    limits: ReferenceMindLimits,
) -> capnp::Result<()> {
    if slot_count > limits.max_slots || slot_count > REFERENCE_MAX_LOCAL_SLOTS {
        return Err(failed("reference Mind input has too many local slots"));
    }
    if let Some(outcome) = self_state.last_outcome
        && (outcome.status == ReferenceOutcomeStatus::Rejected) != outcome.rejected_reason.is_some()
    {
        return Err(failed("rejected outcome and rejection reason disagree"));
    }
    if action_space.max_private_memory_bytes as usize > limits.max_private_memory_bytes {
        return Err(failed(
            "reference Mind action space exceeds the private memory limit",
        ));
    }
    if action_space.metabolism_rate_denominator == 0 {
        return Err(failed(
            "reference Mind metabolism rate denominator must be nonzero",
        ));
    }
    if action_space.effort_cost_denominators.contains(&0)
        || action_space.move_mass_units_per_effort == 0
    {
        return Err(failed(
            "reference Mind action cost denominators must be nonzero",
        ));
    }
    if action_space.terrain_mass_per_elevation == 0 {
        return Err(failed(
            "reference Mind terrain mass conversion must be nonzero",
        ));
    }
    if action_space.signal_enabled != (action_space.signal_emission_cost > 0) {
        return Err(failed(
            "reference Mind signal capability and emission cost disagree",
        ));
    }
    if self_state.metabolism_remainder >= action_space.metabolism_rate_denominator {
        return Err(failed(
            "reference Mind metabolism remainder exceeds its denominator",
        ));
    }
    if self_state.assimilated_energy == 0 && self_state.metabolism_remainder != 0 {
        return Err(failed(
            "reference Mind cell without assimilated energy has a metabolism remainder",
        ));
    }
    let valid_targets = if slot_count == REFERENCE_MAX_LOCAL_SLOTS {
        u32::MAX
    } else {
        (1_u32 << slot_count) - 1
    };
    let target_union = action_space.move_targets
        | action_space.attack_targets
        | action_space.split_targets
        | action_space.regurgitate_targets;
    if target_union & !valid_targets != 0 {
        return Err(failed("reference Mind action space names an absent slot"));
    }
    let valid_efforts = EFFORT_GENTLE_BIT | EFFORT_STANDARD_BIT | EFFORT_BURST_BIT;
    if action_space.effort_mask & !valid_efforts != 0 {
        return Err(failed(
            "reference Mind action space has an unknown effort bit",
        ));
    }
    if private_memory_len > limits.max_private_memory_bytes {
        return Err(failed("reference Mind private memory exceeds its limit"));
    }
    Ok(())
}

fn reader_options(max_bytes: usize) -> ReaderOptions {
    ReaderOptions {
        traversal_limit_in_words: Some(max_bytes.div_ceil(8).saturating_mul(4)),
        nesting_limit: 32,
    }
}

fn check_bytes(actual: usize, limit: usize, kind: &str) -> capnp::Result<()> {
    if actual > limit {
        Err(failed(format!(
            "{kind} is {actual} bytes; limit is {limit}"
        )))
    } else {
        Ok(())
    }
}

fn failed(message: impl Into<String>) -> capnp::Error {
    capnp::Error::failed(message.into())
}

fn set_optional_outcome(
    mut builder: wire::optional_outcome::Builder,
    value: Option<ReferenceOutcome>,
) {
    let Some(value) = value else {
        builder.set_none(());
        return;
    };
    let mut outcome = builder.init_some();
    outcome.set_status(to_wire_outcome_status(value.status));
    let mut rejection = outcome.init_rejected_reason();
    match value.rejected_reason {
        Some(value) => rejection.set_some(to_wire_reject_reason(value)),
        None => rejection.set_none(()),
    }
}

fn read_optional_outcome(
    reader: wire::optional_outcome::Reader,
) -> capnp::Result<Option<ReferenceOutcome>> {
    match reader
        .which()
        .map_err(|_| failed("unknown optional outcome variant"))?
    {
        wire::optional_outcome::Which::None(()) => Ok(None),
        wire::optional_outcome::Which::Some(value) => {
            let value = value?;
            let rejected_reason = match value
                .get_rejected_reason()?
                .which()
                .map_err(|_| failed("unknown optional rejection variant"))?
            {
                wire::optional_reject_reason::Which::None(()) => None,
                wire::optional_reject_reason::Which::Some(value) => Some(from_wire_reject_reason(
                    value.map_err(|_| failed("unknown rejection value"))?,
                )),
            };
            let status = from_wire_outcome_status(value.get_status()?)?;
            if (status == ReferenceOutcomeStatus::Rejected) != rejected_reason.is_some() {
                return Err(failed("rejected outcome and rejection reason disagree"));
            }
            Ok(Some(ReferenceOutcome {
                status,
                rejected_reason,
            }))
        }
    }
}

fn to_wire_effort(value: ReferenceEffort) -> wire::EffortTier {
    match value {
        ReferenceEffort::Gentle => wire::EffortTier::Gentle,
        ReferenceEffort::Standard => wire::EffortTier::Standard,
        ReferenceEffort::Burst => wire::EffortTier::Burst,
    }
}

fn from_wire_effort(value: wire::EffortTier) -> capnp::Result<ReferenceEffort> {
    Ok(match value {
        wire::EffortTier::Gentle => ReferenceEffort::Gentle,
        wire::EffortTier::Standard => ReferenceEffort::Standard,
        wire::EffortTier::Burst => ReferenceEffort::Burst,
    })
}

fn to_wire_activity(value: ReferenceActivity) -> wire::Activity {
    match value {
        ReferenceActivity::Ready => wire::Activity::Ready,
        ReferenceActivity::Moving => wire::Activity::Moving,
        ReferenceActivity::AttackWindup => wire::Activity::AttackWindup,
        ReferenceActivity::Guarding => wire::Activity::Guarding,
        ReferenceActivity::Feeding => wire::Activity::Feeding,
        ReferenceActivity::Splitting => wire::Activity::Splitting,
        ReferenceActivity::ManipulatingTerrain => wire::Activity::ManipulatingTerrain,
        ReferenceActivity::OtherBusy => wire::Activity::OtherBusy,
    }
}

fn from_wire_activity(value: wire::Activity) -> ReferenceActivity {
    match value {
        wire::Activity::Ready => ReferenceActivity::Ready,
        wire::Activity::Moving => ReferenceActivity::Moving,
        wire::Activity::AttackWindup => ReferenceActivity::AttackWindup,
        wire::Activity::Guarding => ReferenceActivity::Guarding,
        wire::Activity::Feeding => ReferenceActivity::Feeding,
        wire::Activity::Splitting => ReferenceActivity::Splitting,
        wire::Activity::ManipulatingTerrain => ReferenceActivity::ManipulatingTerrain,
        wire::Activity::OtherBusy => ReferenceActivity::OtherBusy,
    }
}

fn to_wire_progress(value: ReferenceProgress) -> wire::Progress {
    match value {
        ReferenceProgress::Early => wire::Progress::Early,
        ReferenceProgress::Middle => wire::Progress::Middle,
        ReferenceProgress::Late => wire::Progress::Late,
    }
}

fn from_wire_progress(value: wire::Progress) -> ReferenceProgress {
    match value {
        wire::Progress::Early => ReferenceProgress::Early,
        wire::Progress::Middle => ReferenceProgress::Middle,
        wire::Progress::Late => ReferenceProgress::Late,
    }
}

fn to_wire_outcome_status(value: ReferenceOutcomeStatus) -> wire::OutcomeStatus {
    match value {
        ReferenceOutcomeStatus::Success => wire::OutcomeStatus::Success,
        ReferenceOutcomeStatus::Frustrated => wire::OutcomeStatus::Frustrated,
        ReferenceOutcomeStatus::Contested => wire::OutcomeStatus::Contested,
        ReferenceOutcomeStatus::Interrupted => wire::OutcomeStatus::Interrupted,
        ReferenceOutcomeStatus::Rejected => wire::OutcomeStatus::Rejected,
    }
}

fn from_wire_outcome_status(value: wire::OutcomeStatus) -> capnp::Result<ReferenceOutcomeStatus> {
    Ok(match value {
        wire::OutcomeStatus::Success => ReferenceOutcomeStatus::Success,
        wire::OutcomeStatus::Frustrated => ReferenceOutcomeStatus::Frustrated,
        wire::OutcomeStatus::Contested => ReferenceOutcomeStatus::Contested,
        wire::OutcomeStatus::Interrupted => ReferenceOutcomeStatus::Interrupted,
        wire::OutcomeStatus::Rejected => ReferenceOutcomeStatus::Rejected,
    })
}

fn to_wire_reject_reason(value: ReferenceRejectReason) -> wire::RejectReason {
    match value {
        ReferenceRejectReason::InvalidSlot => wire::RejectReason::InvalidSlot,
        ReferenceRejectReason::ActionNotAllowedInSlot => wire::RejectReason::ActionNotAllowedInSlot,
        ReferenceRejectReason::TargetOutsideWorld => wire::RejectReason::TargetOutsideWorld,
        ReferenceRejectReason::TargetsSelf => wire::RejectReason::TargetsSelf,
        ReferenceRejectReason::ZeroPayload => wire::RejectReason::ZeroPayload,
        ReferenceRejectReason::InsufficientGutEnergy => wire::RejectReason::InsufficientGutEnergy,
        ReferenceRejectReason::ChildAllocationTooSmall => {
            wire::RejectReason::ChildAllocationTooSmall
        }
        ReferenceRejectReason::PrivateMemoryTooLarge => wire::RejectReason::PrivateMemoryTooLarge,
        ReferenceRejectReason::InsufficientEnergy => wire::RejectReason::InsufficientEnergy,
        ReferenceRejectReason::InsufficientMaterial => wire::RejectReason::InsufficientMaterial,
        ReferenceRejectReason::TerrainLimit => wire::RejectReason::TerrainLimit,
        ReferenceRejectReason::ArithmeticOverflow => wire::RejectReason::ArithmeticOverflow,
    }
}

fn from_wire_reject_reason(value: wire::RejectReason) -> ReferenceRejectReason {
    match value {
        wire::RejectReason::InvalidSlot => ReferenceRejectReason::InvalidSlot,
        wire::RejectReason::ActionNotAllowedInSlot => ReferenceRejectReason::ActionNotAllowedInSlot,
        wire::RejectReason::TargetOutsideWorld => ReferenceRejectReason::TargetOutsideWorld,
        wire::RejectReason::TargetsSelf => ReferenceRejectReason::TargetsSelf,
        wire::RejectReason::ZeroPayload => ReferenceRejectReason::ZeroPayload,
        wire::RejectReason::InsufficientGutEnergy => ReferenceRejectReason::InsufficientGutEnergy,
        wire::RejectReason::ChildAllocationTooSmall => {
            ReferenceRejectReason::ChildAllocationTooSmall
        }
        wire::RejectReason::PrivateMemoryTooLarge => ReferenceRejectReason::PrivateMemoryTooLarge,
        wire::RejectReason::InsufficientEnergy => ReferenceRejectReason::InsufficientEnergy,
        wire::RejectReason::InsufficientMaterial => ReferenceRejectReason::InsufficientMaterial,
        wire::RejectReason::TerrainLimit => ReferenceRejectReason::TerrainLimit,
        wire::RejectReason::ArithmeticOverflow => ReferenceRejectReason::ArithmeticOverflow,
    }
}
