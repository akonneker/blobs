//! Fixed-shape policy catalog that decodes directly to reference actions.
//!
//! Target slots are reserved up to the ABI maximum so one model shape can be
//! reused across neighborhood rulesets. Per-observation masks prevent the
//! policy from sampling disabled efforts, actions, or unreachable slots.

use blob_interface::reference_mind::{
    ReferenceActionSpace, ReferenceEffort, ReferenceMemoryUpdate, ReferenceMindAction,
    ReferenceMindDecision, ReferenceMindInput, ReferenceSignalEmission, REFERENCE_MAX_LOCAL_SLOTS,
};

const EFFORT_COUNT: usize = 3;
const WAIT: usize = 0;
const GUARD_START: usize = 1;
const CONSUME: usize = GUARD_START + EFFORT_COUNT;
const MOVE_START: usize = CONSUME + 1;
const ATTACK_START: usize = MOVE_START + REFERENCE_MAX_LOCAL_SLOTS * EFFORT_COUNT;
const SPLIT_START: usize = ATTACK_START + REFERENCE_MAX_LOCAL_SLOTS * EFFORT_COUNT;
const REGURGITATE_START: usize = SPLIT_START + REFERENCE_MAX_LOCAL_SLOTS;
const EXCAVATE: usize = REGURGITATE_START + REFERENCE_MAX_LOCAL_SLOTS;
const DEPOSIT_TERRAIN: usize = EXCAVATE + 1;
const SIGNAL: usize = DEPOSIT_TERRAIN + 1;
pub const NUM_ACTIONS: usize = SIGNAL + 1;
pub const NUM_AMOUNT_CHOICES: usize = 5;
/// Four-bit channel patterns, including zero for no sidecar emission.
pub const NUM_SIGNAL_CHOICES: usize = 16;
pub const NUM_SIGNAL_STRENGTH_CHOICES: usize = 5;
pub const NUM_POLICY_ACTION_KINDS: usize = 10;
pub const NUM_POLICY_TARGETS: usize = REFERENCE_MAX_LOCAL_SLOTS;
pub const NUM_POLICY_EFFORTS: usize = EFFORT_COUNT;
pub const NUM_POLICY_TARGET_LOGITS: usize = NUM_POLICY_ACTION_KINDS * NUM_POLICY_TARGETS;
pub const NUM_POLICY_EFFORT_LOGITS: usize = NUM_POLICY_ACTION_KINDS * NUM_POLICY_EFFORTS;
pub const NUM_POLICY_AMOUNT_LOGITS: usize = NUM_POLICY_ACTION_KINDS * NUM_AMOUNT_CHOICES;

pub const fn policy_wait_action() -> usize {
    WAIT
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum PolicyActionKind {
    Wait,
    Guard,
    Consume,
    Move,
    Attack,
    Split,
    Regurgitate,
    Excavate,
    DepositTerrain,
    Signal,
}

impl PolicyActionKind {
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Wait),
            1 => Some(Self::Guard),
            2 => Some(Self::Consume),
            3 => Some(Self::Move),
            4 => Some(Self::Attack),
            5 => Some(Self::Split),
            6 => Some(Self::Regurgitate),
            7 => Some(Self::Excavate),
            8 => Some(Self::DepositTerrain),
            9 => Some(Self::Signal),
            _ => None,
        }
    }

    pub const fn uses_target(self) -> bool {
        matches!(
            self,
            Self::Move | Self::Attack | Self::Split | Self::Regurgitate
        )
    }

    pub const fn uses_effort(self) -> bool {
        matches!(self, Self::Guard | Self::Move | Self::Attack)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HierarchicalActionChoice {
    pub kind: usize,
    pub target: usize,
    pub effort: usize,
}

pub const fn decompose_policy_action(action: usize) -> Option<HierarchicalActionChoice> {
    let (kind, target, effort) = match action {
        WAIT => (PolicyActionKind::Wait, 0, 0),
        GUARD_START..CONSUME => (PolicyActionKind::Guard, 0, action - GUARD_START),
        CONSUME => (PolicyActionKind::Consume, 0, 0),
        MOVE_START..ATTACK_START => {
            let offset = action - MOVE_START;
            (
                PolicyActionKind::Move,
                offset / EFFORT_COUNT,
                offset % EFFORT_COUNT,
            )
        }
        ATTACK_START..SPLIT_START => {
            let offset = action - ATTACK_START;
            (
                PolicyActionKind::Attack,
                offset / EFFORT_COUNT,
                offset % EFFORT_COUNT,
            )
        }
        SPLIT_START..REGURGITATE_START => (PolicyActionKind::Split, action - SPLIT_START, 0),
        REGURGITATE_START..EXCAVATE => {
            (PolicyActionKind::Regurgitate, action - REGURGITATE_START, 0)
        }
        EXCAVATE => (PolicyActionKind::Excavate, 0, 0),
        DEPOSIT_TERRAIN => (PolicyActionKind::DepositTerrain, 0, 0),
        SIGNAL => (PolicyActionKind::Signal, 0, 0),
        _ => return None,
    };
    Some(HierarchicalActionChoice {
        kind: kind.index(),
        target,
        effort,
    })
}

pub const fn compose_policy_action(choice: HierarchicalActionChoice) -> Option<usize> {
    let kind = match PolicyActionKind::from_index(choice.kind) {
        Some(kind) => kind,
        None => return None,
    };
    if choice.target >= NUM_POLICY_TARGETS || choice.effort >= NUM_POLICY_EFFORTS {
        return None;
    }
    match kind {
        PolicyActionKind::Wait if choice.target == 0 && choice.effort == 0 => Some(WAIT),
        PolicyActionKind::Guard if choice.target == 0 => Some(GUARD_START + choice.effort),
        PolicyActionKind::Consume if choice.target == 0 && choice.effort == 0 => Some(CONSUME),
        PolicyActionKind::Move => Some(MOVE_START + choice.target * EFFORT_COUNT + choice.effort),
        PolicyActionKind::Attack => {
            Some(ATTACK_START + choice.target * EFFORT_COUNT + choice.effort)
        }
        PolicyActionKind::Split if choice.effort == 0 => Some(SPLIT_START + choice.target),
        PolicyActionKind::Regurgitate if choice.effort == 0 => {
            Some(REGURGITATE_START + choice.target)
        }
        PolicyActionKind::Excavate if choice.target == 0 && choice.effort == 0 => Some(EXCAVATE),
        PolicyActionKind::DepositTerrain if choice.target == 0 && choice.effort == 0 => {
            Some(DEPOSIT_TERRAIN)
        }
        PolicyActionKind::Signal if choice.target == 0 && choice.effort == 0 => Some(SIGNAL),
        _ => None,
    }
}

pub fn policy_action_kind_mask(action_mask: &[bool]) -> [bool; NUM_POLICY_ACTION_KINDS] {
    let allowed = |action| action_mask.get(action).copied().unwrap_or(false);
    let any_allowed = |start, end| (start..end).any(&allowed);
    [
        allowed(WAIT),
        any_allowed(GUARD_START, CONSUME),
        allowed(CONSUME),
        any_allowed(MOVE_START, ATTACK_START),
        any_allowed(ATTACK_START, SPLIT_START),
        any_allowed(SPLIT_START, REGURGITATE_START),
        any_allowed(REGURGITATE_START, EXCAVATE),
        allowed(EXCAVATE),
        allowed(DEPOSIT_TERRAIN),
        allowed(SIGNAL),
    ]
}

pub fn policy_target_mask(action_mask: &[bool], kind: usize) -> [bool; NUM_POLICY_TARGETS] {
    let Some(kind_value) = PolicyActionKind::from_index(kind) else {
        return [false; NUM_POLICY_TARGETS];
    };
    if !kind_value.uses_target() {
        let mut mask = [false; NUM_POLICY_TARGETS];
        mask[0] = true;
        return mask;
    }
    let allowed = |action| action_mask.get(action).copied().unwrap_or(false);
    std::array::from_fn(|target| match kind_value {
        PolicyActionKind::Move => {
            (0..EFFORT_COUNT).any(|effort| allowed(MOVE_START + target * EFFORT_COUNT + effort))
        }
        PolicyActionKind::Attack => {
            (0..EFFORT_COUNT).any(|effort| allowed(ATTACK_START + target * EFFORT_COUNT + effort))
        }
        PolicyActionKind::Split => allowed(SPLIT_START + target),
        PolicyActionKind::Regurgitate => allowed(REGURGITATE_START + target),
        _ => unreachable!("non-target kinds returned above"),
    })
}

pub fn policy_effort_mask(
    action_mask: &[bool],
    kind: usize,
    target: usize,
) -> [bool; NUM_POLICY_EFFORTS] {
    let Some(kind_value) = PolicyActionKind::from_index(kind) else {
        return [false; NUM_POLICY_EFFORTS];
    };
    if !kind_value.uses_effort() {
        let mut mask = [false; NUM_POLICY_EFFORTS];
        mask[0] = true;
        return mask;
    }
    let allowed = |action| action_mask.get(action).copied().unwrap_or(false);
    let start = match kind_value {
        PolicyActionKind::Guard => GUARD_START,
        PolicyActionKind::Move => MOVE_START + target * EFFORT_COUNT,
        PolicyActionKind::Attack => ATTACK_START + target * EFFORT_COUNT,
        _ => unreachable!("non-effort kinds returned above"),
    };
    std::array::from_fn(|effort| allowed(start + effort))
}

/// Coarse physical family used for diagnostics and supervised class
/// balancing. Target slots and effort variants remain distinct policy labels,
/// but should not each receive the weight of an independent behavior family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum PolicyActionFamily {
    Wait,
    Guard,
    Consume,
    Move,
    Attack,
    Split,
    Regurgitate,
    Terrain,
    Signal,
}

impl PolicyActionFamily {
    pub const COUNT: usize = 9;
    pub const ALL: [Self; Self::COUNT] = [
        Self::Wait,
        Self::Guard,
        Self::Consume,
        Self::Move,
        Self::Attack,
        Self::Split,
        Self::Regurgitate,
        Self::Terrain,
        Self::Signal,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Wait => "wait",
            Self::Guard => "guard",
            Self::Consume => "consume",
            Self::Move => "move",
            Self::Attack => "attack",
            Self::Split => "split",
            Self::Regurgitate => "regurgitate",
            Self::Terrain => "terrain",
            Self::Signal => "signal",
        }
    }
}

pub const fn policy_action_family(action: usize) -> Option<PolicyActionFamily> {
    match action {
        WAIT => Some(PolicyActionFamily::Wait),
        GUARD_START..CONSUME => Some(PolicyActionFamily::Guard),
        CONSUME => Some(PolicyActionFamily::Consume),
        MOVE_START..ATTACK_START => Some(PolicyActionFamily::Move),
        ATTACK_START..SPLIT_START => Some(PolicyActionFamily::Attack),
        SPLIT_START..REGURGITATE_START => Some(PolicyActionFamily::Split),
        REGURGITATE_START..EXCAVATE => Some(PolicyActionFamily::Regurgitate),
        EXCAVATE..SIGNAL => Some(PolicyActionFamily::Terrain),
        SIGNAL => Some(PolicyActionFamily::Signal),
        _ => None,
    }
}

/// A row-separable policy choice. Signal selection is deliberately factored
/// away from the physical catalog so future multi-channel patterns do not
/// multiply every movement, target, and effort choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyChoice {
    pub action: usize,
    /// Conditional payload/amount tier. Non-parameterized actions use zero.
    pub amount: usize,
    /// Four-bit anonymous channel pattern. Zero means no sidecar emission.
    pub signal: usize,
    /// Conditional quantized signal strength. When `signal` is zero only the
    /// canonical zero index is legal and contributes no policy entropy.
    pub signal_strength: usize,
}

#[derive(Debug, Clone)]
pub struct PolicyMasks {
    pub actions: [bool; NUM_ACTIONS],
    pub amount_choice_bits: [u8; NUM_ACTIONS],
    /// Five signal-strength bits packed for each of five action-amount tiers.
    pub sidecar_strength_bits: [u32; NUM_ACTIONS],
    /// Explicit Signal strength bits indexed by channel-count minus one.
    pub explicit_signal_strength_bits: [u8; 4],
}

const SIGNAL_STRENGTH_MULTIPLIERS: [u64; NUM_SIGNAL_STRENGTH_CHOICES] = [1, 2, 4, 8, 16];

fn signal_strength_amount(quantum: u64, choice: usize) -> Option<u64> {
    quantum.checked_mul(*SIGNAL_STRENGTH_MULTIPLIERS.get(choice)?)
}

fn signal_strength_bits(maximum_quanta_per_channel: u64) -> u8 {
    SIGNAL_STRENGTH_MULTIPLIERS
        .iter()
        .enumerate()
        .fold(0_u8, |bits, (choice, multiplier)| {
            bits | (u8::from(*multiplier <= maximum_quanta_per_channel) << choice)
        })
}

fn amount_value(maximum: u64, choice: usize) -> Option<u64> {
    match choice {
        0 => Some(1.min(maximum)),
        1 => multiply_ratio_ceil(u128::from(maximum), 1, 8),
        2 => multiply_ratio_ceil(u128::from(maximum), 1, 4),
        3 => multiply_ratio_ceil(u128::from(maximum), 1, 2),
        4 => Some(maximum),
        _ => None,
    }
    .filter(|amount| *amount > 0)
}

fn split_bounds(input: &ReferenceMindInput) -> Option<(u64, u64)> {
    let minimum = input
        .action_space
        .child_core_mass
        .checked_add(input.action_space.minimum_survival_energy)?;
    let maximum = input
        .self_state
        .assimilated_energy
        .checked_sub(input.action_space.minimum_survival_energy)?
        .checked_sub(fixed_effort_cost(
            input,
            input.action_space.split_effort_base,
            ReferenceEffort::Standard,
        ))?;
    (maximum >= minimum).then_some((minimum, maximum))
}

fn split_amount(input: &ReferenceMindInput, choice: usize) -> Option<u64> {
    let (minimum, maximum) = split_bounds(input)?;
    let extra = maximum - minimum;
    let selected_extra = match choice {
        0 => 0,
        1 => multiply_ratio_ceil(u128::from(extra), 1, 8)?,
        2 => multiply_ratio_ceil(u128::from(extra), 1, 4)?,
        3 => multiply_ratio_ceil(u128::from(extra), 1, 2)?,
        4 => extra,
        _ => return None,
    };
    minimum.checked_add(selected_extra)
}

fn parameterized_action(action_id: usize) -> bool {
    action_id == CONSUME
        || (ATTACK_START..SPLIT_START).contains(&action_id)
        || (SPLIT_START..REGURGITATE_START).contains(&action_id)
        || (REGURGITATE_START..EXCAVATE).contains(&action_id)
}

/// Commit a learned private state to the acting cell and, for a split, seed
/// the child with the same state. The bytes remain private to those two Mind
/// instances and are never added to an observation.
pub fn attach_policy_memory(decision: &mut ReferenceMindDecision, memory: Vec<u8>) {
    if let ReferenceMindAction::Split { private_memory, .. } = &mut decision.action {
        private_memory.clone_from(&memory);
    }
    decision.memory_update = ReferenceMemoryUpdate::Replace(memory);
}

fn effort(index: usize) -> ReferenceEffort {
    match index {
        0 => ReferenceEffort::Gentle,
        1 => ReferenceEffort::Standard,
        _ => ReferenceEffort::Burst,
    }
}

fn effort_index(effort: ReferenceEffort) -> usize {
    match effort {
        ReferenceEffort::Gentle => 0,
        ReferenceEffort::Standard => 1,
        ReferenceEffort::Burst => 2,
    }
}

fn target_is_reachable(input: &ReferenceMindInput, slot: u8) -> bool {
    input
        .slots
        .get(slot as usize)
        .is_some_and(|observation| observation.slot == slot && observation.reachable)
}

fn multiply_ratio_ceil(value: u128, numerator: u64, denominator: u64) -> Option<u64> {
    if denominator == 0 {
        return None;
    }
    let product = value.checked_mul(u128::from(numerator))?;
    let rounded =
        product.checked_add(u128::from(denominator).saturating_sub(1))? / u128::from(denominator);
    u64::try_from(rounded).ok()
}

fn effort_cost(input: &ReferenceMindInput, raw: u64, effort: ReferenceEffort) -> Option<u64> {
    multiply_ratio_ceil(
        u128::from(raw),
        u64::from(input.action_space.effort_cost_numerators[effort.index()]),
        u64::from(input.action_space.effort_cost_denominators[effort.index()]),
    )
}

fn move_effort_cost(input: &ReferenceMindInput, slot: u8, effort: ReferenceEffort) -> Option<u64> {
    let distance = input.slots.get(usize::from(slot))?.distance_cost_q10;
    let mass = u128::from(input.self_state.core_mass)
        .checked_add(u128::from(input.self_state.assimilated_energy))?
        .checked_add(u128::from(input.self_state.gut_energy))?
        .checked_add(u128::from(input.self_state.carried_material_mass))?;
    let denominator = input
        .action_space
        .move_mass_units_per_effort
        .checked_mul(1024)?;
    let inertial = multiply_ratio_ceil(mass, u64::from(distance), denominator)?;
    let raw = input.action_space.move_effort_base.checked_add(inertial)?;
    effort_cost(input, raw, effort)
}

fn fixed_effort_cost(input: &ReferenceMindInput, raw: u64, effort: ReferenceEffort) -> u64 {
    effort_cost(input, raw, effort).unwrap_or(u64::MAX)
}

fn affordable(
    input: &ReferenceMindInput,
    effort_cost: u64,
    payload: u64,
    signal_cost: u64,
) -> bool {
    effort_cost
        .checked_add(payload)
        .and_then(|required| required.checked_add(signal_cost))
        .and_then(|required| required.checked_add(input.action_space.minimum_survival_energy))
        .is_some_and(|required| input.self_state.assimilated_energy >= required)
}

fn split_allocation(input: &ReferenceMindInput) -> Option<u64> {
    let minimum_child = input
        .action_space
        .child_core_mass
        .checked_add(input.action_space.minimum_survival_energy)?;
    let expendable = input
        .self_state
        .assimilated_energy
        .checked_sub(input.action_space.minimum_survival_energy)?
        .checked_sub(fixed_effort_cost(
            input,
            input.action_space.split_effort_base,
            ReferenceEffort::Standard,
        ))?;
    let allocation = (expendable / 3).max(minimum_child);
    (allocation <= expendable).then_some(allocation)
}

fn attack_max_payload(input: &ReferenceMindInput, effort: ReferenceEffort) -> Option<u64> {
    input
        .self_state
        .assimilated_energy
        .checked_sub(input.action_space.minimum_survival_energy)?
        .checked_sub(fixed_effort_cost(
            input,
            input.action_space.attack_effort_base,
            effort,
        ))
        .filter(|expendable| *expendable > 0)
}

fn attack_payload(input: &ReferenceMindInput, effort: ReferenceEffort) -> Option<u64> {
    amount_value(attack_max_payload(input, effort)?, 1)
}

/// Checks only information available at the anonymous decision frontier. A
/// `true` result guarantees that the resolver's deterministic commit planner
/// will not reject the action for its shape, target, payload, or energy cost.
pub(crate) fn action_is_commit_legal(
    input: &ReferenceMindInput,
    action: &ReferenceMindAction,
    emits_signal: bool,
) -> bool {
    let signal_cost = if emits_signal {
        if !input.action_space.signal_enabled {
            return false;
        }
        input.action_space.signal_emission_cost
    } else {
        0
    };
    let (enabled, effort_cost, payload) = match action {
        ReferenceMindAction::Wait => (input.action_space.wait_enabled, 0, 0),
        ReferenceMindAction::Guard { effort } => (
            input.action_space.guard_enabled && input.action_space.supports_effort(*effort),
            fixed_effort_cost(input, input.action_space.guard_effort_base, *effort),
            0,
        ),
        ReferenceMindAction::Consume { amount } => (
            input.action_space.consume_enabled
                && *amount > 0
                && *amount <= input.action_space.max_consume_amount,
            fixed_effort_cost(
                input,
                input.action_space.consume_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        ),
        ReferenceMindAction::Move {
            target_slot,
            effort,
        } => (
            target_is_reachable(input, *target_slot)
                && input.action_space.supports_effort(*effort)
                && ReferenceActionSpace::allows_target(
                    input.action_space.move_targets,
                    *target_slot,
                ),
            move_effort_cost(input, *target_slot, *effort).unwrap_or(u64::MAX),
            0,
        ),
        ReferenceMindAction::Attack {
            target_slot,
            effort,
            payload,
        } => (
            *payload > 0
                && target_is_reachable(input, *target_slot)
                && input.action_space.supports_effort(*effort)
                && ReferenceActionSpace::allows_target(
                    input.action_space.attack_targets,
                    *target_slot,
                ),
            fixed_effort_cost(input, input.action_space.attack_effort_base, *effort),
            *payload,
        ),
        ReferenceMindAction::Split {
            target_slot,
            child_allocation,
            private_memory,
            ..
        } => {
            let minimum_child = input
                .action_space
                .child_core_mass
                .checked_add(input.action_space.minimum_survival_energy);
            (
                minimum_child.is_some_and(|minimum| *child_allocation >= minimum)
                    && private_memory.len() <= input.action_space.max_private_memory_bytes as usize
                    && target_is_reachable(input, *target_slot)
                    && ReferenceActionSpace::allows_target(
                        input.action_space.split_targets,
                        *target_slot,
                    ),
                fixed_effort_cost(
                    input,
                    input.action_space.split_effort_base,
                    ReferenceEffort::Standard,
                ),
                *child_allocation,
            )
        }
        ReferenceMindAction::Regurgitate {
            target_slot,
            amount,
        } => (
            *amount > 0
                && *amount <= input.self_state.gut_energy
                && target_is_reachable(input, *target_slot)
                && ReferenceActionSpace::allows_target(
                    input.action_space.regurgitate_targets,
                    *target_slot,
                ),
            fixed_effort_cost(
                input,
                input.action_space.regurgitate_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        ),
        ReferenceMindAction::Signal { amounts } => {
            let total = amounts
                .iter()
                .try_fold(0_u64, |total, amount| total.checked_add(*amount));
            let quantum = input.action_space.signal_emission_cost;
            (
                !emits_signal
                    && input.action_space.signal_enabled
                    && total.is_some_and(|total| total > 0)
                    && amounts
                        .iter()
                        .all(|amount| *amount == 0 || quantum > 0 && *amount % quantum == 0),
                0,
                total.unwrap_or(u64::MAX),
            )
        }
        ReferenceMindAction::Excavate => (
            input.action_space.excavate_enabled,
            fixed_effort_cost(
                input,
                input.action_space.excavate_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        ),
        ReferenceMindAction::DepositTerrain => (
            input.action_space.deposit_terrain_enabled
                && input.self_state.carried_material_mass
                    >= input.action_space.terrain_mass_per_elevation,
            fixed_effort_cost(
                input,
                input.action_space.deposit_terrain_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        ),
    };
    enabled && affordable(input, effort_cost, payload, signal_cost)
}

/// Produces the sampling mask for one canonical decision frontier.
pub fn action_mask(input: &ReferenceMindInput) -> [bool; NUM_ACTIONS] {
    let mut physical = [false; NUM_ACTIONS];
    let mut set = |index: usize, enabled: bool, effort_cost: u64, payload: u64| {
        physical[index] = enabled && affordable(input, effort_cost, payload, 0);
    };
    set(WAIT, input.action_space.wait_enabled, 0, 0);
    for effort_index in 0..EFFORT_COUNT {
        let index = GUARD_START + effort_index;
        let selected_effort = effort(effort_index);
        set(
            index,
            input.action_space.guard_enabled && input.action_space.supports_effort(selected_effort),
            fixed_effort_cost(input, input.action_space.guard_effort_base, selected_effort),
            0,
        );
    }
    set(
        CONSUME,
        input.action_space.consume_enabled && input.action_space.max_consume_amount > 0,
        fixed_effort_cost(
            input,
            input.action_space.consume_effort_base,
            ReferenceEffort::Standard,
        ),
        0,
    );

    for slot in 0..REFERENCE_MAX_LOCAL_SLOTS {
        let target_slot = slot as u8;
        let reachable = target_is_reachable(input, target_slot);
        for effort_index in 0..EFFORT_COUNT {
            let selected_effort = effort(effort_index);
            let supported = input.action_space.supports_effort(selected_effort);
            let move_index = MOVE_START + slot * EFFORT_COUNT + effort_index;
            set(
                move_index,
                reachable
                    && supported
                    && ReferenceActionSpace::allows_target(
                        input.action_space.move_targets,
                        target_slot,
                    ),
                move_effort_cost(input, target_slot, selected_effort).unwrap_or(u64::MAX),
                0,
            );
            let attack_index = ATTACK_START + slot * EFFORT_COUNT + effort_index;
            let payload = attack_payload(input, selected_effort).unwrap_or(0);
            set(
                attack_index,
                payload > 0
                    && reachable
                    && supported
                    && ReferenceActionSpace::allows_target(
                        input.action_space.attack_targets,
                        target_slot,
                    ),
                fixed_effort_cost(
                    input,
                    input.action_space.attack_effort_base,
                    selected_effort,
                ),
                payload,
            );
        }
        let split_index = SPLIT_START + slot;
        let allocation = split_allocation(input).unwrap_or(0);
        set(
            split_index,
            allocation > 0
                && reachable
                && ReferenceActionSpace::allows_target(
                    input.action_space.split_targets,
                    target_slot,
                ),
            fixed_effort_cost(
                input,
                input.action_space.split_effort_base,
                ReferenceEffort::Standard,
            ),
            allocation,
        );
        let regurgitate_index = REGURGITATE_START + slot;
        set(
            regurgitate_index,
            input.self_state.gut_energy > 0
                && reachable
                && ReferenceActionSpace::allows_target(
                    input.action_space.regurgitate_targets,
                    target_slot,
                ),
            fixed_effort_cost(
                input,
                input.action_space.regurgitate_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        );
    }
    set(
        EXCAVATE,
        input.action_space.excavate_enabled,
        fixed_effort_cost(
            input,
            input.action_space.excavate_effort_base,
            ReferenceEffort::Standard,
        ),
        0,
    );
    set(
        DEPOSIT_TERRAIN,
        input.action_space.deposit_terrain_enabled
            && input.self_state.carried_material_mass
                >= input.action_space.terrain_mass_per_elevation,
        fixed_effort_cost(
            input,
            input.action_space.deposit_terrain_effort_base,
            ReferenceEffort::Standard,
        ),
        0,
    );
    let signal_amount = signal_strength_amount(input.action_space.signal_emission_cost, 0)
        .filter(|_| input.action_space.signal_enabled)
        .unwrap_or(0);
    set(SIGNAL, signal_amount > 0, 0, signal_amount);
    if !physical.iter().any(|allowed| *allowed) {
        physical[WAIT] = true;
    }
    physical
}

fn decode_valid_action_with_amount(
    action_id: usize,
    amount_choice: usize,
    input: &ReferenceMindInput,
) -> ReferenceMindDecision {
    let physical_id = action_id;
    let action = if physical_id == WAIT {
        ReferenceMindAction::Wait
    } else if physical_id < CONSUME {
        ReferenceMindAction::Guard {
            effort: effort(physical_id - GUARD_START),
        }
    } else if physical_id == CONSUME {
        ReferenceMindAction::Consume {
            amount: amount_value(input.action_space.max_consume_amount, amount_choice).unwrap_or(1),
        }
    } else if physical_id < ATTACK_START {
        let relative = physical_id - MOVE_START;
        ReferenceMindAction::Move {
            target_slot: (relative / EFFORT_COUNT) as u8,
            effort: effort(relative % EFFORT_COUNT),
        }
    } else if physical_id < SPLIT_START {
        let relative = physical_id - ATTACK_START;
        ReferenceMindAction::Attack {
            target_slot: (relative / EFFORT_COUNT) as u8,
            effort: effort(relative % EFFORT_COUNT),
            payload: attack_max_payload(input, effort(relative % EFFORT_COUNT))
                .and_then(|maximum| amount_value(maximum, amount_choice))
                .unwrap_or(1),
        }
    } else if physical_id < REGURGITATE_START {
        ReferenceMindAction::Split {
            target_slot: (physical_id - SPLIT_START) as u8,
            child_allocation: split_amount(input, amount_choice).unwrap_or(0),
            marker: input.self_state.marker,
            private_memory: input.private_memory.clone(),
        }
    } else if physical_id < EXCAVATE {
        ReferenceMindAction::Regurgitate {
            target_slot: (physical_id - REGURGITATE_START) as u8,
            amount: amount_value(input.self_state.gut_energy, amount_choice).unwrap_or(1),
        }
    } else if physical_id == EXCAVATE {
        ReferenceMindAction::Excavate
    } else if physical_id == DEPOSIT_TERRAIN {
        ReferenceMindAction::DepositTerrain
    } else {
        ReferenceMindAction::Signal {
            amounts: [input.action_space.signal_emission_cost, 0, 0, 0],
        }
    };
    ReferenceMindDecision {
        action,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Retain,
    }
}

fn amount_mask_for_legal_action(
    input: &ReferenceMindInput,
    action_id: usize,
) -> [bool; NUM_AMOUNT_CHOICES] {
    let mut mask = [false; NUM_AMOUNT_CHOICES];
    if !parameterized_action(action_id) {
        mask[0] = true;
        return mask;
    }
    let mut prior = [None; NUM_AMOUNT_CHOICES];
    for choice in 0..NUM_AMOUNT_CHOICES {
        let decision = decode_valid_action_with_amount(action_id, choice, input);
        let value = match decision.action {
            ReferenceMindAction::Attack { payload, .. } => payload,
            ReferenceMindAction::Consume { amount } => amount,
            ReferenceMindAction::Split {
                child_allocation, ..
            } => child_allocation,
            ReferenceMindAction::Regurgitate { amount, .. } => amount,
            _ => 0,
        };
        let unique = !prior[..choice].contains(&Some(value));
        prior[choice] = Some(value);
        mask[choice] = unique && value > 0;
    }
    if !mask.iter().any(|allowed| *allowed) {
        mask[0] = true;
    }
    mask
}

pub fn amount_mask(input: &ReferenceMindInput, action_id: usize) -> [bool; NUM_AMOUNT_CHOICES] {
    if action_id >= NUM_ACTIONS || !action_mask(input)[action_id] {
        let mut mask = [false; NUM_AMOUNT_CHOICES];
        mask[0] = true;
        return mask;
    }
    amount_mask_for_legal_action(input, action_id)
}

/// Convenience decoder using the historical one-eighth amount tier where it
/// is distinct, otherwise the first canonical legal amount.
pub fn decode_action(action_id: usize, input: &ReferenceMindInput) -> ReferenceMindDecision {
    if action_id >= NUM_ACTIONS || !action_mask(input)[action_id] {
        return decode_valid_action_with_amount(WAIT, 0, input);
    }
    let mask = amount_mask_for_legal_action(input, action_id);
    let amount = [1, 0, 2, 3, 4]
        .into_iter()
        .find(|choice| mask[*choice])
        .unwrap_or(0);
    decode_valid_action_with_amount(action_id, amount, input)
}

fn packed_strength_group(bits: u32, amount_choice: usize) -> u8 {
    ((bits >> (amount_choice * NUM_SIGNAL_STRENGTH_CHOICES)) & 0x1f) as u8
}

fn signal_mask_from_policy(
    masks: &PolicyMasks,
    action_id: usize,
    amount_choice: usize,
) -> [bool; NUM_SIGNAL_CHOICES] {
    let mut mask = [false; NUM_SIGNAL_CHOICES];
    if action_id == SIGNAL {
        for (pattern, allowed) in mask.iter_mut().enumerate().skip(1) {
            *allowed = masks.explicit_signal_strength_bits[pattern.count_ones() as usize - 1] != 0;
        }
    } else {
        mask[0] = true;
        if packed_strength_group(masks.sidecar_strength_bits[action_id], amount_choice) != 0 {
            for pattern in [1, 2, 4, 8] {
                mask[pattern] = true;
            }
        }
    }
    mask
}

fn signal_strength_mask_from_policy(
    masks: &PolicyMasks,
    action_id: usize,
    amount_choice: usize,
    signal_pattern: usize,
) -> [bool; NUM_SIGNAL_STRENGTH_CHOICES] {
    if signal_pattern == 0 {
        return [true, false, false, false, false];
    }
    let bits = if action_id == SIGNAL {
        masks.explicit_signal_strength_bits[signal_pattern.count_ones() as usize - 1]
    } else {
        packed_strength_group(masks.sidecar_strength_bits[action_id], amount_choice)
    };
    std::array::from_fn(|choice| bits & (1 << choice) != 0)
}

/// Signal patterns legal for one already-selected physical action. Ordinary
/// actions may select no signal or one anonymous channel. The explicit Signal
/// action must select any nonempty subset of the four channels.
pub fn signal_mask(
    input: &ReferenceMindInput,
    action_id: usize,
    amount_choice: usize,
) -> [bool; NUM_SIGNAL_CHOICES] {
    let mut fallback = [false; NUM_SIGNAL_CHOICES];
    if action_id >= NUM_ACTIONS || amount_choice >= NUM_AMOUNT_CHOICES {
        fallback[0] = true;
        return fallback;
    }
    let masks = policy_masks(input);
    if !masks.actions[action_id] || masks.amount_choice_bits[action_id] & (1 << amount_choice) == 0
    {
        fallback[0] = true;
        return fallback;
    }
    signal_mask_from_policy(&masks, action_id, amount_choice)
}

pub fn signal_strength_mask(
    input: &ReferenceMindInput,
    action_id: usize,
    amount_choice: usize,
    signal_pattern: usize,
) -> [bool; NUM_SIGNAL_STRENGTH_CHOICES] {
    if signal_pattern == 0 {
        return [true, false, false, false, false];
    }
    if action_id >= NUM_ACTIONS
        || amount_choice >= NUM_AMOUNT_CHOICES
        || signal_pattern >= NUM_SIGNAL_CHOICES
    {
        return [true, false, false, false, false];
    }
    let masks = policy_masks(input);
    if !masks.actions[action_id]
        || masks.amount_choice_bits[action_id] & (1 << amount_choice) == 0
        || !signal_mask_from_policy(&masks, action_id, amount_choice)[signal_pattern]
    {
        return [true, false, false, false, false];
    }
    signal_strength_mask_from_policy(&masks, action_id, amount_choice, signal_pattern)
}

/// Construct every conditional policy mask without recursively rebuilding the
/// physical catalog. Packed conditional bits keep each observation compact.
pub fn policy_masks(input: &ReferenceMindInput) -> PolicyMasks {
    let actions = action_mask(input);
    let mut amount_choice_bits = [1_u8; NUM_ACTIONS];
    let mut sidecar_strength_bits = [0_u32; NUM_ACTIONS];
    let mut explicit_signal_strength_bits = [0_u8; 4];
    let amount_bits = |values: [Option<u64>; NUM_AMOUNT_CHOICES]| {
        let mut bits = 0_u8;
        let mut prior = [None; NUM_AMOUNT_CHOICES];
        for (choice, value) in values.into_iter().enumerate() {
            let Some(value) = value.filter(|value| *value > 0) else {
                continue;
            };
            if !prior[..choice].contains(&Some(value)) {
                bits |= 1 << choice;
            }
            prior[choice] = Some(value);
        }
        bits
    };
    let simple_amounts = |maximum| std::array::from_fn(|choice| amount_value(maximum, choice));
    let signal_bits = |effort_cost: u64,
                       values: [Option<u64>; NUM_AMOUNT_CHOICES],
                       bits: u8,
                       payload: bool| {
        let quantum = input.action_space.signal_emission_cost;
        if !input.action_space.signal_enabled || quantum == 0 {
            return 0;
        }
        (0..NUM_AMOUNT_CHOICES).fold(0_u32, |allowed, choice| {
            if bits & (1 << choice) == 0 {
                return allowed;
            }
            let assimilated_payload = if payload {
                values[choice].unwrap_or(u64::MAX)
            } else {
                0
            };
            let required = effort_cost
                .checked_add(assimilated_payload)
                .and_then(|value| value.checked_add(input.action_space.minimum_survival_energy));
            let quanta = required
                .and_then(|required| input.self_state.assimilated_energy.checked_sub(required))
                .map_or(0, |remaining| remaining / quantum);
            allowed
                | (u32::from(signal_strength_bits(quanta))
                    << (choice * NUM_SIGNAL_STRENGTH_CHOICES))
        })
    };

    let set_simple_signal =
        |action: usize, effort_cost: u64, sidecar_strength_bits: &mut [u32; NUM_ACTIONS]| {
            if actions[action] {
                sidecar_strength_bits[action] =
                    signal_bits(effort_cost, [Some(0), None, None, None, None], 1, false);
            }
        };

    set_simple_signal(WAIT, 0, &mut sidecar_strength_bits);
    for effort_index in 0..EFFORT_COUNT {
        let selected = effort(effort_index);
        set_simple_signal(
            GUARD_START + effort_index,
            fixed_effort_cost(input, input.action_space.guard_effort_base, selected),
            &mut sidecar_strength_bits,
        );
    }
    for slot in 0..REFERENCE_MAX_LOCAL_SLOTS {
        for effort_index in 0..EFFORT_COUNT {
            let selected = effort(effort_index);
            let action = MOVE_START + slot * EFFORT_COUNT + effort_index;
            set_simple_signal(
                action,
                move_effort_cost(input, slot as u8, selected).unwrap_or(u64::MAX),
                &mut sidecar_strength_bits,
            );
        }
    }

    if actions[CONSUME] {
        let values = simple_amounts(input.action_space.max_consume_amount);
        let bits = amount_bits(values);
        amount_choice_bits[CONSUME] = bits;
        sidecar_strength_bits[CONSUME] = signal_bits(
            fixed_effort_cost(
                input,
                input.action_space.consume_effort_base,
                ReferenceEffort::Standard,
            ),
            values,
            bits,
            false,
        );
    }

    for effort_index in 0..EFFORT_COUNT {
        let selected = effort(effort_index);
        let maximum = attack_max_payload(input, selected).unwrap_or(0);
        let values = simple_amounts(maximum);
        let bits = amount_bits(values);
        let signals = signal_bits(
            fixed_effort_cost(input, input.action_space.attack_effort_base, selected),
            values,
            bits,
            true,
        );
        for slot in 0..REFERENCE_MAX_LOCAL_SLOTS {
            let action = ATTACK_START + slot * EFFORT_COUNT + effort_index;
            if actions[action] {
                amount_choice_bits[action] = bits;
                sidecar_strength_bits[action] = signals;
            }
        }
    }

    let split_values = std::array::from_fn(|choice| split_amount(input, choice));
    let split_bits = amount_bits(split_values);
    let split_signals = signal_bits(
        fixed_effort_cost(
            input,
            input.action_space.split_effort_base,
            ReferenceEffort::Standard,
        ),
        split_values,
        split_bits,
        true,
    );
    let regurgitate_values = simple_amounts(input.self_state.gut_energy);
    let regurgitate_bits = amount_bits(regurgitate_values);
    let regurgitate_signals = signal_bits(
        fixed_effort_cost(
            input,
            input.action_space.regurgitate_effort_base,
            ReferenceEffort::Standard,
        ),
        regurgitate_values,
        regurgitate_bits,
        false,
    );
    for slot in 0..REFERENCE_MAX_LOCAL_SLOTS {
        let split = SPLIT_START + slot;
        if actions[split] {
            amount_choice_bits[split] = split_bits;
            sidecar_strength_bits[split] = split_signals;
        }
        let regurgitate = REGURGITATE_START + slot;
        if actions[regurgitate] {
            amount_choice_bits[regurgitate] = regurgitate_bits;
            sidecar_strength_bits[regurgitate] = regurgitate_signals;
        }
    }
    set_simple_signal(
        EXCAVATE,
        fixed_effort_cost(
            input,
            input.action_space.excavate_effort_base,
            ReferenceEffort::Standard,
        ),
        &mut sidecar_strength_bits,
    );
    set_simple_signal(
        DEPOSIT_TERRAIN,
        fixed_effort_cost(
            input,
            input.action_space.deposit_terrain_effort_base,
            ReferenceEffort::Standard,
        ),
        &mut sidecar_strength_bits,
    );
    if actions[SIGNAL] {
        let quantum = input.action_space.signal_emission_cost;
        let expendable = input
            .self_state
            .assimilated_energy
            .saturating_sub(input.action_space.minimum_survival_energy);
        for channel_count in 1..=4 {
            let quanta = (expendable / channel_count as u64)
                .checked_div(quantum)
                .unwrap_or(0);
            explicit_signal_strength_bits[channel_count - 1] = signal_strength_bits(quanta);
        }
    }
    PolicyMasks {
        actions,
        amount_choice_bits,
        sidecar_strength_bits,
        explicit_signal_strength_bits,
    }
}

/// Decode all factored heads against the same anonymous decision frontier.
/// Any malformed pair fails closed to a no-signal Wait.
pub fn decode_policy_choice(
    choice: PolicyChoice,
    input: &ReferenceMindInput,
) -> ReferenceMindDecision {
    let masks = policy_masks(input);
    let amount_allowed = choice.amount < NUM_AMOUNT_CHOICES
        && choice.action < NUM_ACTIONS
        && masks.amount_choice_bits[choice.action] & (1 << choice.amount) != 0;
    let signal_allowed = choice.signal == 0
        || (choice.signal < NUM_SIGNAL_CHOICES
            && choice.action < NUM_ACTIONS
            && choice.amount < NUM_AMOUNT_CHOICES
            && signal_mask_from_policy(&masks, choice.action, choice.amount)[choice.signal]);
    let strength_allowed = choice.signal < NUM_SIGNAL_CHOICES
        && choice.signal_strength < NUM_SIGNAL_STRENGTH_CHOICES
        && choice.action < NUM_ACTIONS
        && choice.amount < NUM_AMOUNT_CHOICES
        && signal_strength_mask_from_policy(&masks, choice.action, choice.amount, choice.signal)
            [choice.signal_strength];
    if choice.action >= NUM_ACTIONS
        || !masks.actions[choice.action]
        || !amount_allowed
        || !signal_allowed
        || !strength_allowed
        || (choice.action == SIGNAL && choice.signal == 0)
    {
        return decode_valid_action_with_amount(WAIT, 0, input);
    }
    let mut decision = decode_valid_action_with_amount(choice.action, choice.amount, input);
    if choice.signal > 0 {
        let amount = signal_strength_amount(
            input.action_space.signal_emission_cost,
            choice.signal_strength,
        )
        .expect("masked signal strength must fit");
        if choice.action == SIGNAL {
            decision.action = ReferenceMindAction::Signal {
                amounts: std::array::from_fn(|channel| {
                    if choice.signal & (1 << channel) != 0 {
                        amount
                    } else {
                        0
                    }
                }),
            };
        } else {
            debug_assert!(choice.signal.is_power_of_two());
            decision.signal = Some(ReferenceSignalEmission {
                channel: choice.signal.trailing_zeros() as u8,
                amount,
            });
        }
    }
    decision
}

pub fn encode_action(action: &ReferenceMindAction) -> Option<usize> {
    match action {
        ReferenceMindAction::Wait => Some(WAIT),
        ReferenceMindAction::Guard { effort } => Some(GUARD_START + effort_index(*effort)),
        ReferenceMindAction::Consume { .. } => Some(CONSUME),
        ReferenceMindAction::Move {
            target_slot,
            effort,
        } if usize::from(*target_slot) < REFERENCE_MAX_LOCAL_SLOTS => {
            Some(MOVE_START + usize::from(*target_slot) * EFFORT_COUNT + effort_index(*effort))
        }
        ReferenceMindAction::Attack {
            target_slot,
            effort,
            ..
        } if usize::from(*target_slot) < REFERENCE_MAX_LOCAL_SLOTS => {
            Some(ATTACK_START + usize::from(*target_slot) * EFFORT_COUNT + effort_index(*effort))
        }
        ReferenceMindAction::Split { target_slot, .. }
            if usize::from(*target_slot) < REFERENCE_MAX_LOCAL_SLOTS =>
        {
            Some(SPLIT_START + usize::from(*target_slot))
        }
        ReferenceMindAction::Regurgitate { target_slot, .. }
            if usize::from(*target_slot) < REFERENCE_MAX_LOCAL_SLOTS =>
        {
            Some(REGURGITATE_START + usize::from(*target_slot))
        }
        ReferenceMindAction::Excavate => Some(EXCAVATE),
        ReferenceMindAction::DepositTerrain => Some(DEPOSIT_TERRAIN),
        ReferenceMindAction::Signal { .. } => Some(SIGNAL),
        _ => None,
    }
}

/// Encodes a Mind decision into the nearest legal factored amount tier. The
/// caller can compare a decoded choice with the source to distinguish exact
/// from abstract behavior-cloning labels.
pub fn encode_decision(
    decision: &ReferenceMindDecision,
    input: &ReferenceMindInput,
) -> Option<PolicyChoice> {
    let action = encode_action(&decision.action)?;
    let masks = policy_masks(input);
    if !masks.actions[action] {
        return None;
    }
    let desired = match decision.action {
        ReferenceMindAction::Attack { payload, .. } => Some(payload),
        ReferenceMindAction::Consume { amount } => Some(amount),
        ReferenceMindAction::Split {
            child_allocation, ..
        } => Some(child_allocation),
        ReferenceMindAction::Regurgitate { amount, .. } => Some(amount),
        _ => None,
    };
    let amount = if let Some(desired) = desired {
        (0..NUM_AMOUNT_CHOICES)
            .filter(|choice| masks.amount_choice_bits[action] & (1 << choice) != 0)
            .min_by_key(|choice| {
                let candidate = decode_valid_action_with_amount(action, *choice, input);
                let value = match candidate.action {
                    ReferenceMindAction::Attack { payload, .. } => payload,
                    ReferenceMindAction::Consume { amount } => amount,
                    ReferenceMindAction::Split {
                        child_allocation, ..
                    } => child_allocation,
                    ReferenceMindAction::Regurgitate { amount, .. } => amount,
                    _ => 0,
                };
                value.abs_diff(desired)
            })?
    } else {
        0
    };
    let (signal, desired_strength) = match (&decision.action, decision.signal) {
        (ReferenceMindAction::Signal { .. }, Some(_)) => return None,
        (ReferenceMindAction::Signal { amounts }, None) => {
            let pattern = amounts
                .iter()
                .enumerate()
                .fold(0_usize, |pattern, (channel, amount)| {
                    pattern | (usize::from(*amount > 0) << channel)
                });
            let count = u64::from(pattern.count_ones());
            if pattern == 0 {
                return None;
            }
            let total = amounts
                .iter()
                .try_fold(0_u64, |total, value| total.checked_add(*value))?;
            (pattern, total / count)
        }
        (_, Some(emission)) if emission.channel < 4 && emission.amount > 0 => {
            (1 << emission.channel, emission.amount)
        }
        (_, Some(_)) => return None,
        (_, None) => (0, 0),
    };
    if !signal_mask_from_policy(&masks, action, amount)[signal] {
        return None;
    }
    let strength_mask = signal_strength_mask_from_policy(&masks, action, amount, signal);
    let signal_strength = if signal == 0 {
        0
    } else {
        (0..NUM_SIGNAL_STRENGTH_CHOICES)
            .filter(|choice| strength_mask[*choice])
            .min_by_key(|choice| {
                signal_strength_amount(input.action_space.signal_emission_cost, *choice)
                    .unwrap_or(u64::MAX)
                    .abs_diff(desired_strength)
            })?
    };
    Some(PolicyChoice {
        action,
        amount,
        signal,
        signal_strength,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_interface::randomness::PrivateRandom;
    use blob_interface::reference_mind::{
        CurrentTileObservation, LocalObservation, ReferenceActionSpace, ReferenceSelfState,
        EFFORT_BURST_BIT, EFFORT_GENTLE_BIT, EFFORT_STANDARD_BIT,
    };

    fn input() -> ReferenceMindInput {
        ReferenceMindInput {
            self_state: ReferenceSelfState {
                core_mass: 10,
                assimilated_energy: 100,
                gut_energy: 20,
                metabolism_remainder: 0,
                carried_material_mass: 0,
                marker: 7,
                guarded: false,
                last_outcome: None,
            },
            current_tile: CurrentTileObservation {
                elevation: 0,
                plant_energy: 5,
                plant_capacity: 10,
                plant_growth_rate: 1,
                loose_energy: 0,
                diffuse_energy: 0,
                signal_energy: [0; 4],
            },
            slots: (0..8)
                .map(|slot| LocalObservation {
                    slot,
                    dx: 1,
                    dy: 0,
                    distance_cost_q10: 1024,
                    reachable: true,
                    elevation: Some(0),
                    plant_energy: Some(0),
                    plant_capacity: Some(0),
                    plant_growth_rate: Some(0),
                    loose_energy: Some(0),
                    diffuse_energy: Some(0),
                    signal_energy: Some([0; 4]),
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
                max_consume_amount: 12,
                gut_capacity: 64,
                max_private_memory_bytes: 2048,
                minimum_survival_energy: 1,
                child_core_mass: 10,
                metabolism_rate_numerator: 1,
                metabolism_rate_denominator: 1024,
                terrain_mass_per_elevation: 10,
                signal_emission_cost: 1,
                effort_cost_numerators: [1, 1, 2],
                effort_cost_denominators: [2, 1, 1],
                move_effort_base: 1,
                move_mass_units_per_effort: 100,
                attack_effort_base: 2,
                guard_effort_base: 1,
                consume_effort_base: 1,
                split_effort_base: 2,
                regurgitate_effort_base: 1,
                excavate_effort_base: 2,
                deposit_terrain_effort_base: 2,
            },
            private_memory: vec![1, 2, 3],
            randomness: PrivateRandom::ZERO,
        }
    }

    #[test]
    fn catalog_decodes_only_reference_actions_and_preserves_memory() {
        let input = input();
        let mask = action_mask(&input);
        for (action_id, allowed) in mask.into_iter().enumerate() {
            if allowed {
                let decision = decode_action(action_id, &input);
                assert!(action_is_commit_legal(
                    &input,
                    &decision.action,
                    decision.signal.is_some()
                ));
                assert_eq!(decision.memory_update, ReferenceMemoryUpdate::Retain);
                assert_eq!(encode_action(&decision.action), Some(action_id));
                assert_eq!(decision.signal, None);
            }
        }
    }

    #[test]
    fn masks_ruleset_disabled_and_unreachable_targets() {
        let mut input = input();
        input.slots[3].reachable = false;
        input.action_space.move_targets &= !(1 << 4);
        let mask = action_mask(&input);
        for effort_index in 0..EFFORT_COUNT {
            assert!(!mask[MOVE_START + 3 * EFFORT_COUNT + effort_index]);
            assert!(!mask[MOVE_START + 4 * EFFORT_COUNT + effort_index]);
        }
    }

    #[test]
    fn invalid_policy_choice_becomes_reference_wait() {
        let mut input = input();
        input.action_space.guard_enabled = false;
        assert_eq!(
            decode_action(GUARD_START, &input).action,
            ReferenceMindAction::Wait
        );
    }

    #[test]
    fn decoded_policy_decisions_round_trip_through_the_abstract_catalog() {
        let input = input();
        for action_id in 0..NUM_ACTIONS {
            if action_mask(&input)[action_id] {
                let decision = decode_action(action_id, &input);
                let encoded = encode_decision(&decision, &input).unwrap();
                assert_eq!(encoded.action, action_id);
                assert_eq!(encoded.signal == 0, action_id != SIGNAL);
                for amount in 0..NUM_AMOUNT_CHOICES {
                    if amount_mask(&input, action_id)[amount] {
                        for signal in 0..NUM_SIGNAL_CHOICES {
                            if signal_mask(&input, action_id, amount)[signal] {
                                for signal_strength in 0..NUM_SIGNAL_STRENGTH_CHOICES {
                                    if signal_strength_mask(&input, action_id, amount, signal)
                                        [signal_strength]
                                    {
                                        let choice = PolicyChoice {
                                            action: action_id,
                                            amount,
                                            signal,
                                            signal_strength,
                                        };
                                        assert_eq!(
                                            encode_decision(
                                                &decode_policy_choice(choice, &input),
                                                &input,
                                            ),
                                            Some(choice)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn learned_memory_updates_both_parent_and_split_child() {
        let input = input();
        let split = SPLIT_START;
        assert!(action_mask(&input)[split]);
        let mut decision = decode_action(split, &input);
        attach_policy_memory(&mut decision, vec![8, 9]);
        assert_eq!(
            decision.memory_update,
            ReferenceMemoryUpdate::Replace(vec![8, 9])
        );
        let ReferenceMindAction::Split { private_memory, .. } = decision.action else {
            panic!("split catalog entry decoded as another action")
        };
        assert_eq!(private_memory, vec![8, 9]);
    }

    #[test]
    fn masks_actions_and_signals_that_cannot_preserve_survival_energy() {
        let mut input = input();
        input.self_state.assimilated_energy = 2;
        let mask = action_mask(&input);
        assert!(mask[GUARD_START]);
        assert!(!mask[GUARD_START + 2]);
        assert!(!signal_mask(&input, GUARD_START, 0)[1]);
        assert!((0..REFERENCE_MAX_LOCAL_SLOTS).all(|slot| {
            (0..EFFORT_COUNT).all(|effort| !mask[ATTACK_START + slot * EFFORT_COUNT + effort])
        }));
    }

    #[test]
    fn parameter_heads_cover_monotonic_commitments_without_duplicate_tiers() {
        let input = input();
        for action in [CONSUME, ATTACK_START + 1, SPLIT_START, REGURGITATE_START] {
            let mask = amount_mask(&input, action);
            let values = (0..NUM_AMOUNT_CHOICES)
                .filter(|choice| mask[*choice])
                .map(|amount| {
                    let decision = decode_policy_choice(
                        PolicyChoice {
                            action,
                            amount,
                            signal: 0,
                            signal_strength: 0,
                        },
                        &input,
                    );
                    match decision.action {
                        ReferenceMindAction::Attack { payload, .. } => payload,
                        ReferenceMindAction::Consume { amount } => amount,
                        ReferenceMindAction::Split {
                            child_allocation, ..
                        } => child_allocation,
                        ReferenceMindAction::Regurgitate { amount, .. } => amount,
                        other => panic!("parameterized choice decoded as {other:?}"),
                    }
                })
                .collect::<Vec<_>>();
            assert!(values.len() >= 2);
            assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn signal_pattern_and_strength_heads_cover_sidecars_and_multichannel_actions() {
        let mut input = input();
        input.action_space.signal_emission_cost = 3;

        let sidecar = PolicyChoice {
            action: WAIT,
            amount: 0,
            signal: 0b0100,
            signal_strength: 2,
        };
        let decoded = decode_policy_choice(sidecar, &input);
        assert_eq!(
            decoded.signal,
            Some(ReferenceSignalEmission {
                channel: 2,
                amount: 12,
            })
        );
        assert_eq!(encode_decision(&decoded, &input), Some(sidecar));

        let vector = PolicyChoice {
            action: SIGNAL,
            amount: 0,
            signal: 0b1011,
            signal_strength: 1,
        };
        let decoded = decode_policy_choice(vector, &input);
        assert_eq!(
            decoded.action,
            ReferenceMindAction::Signal {
                amounts: [6, 6, 0, 6],
            }
        );
        assert_eq!(decoded.signal, None);
        assert_eq!(encode_decision(&decoded, &input), Some(vector));

        let asymmetric = ReferenceMindDecision {
            action: ReferenceMindAction::Signal {
                amounts: [3, 6, 0, 12],
            },
            signal: None,
            memory_update: ReferenceMemoryUpdate::Retain,
        };
        let abstracted = encode_decision(&asymmetric, &input).unwrap();
        assert_ne!(decode_policy_choice(abstracted, &input), asymmetric);

        input.self_state.assimilated_energy = 20;
        assert!(!signal_strength_mask(&input, SIGNAL, 0, 0b1111)[1]);
    }

    #[test]
    fn hierarchical_catalog_round_trips_and_projects_conditional_masks() {
        for action in 0..NUM_ACTIONS {
            let choice = decompose_policy_action(action).unwrap();
            assert_eq!(compose_policy_action(choice), Some(action));
        }

        let input = input();
        let flat = action_mask(&input);
        let kinds = policy_action_kind_mask(&flat);
        for (action, allowed) in flat.iter().copied().enumerate() {
            let choice = decompose_policy_action(action).unwrap();
            if allowed {
                assert!(kinds[choice.kind]);
                assert!(policy_target_mask(&flat, choice.kind)[choice.target]);
                assert!(policy_effort_mask(&flat, choice.kind, choice.target)[choice.effort]);
            }
        }
        for (kind, kind_allowed) in kinds.iter().copied().enumerate() {
            for target in 0..NUM_POLICY_TARGETS {
                for effort in 0..NUM_POLICY_EFFORTS {
                    let choice = HierarchicalActionChoice {
                        kind,
                        target,
                        effort,
                    };
                    if let Some(action) = compose_policy_action(choice) {
                        let projected = kind_allowed
                            && policy_target_mask(&flat, kind)[target]
                            && policy_effort_mask(&flat, kind, target)[effort];
                        assert_eq!(projected, flat[action]);
                    }
                }
            }
        }
    }
}
