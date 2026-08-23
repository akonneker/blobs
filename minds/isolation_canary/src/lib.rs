use std::sync::atomic::{AtomicU32, Ordering};

use blob_interface::reference_mind::{
    ReferenceMemoryUpdate, ReferenceMindAction, ReferenceMindDecision,
};
use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, capnp_to_reference_mind_input, reference_mind_decision_to_capnp,
};
use extism_pdk::*;

static INVOCATIONS_SINCE_RESET: AtomicU32 = AtomicU32::new(0);

/// Reference-native canary. Pristine arbitrary WASM state always selects Wait;
/// explicit per-cell memory is incremented and must survive only through the
/// decision output and authoritative replay commitment.
#[plugin_fn]
pub fn reference_mind_function(input: Vec<u8>) -> FnResult<Vec<u8>> {
    let limits = ReferenceMindLimits::default();
    let input = capnp_to_reference_mind_input(&input, limits)
        .map_err(|error| Error::msg(error.to_string()))?;
    let prior = INVOCATIONS_SINCE_RESET.fetch_add(1, Ordering::Relaxed);
    let action = if prior == 0 {
        ReferenceMindAction::Wait
    } else {
        ReferenceMindAction::Guard {
            effort: blob_interface::reference_mind::ReferenceEffort::Standard,
        }
    };
    let mut memory = input.private_memory;
    if memory.is_empty() {
        memory.push(1);
    } else {
        memory[0] = memory[0].wrapping_add(1);
    }
    let decision = ReferenceMindDecision {
        action,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Replace(memory),
    };
    Ok(reference_mind_decision_to_capnp(&decision, limits)
        .map_err(|error| Error::msg(error.to_string()))?)
}
