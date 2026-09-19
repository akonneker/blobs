//! Frozen learned policy under the ordinary anonymous Extism PDK Mind ABI.
#[cfg(target_arch = "wasm32")]
use extism_pdk::*;
#[cfg(target_arch = "wasm32")]
#[plugin_fn]
pub fn reference_mind_function(bytes: Vec<u8>) -> FnResult<Vec<u8>> {
    use blob_interface::reference_mind_converter::{
        capnp_to_reference_mind_input, reference_mind_decision_to_capnp, ReferenceMindLimits,
    };
    let limits = ReferenceMindLimits::default();
    let input =
        capnp_to_reference_mind_input(&bytes, limits).map_err(|e| Error::msg(e.to_string()))?;
    let policy = blob_policy::composite::DeployedPolicy::from_bytes(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/weights.bin"
    )))
    .map_err(Error::msg)?;
    let decision = policy.try_decide(&input).map_err(Error::msg)?;
    Ok(reference_mind_decision_to_capnp(&decision, limits)
        .map_err(|e| Error::msg(e.to_string()))?)
}
