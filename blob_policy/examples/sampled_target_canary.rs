//! ABI numerical canary for the sampled target contract. No learned weights.
#[cfg(target_arch = "wasm32")]
#[path = "sampled_target_fixture/mod.rs"]
mod fixture;

#[cfg(target_arch = "wasm32")]
use extism_pdk::*;

#[cfg(target_arch = "wasm32")]
#[plugin_fn]
pub fn reference_mind_function(bytes: Vec<u8>) -> FnResult<Vec<u8>> {
    use blob_interface::reference_mind_converter::*;
    let limits = ReferenceMindLimits::default();
    let input = capnp_to_reference_mind_input(&bytes, limits)
        .map_err(|error| Error::msg(error.to_string()))?;
    let decision = fixture::decide(&input).map_err(Error::msg)?;
    Ok(reference_mind_decision_to_capnp(&decision, limits)
        .map_err(|error| Error::msg(error.to_string()))?)
}
