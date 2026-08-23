use extism_pdk::*;

/// Deliberately does not export `reference_mind_function`.
#[plugin_fn]
pub fn wrong_export(_input: Vec<u8>) -> FnResult<Vec<u8>> {
    Ok(Vec::new())
}
