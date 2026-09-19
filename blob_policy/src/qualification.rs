//! Host-only deployment fixtures. This module is not compiled into WASM Minds.
/// Exposed development evidence, never a confirmation cohort.
pub const DEVELOPMENT_SEEDS: [u64; 4] = [72101, 72102, 72103, 72104];

pub fn check_confirmation_reservations(
    reserved: impl IntoIterator<Item = u64>,
) -> Result<(), String> {
    if let Some(seed) = reserved
        .into_iter()
        .find(|seed| DEVELOPMENT_SEEDS.contains(seed))
    {
        return Err(format!(
            "deployment development seed {seed} is reserved for confirmation"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deployment_fixtures_cannot_consume_reserved_confirmation_seeds() {
        for seed in DEVELOPMENT_SEEDS {
            assert!(check_confirmation_reservations([seed]).is_err());
        }
        assert!(check_confirmation_reservations([1434999901, 1434999902]).is_ok());
        assert!(check_confirmation_reservations([]).is_ok());
    }
}
