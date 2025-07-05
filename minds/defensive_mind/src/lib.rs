use extism_pdk::*;
use tinyrand::{StdRand, Seeded, Rand, RandRange};
use blob_mind_utils::*;

use blob_interface::cell::{BlobState, CellAction, CellContext};
use blob_interface::mind_output_converter::mind_output_to_capnp;
use blob_interface::mind_input_converter::capnp_to_mind_input;

// Constants for defensive behavior
const SPLIT_ENERGY_AMOUNT: i32 = 60;

fn defensive_strategy(blob_state: &BlobState, context: &CellContext, seed: u64) -> CellAction {
    let mut rng = StdRand::seed(seed ^ blob_state.age as u64);
    
    // Calculate dynamic thresholds based on max_energy
    let safe_energy_threshold = (blob_state.max_energy * 2) / 10; // 20% of max energy
    let min_energy_to_split = (blob_state.max_energy * 6) / 10; // 60% of max energy - defensive, so higher threshold
    let critical_energy = blob_state.max_energy / 10; // 10% of max energy
    
    // Priority 1: Always defend if energy is critically low
    if blob_state.energy < critical_energy {
        return CellAction::Defend;
    }
    
    // Priority 2: Eat if we're on energy and not at max capacity
    if blob_state.energy < blob_state.max_energy && has_energy_here(context) {
        return CellAction::Eat;
    }
    
    // Priority 3: Build defensive positions (lift terrain when safe)
    if blob_state.energy > safe_energy_threshold * 2 && !blob_state.loaded && rng.next_range(0usize..8) == 0 {
        // Try to lift terrain to create defensive positions
        if context.elevation[8] > 0 {
            return CellAction::LiftTerrain;
        }
    }
    
    // Priority 4: Dump terrain to create barriers
    if blob_state.loaded && rng.next_range(0usize..4) == 0 {
        return CellAction::DumpTerrain;
    }
    
    // Priority 5: Split conservatively when we have abundant energy
    if blob_state.energy > min_energy_to_split {
        // Only split in very safe locations
        if is_high_ground(context) {
            if let Some(direction) = find_safe_move_direction(context, &mut rng) {
                let child_memory = [0u8; 2048];
                return CellAction::Split(direction, SPLIT_ENERGY_AMOUNT, 3, child_memory);
            }
        }
    }
    
    // Priority 6: Move cautiously towards energy if needed
    if blob_state.energy < safe_energy_threshold {
        if let Some(direction) = find_best_energy_direction(context) {
            // Only move if the destination is safe (similar elevation)
            let dir_index = direction.index();
            let elevation_diff = (context.elevation[dir_index] - context.elevation[8]).abs();
            if elevation_diff <= 1 {
                return CellAction::Move(direction);
            }
        }
    }
    
    // Priority 7: Stay in place or move very conservatively
    if blob_state.energy > safe_energy_threshold {
        // Occasionally move to better positions, but very carefully
        if rng.next_range(0usize..20) == 0 {
            if let Some(direction) = find_safe_move_direction(context, &mut rng) {
                return CellAction::Move(direction);
            }
        }
    }
    
    // Default: defend in place
    CellAction::Defend
}

#[plugin_fn]
pub fn mind_function(input: Vec<u8>) -> FnResult<Vec<u8>> {
    let (blob_state, cell_context, seed) = capnp_to_mind_input(&input)?;
    
    let action = defensive_strategy(&blob_state, &cell_context, seed);
    
    let capnp_vec = mind_output_to_capnp(&action, &blob_state.memory).map_err(|e| Error::msg(e.to_string()))?;
    Ok(capnp_vec)
} 