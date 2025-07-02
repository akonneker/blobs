use extism_pdk::*;
use tinyrand::{StdRand, Seeded, Rand, RandRange};
use blob_mind_utils::*;

use blob_interface::cell::{BlobState, CellAction, CellContext};
use blob_interface::action_converter::cell_action_to_capnp;
use blob_interface::mind_input_converter::capnp_to_mind_input;
use blob_interface::types::Direction;

// Constants for aggressive behavior
const MIN_ENERGY_TO_ATTACK: u32 = 60;
const SPLIT_ENERGY_AMOUNT: i32 = 70;

fn aggressive_strategy(blob_state: &BlobState, context: &CellContext, seed: u64) -> CellAction {
    let mut rng = StdRand::seed(seed ^ blob_state.age as u64);
    
    // Calculate dynamic thresholds based on max_energy
    let min_energy_to_split = (blob_state.max_energy * 3) / 10; // 30% of max energy
    
    // Priority 1: If we have low energy, focus on eating
    if blob_state.energy < blob_state.max_energy / 4 {
        if has_energy_here(context) {
            return CellAction::Eat;
        }
        
        // Move towards energy aggressively
        if let Some(direction) = find_best_energy_direction(context) {
            return CellAction::Move(direction);
        }
    }
    
    // Priority 2: Eat if there's energy and we're not at max
    if has_energy_here(context) && blob_state.energy < blob_state.max_energy {
        return CellAction::Eat;
    }
    
    // Priority 3: Attack if we have sufficient energy and detect threats
    if blob_state.energy > MIN_ENERGY_TO_ATTACK {
        // Look for nearby cells to attack (simplified - attack in random direction)
        if rng.next_range(0usize..5) == 0 {
            let directions = Direction::all();
            let attack_direction = directions[rng.next_range(0..directions.len())];
            return CellAction::Attack(attack_direction);
        }
    }
    
    // Priority 4: Split aggressively if we have high energy
    if blob_state.energy > min_energy_to_split {
        if let Some(direction) = find_safe_move_direction(context, &mut rng) {
            // Create aggressive child
            let child_memory = [0u8; 2048]; // Start fresh
            return CellAction::Split(direction, SPLIT_ENERGY_AMOUNT, 2, child_memory);
        }
    }
    
    // Priority 5: Move aggressively towards targets
    if blob_state.energy > 30 {
        // Prefer moving to higher ground for better position
        if !is_high_ground(context) {
            let directions = Direction::all();
            let center_elevation = context.elevation[8];
            
            for (i, direction) in directions.iter().enumerate() {
                if context.elevation[i] > center_elevation {
                    let elevation_diff = context.elevation[i] - center_elevation;
                    if elevation_diff <= 1 {
                        return CellAction::Move(*direction);
                    }
                }
            }
        }
        
        // Otherwise move randomly but safely
        if let Some(direction) = find_safe_move_direction(context, &mut rng) {
            return CellAction::Move(direction);
        }
    }
    
    // Fallback: defend if we can't do anything else
    CellAction::Defend
}

#[plugin_fn]
pub fn mind_function(input: Vec<u8>) -> FnResult<Vec<u8>> {
    let (blob_state, cell_context, seed) = capnp_to_mind_input(&input)?;
    
    let action = aggressive_strategy(&blob_state, &cell_context, seed);
    
    let capnp_vec = cell_action_to_capnp(&action).map_err(|e| Error::msg(e.to_string()))?;
    Ok(capnp_vec)
}
