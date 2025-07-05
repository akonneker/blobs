use extism_pdk::*;
use tinyrand::{StdRand, Seeded, Rand, RandRange};

use blob_interface::cell::{BlobState, CellAction, CellContext};
use blob_interface::mind_output_converter::mind_output_to_capnp;
use blob_interface::mind_input_converter::capnp_to_mind_input;
use blob_interface::types::Direction;

const VISITED_PHEROMONE: u32 = 42;

// Memory layout: first byte stores last direction (0-7, 255 for none)
const LAST_DIRECTION_OFFSET: usize = 0;

// Helper functions
fn has_energy_here(context: &CellContext) -> bool {
    context.energy[8] > 0
}

fn is_visited(context: &CellContext) -> bool {
    match context.pheromone[8] {
        Some(pheromone) => pheromone == VISITED_PHEROMONE,
        None => false,
    }
}

fn get_last_direction(blob_state: &BlobState) -> Option<Direction> {
    if blob_state.memory.len() > LAST_DIRECTION_OFFSET {
        let direction_val = blob_state.memory[LAST_DIRECTION_OFFSET];
        // Treat 0 as "no direction" since new cells start with zero-initialized memory
        // Valid directions are stored as 1-8, with 255 also meaning "no direction"
        if direction_val >= 1 && direction_val <= 8 {
            Some(Direction::all()[(direction_val - 1) as usize])
        } else {
            None
        }
    } else {
        None
    }
}

fn set_last_direction(memory: &mut [u8; 2048], direction: Option<Direction>) {
    if memory.len() > LAST_DIRECTION_OFFSET {
        memory[LAST_DIRECTION_OFFSET] = match direction {
            Some(dir) => (dir.index() + 1) as u8, // Store as 1-8 instead of 0-7
            None => 0, // Use 0 for "no direction"
        };
    }
}

// Get a consistent direction bias based on cell properties to reduce random wandering
fn get_preferred_direction(blob_state: &BlobState, rng: &mut StdRand) -> Direction {
    // Use cell ID and age to create a stable bias
    let bias_seed = blob_state.age.wrapping_mul(31).wrapping_add(blob_state.marker.wrapping_mul(17));
    let mut bias_rng = StdRand::seed(bias_seed as u64);
    let directions = Direction::all();
    directions[bias_rng.next_range(0..directions.len())]
}

fn opposite_direction(direction: Direction) -> Direction {
    match direction {
        Direction::North => Direction::South,
        Direction::South => Direction::North,
        Direction::East => Direction::West,
        Direction::West => Direction::East,
        Direction::NorthEast => Direction::SouthWest,
        Direction::NorthWest => Direction::SouthEast,
        Direction::SouthEast => Direction::NorthWest,
        Direction::SouthWest => Direction::NorthEast,
    }
}

fn find_any_energy_direction(context: &CellContext, rng: &mut StdRand) -> Option<Direction> {
    let directions = Direction::all();
    let mut energy_directions = Vec::new();
    
    for (i, direction) in directions.iter().enumerate() {
        // Check if this direction has energy and is unoccupied
        if context.energy[i] > 0 {
            // Check if the square is unoccupied (no marker means no blob)
            let is_unoccupied = context.markers[i].is_none();
            
            if is_unoccupied {
                energy_directions.push(*direction);
            }
        }
    }
    
    if energy_directions.is_empty() {
        None
    } else {
        let index = rng.next_range(0..energy_directions.len());
        Some(energy_directions[index])
    }
}

fn find_enemy_direction(context: &CellContext, own_marker: u32) -> Option<Direction> {
    let directions = Direction::all();
    
    for (i, direction) in directions.iter().enumerate() {
        if let Some(marker) = context.markers[i] {
            // If there's a marker and it's different from ours, it's an enemy
            if marker != own_marker {
                return Some(*direction);
            }
        }
    }
    
    None
}

fn find_unvisited_direction(context: &CellContext, rng: &mut StdRand, last_direction: Option<Direction>, blob_state: &BlobState) -> Option<Direction> {
    let directions = Direction::all();
    
    // First, if we have a last direction, check if we can continue in that direction
    if let Some(last_dir) = last_direction {
        let last_dir_index = last_dir.index();
        
        // Check if continuing in the same direction is valid
        let is_square_visited = match context.pheromone[last_dir_index] {
            Some(pheromone) => pheromone == VISITED_PHEROMONE,
            None => false,
        };
        
        let elevation_diff = (context.elevation[last_dir_index] - context.elevation[8]).abs();
        let is_occupied = context.markers[last_dir_index].is_some();
        
        // If the last direction leads to an unvisited, safe, unoccupied square, continue that way
        if !is_square_visited && elevation_diff <= 1 && !is_occupied {
            return Some(last_dir);
        }
    }
    
    // If we can't continue in the same direction, find all other unvisited directions
    let mut unvisited_directions = Vec::new();
    
    for (i, direction) in directions.iter().enumerate() {
        let is_square_visited = match context.pheromone[i] {
            Some(pheromone) => pheromone == VISITED_PHEROMONE,
            None => false,
        };
        
        // Check elevation difference is safe (within 1)
        let elevation_diff = (context.elevation[i] - context.elevation[8]).abs();
        let is_occupied = context.markers[i].is_some();
        
        if !is_square_visited && elevation_diff <= 1 && !is_occupied {
            unvisited_directions.push(*direction);
        }
    }
    
    // If no last direction (new cell), prefer the consistent direction if it's available
    if last_direction.is_none() && !unvisited_directions.is_empty() {
        let preferred = get_preferred_direction(blob_state, rng);
        if unvisited_directions.contains(&preferred) {
            return Some(preferred);
        }
    }
    
    // Avoid backtracking if possible
    if let Some(last_dir) = last_direction {
        let opposite = opposite_direction(last_dir);
        unvisited_directions.retain(|&dir| dir != opposite);
        
        // If we filtered out all directions, add them back (we have no choice)
        if unvisited_directions.is_empty() {
            for (i, direction) in directions.iter().enumerate() {
                let is_square_visited = match context.pheromone[i] {
                    Some(pheromone) => pheromone == VISITED_PHEROMONE,
                    None => false,
                };
                
                let elevation_diff = (context.elevation[i] - context.elevation[8]).abs();
                let is_occupied = context.markers[i].is_some();
                
                if !is_square_visited && elevation_diff <= 1 && !is_occupied {
                    unvisited_directions.push(*direction);
                }
            }
        }
    }
    
    if unvisited_directions.is_empty() {
        None
    } else {
        let index = rng.next_range(0..unvisited_directions.len());
        Some(unvisited_directions[index])
    }
}

fn find_random_safe_direction(context: &CellContext, rng: &mut StdRand, last_direction: Option<Direction>) -> Option<Direction> {
    let directions = Direction::all();
    
    // First, if we have a last direction, check if we can continue in that direction
    if let Some(last_dir) = last_direction {
        let last_dir_index = last_dir.index();
        
        // Check if continuing in the same direction is valid
        let elevation_diff = (context.elevation[last_dir_index] - context.elevation[8]).abs();
        let is_occupied = context.markers[last_dir_index].is_some();
        
        // If the last direction leads to a safe, unoccupied square, continue that way
        if elevation_diff <= 1 && !is_occupied {
            return Some(last_dir);
        }
    }
    
    // If we can't continue in the same direction, find all other safe directions
    let mut safe_directions = Vec::new();
    
    for (i, direction) in directions.iter().enumerate() {
        let elevation_diff = (context.elevation[i] - context.elevation[8]).abs();
        let is_occupied = context.markers[i].is_some();
        
        if elevation_diff <= 1 && !is_occupied {
            safe_directions.push(*direction);
        }
    }
    
    // Avoid backtracking if possible
    if let Some(last_dir) = last_direction {
        let opposite = opposite_direction(last_dir);
        safe_directions.retain(|&dir| dir != opposite);
        
        // If we filtered out all directions, add them back
        if safe_directions.is_empty() {
            for (i, direction) in directions.iter().enumerate() {
                let elevation_diff = (context.elevation[i] - context.elevation[8]).abs();
                let is_occupied = context.markers[i].is_some();
                
                if elevation_diff <= 1 && !is_occupied {
                    safe_directions.push(*direction);
                }
            }
        }
    }
    
    if safe_directions.is_empty() {
        None
    } else {
        let index = rng.next_range(0..safe_directions.len());
        Some(safe_directions[index])
    }
}

#[plugin_fn]
pub fn mind_function(input: Vec<u8>) -> FnResult<Vec<u8>> {
    let (blob_state, cell_context, seed) = capnp_to_mind_input(&input)?;
    
    // Improve seed diversity by combining multiple entropy sources
    // Include marker value, memory state, and context-based entropy for better randomization
    let entropy_boost = blob_state.marker as u64 
        ^ (blob_state.age as u64 * 31)
        ^ (blob_state.energy as u64 * 17)
        ^ (blob_state.memory[0] as u64 * 13)
        ^ (cell_context.elevation[8] as u64 * 19)
        ^ (cell_context.energy[8] as u64 * 23);
    
    let enhanced_seed = seed.wrapping_mul(0x9e3779b97f4a7c15u64).wrapping_add(entropy_boost);
    let mut rng = StdRand::seed(enhanced_seed);
    
    let last_direction = get_last_direction(&blob_state);
    let mut new_memory = blob_state.memory;
    
    // Priority 1: Mark current square as visited if not already
    if !is_visited(&cell_context) {
        let capnp_vec = mind_output_to_capnp(&CellAction::SetPheromone(VISITED_PHEROMONE), &new_memory)
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 2: If there's energy here and we're not at max energy, eat it
    if has_energy_here(&cell_context) && blob_state.energy < blob_state.max_energy {
        let capnp_vec = mind_output_to_capnp(&CellAction::Eat, &new_memory)
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 3: If at max energy and there's energy here (plant), split
    if blob_state.energy >= blob_state.max_energy && has_energy_here(&cell_context) {
        // Find a safe direction to split
        if let Some(direction) = find_random_safe_direction(&cell_context, &mut rng, last_direction) {
            let split_energy = blob_state.energy / 2;
            let capnp_vec = mind_output_to_capnp(&CellAction::Split(
                direction,
                split_energy as i32,
                blob_state.marker,
                new_memory,
            ), &new_memory).map_err(|e| Error::msg(e.to_string()))?;
            return Ok(capnp_vec);
        }
    }
    
    // Priority 4: Attack enemies (blobs with unrecognized markers)
    if let Some(direction) = find_enemy_direction(&cell_context, blob_state.marker) {
        let capnp_vec = mind_output_to_capnp(&CellAction::Attack(direction), &new_memory)
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 5: Move to ANY adjacent energy (greedy food seeking)
    if let Some(direction) = find_any_energy_direction(&cell_context, &mut rng) {
        set_last_direction(&mut new_memory, Some(direction));
        let capnp_vec = mind_output_to_capnp(&CellAction::Move(direction), &new_memory)
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 6: Move to any unvisited square
    if let Some(direction) = find_unvisited_direction(&cell_context, &mut rng, last_direction, &blob_state) {
        set_last_direction(&mut new_memory, Some(direction));
        let capnp_vec = mind_output_to_capnp(&CellAction::Move(direction), &new_memory)
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 7: All squares visited, move randomly (avoid backtracking)
    if let Some(direction) = find_random_safe_direction(&cell_context, &mut rng, last_direction) {
        set_last_direction(&mut new_memory, Some(direction));
        let capnp_vec = mind_output_to_capnp(&CellAction::Move(direction), &new_memory)
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Fallback: Do nothing
    let capnp_vec = mind_output_to_capnp(&CellAction::DoNothing, &new_memory)
        .map_err(|e| Error::msg(e.to_string()))?;
    Ok(capnp_vec)
}
