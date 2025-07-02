use extism_pdk::*;
use tinyrand::{StdRand, Seeded, Rand, RandRange};

use blob_interface::cell::{BlobState, CellAction, CellContext};
use blob_interface::action_converter::cell_action_to_capnp;
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
        if direction_val < 8 {
            Some(Direction::all()[direction_val as usize])
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
            Some(dir) => dir.index() as u8,
            None => 255,
        };
    }
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

fn find_unvisited_energy_direction(context: &CellContext) -> Option<Direction> {
    let directions = Direction::all();
    
    for (i, direction) in directions.iter().enumerate() {
        // Check if this direction has energy and is unvisited
        if context.energy[i] > 0 {
            let is_square_visited = match context.pheromone[i] {
                Some(pheromone) => pheromone == VISITED_PHEROMONE,
                None => false,
            };
            
            // Check if the square is unoccupied (no marker means no blob)
            let is_unoccupied = context.markers[i].is_none();
            
            if !is_square_visited && is_unoccupied {
                return Some(*direction);
            }
        }
    }
    
    None
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

fn find_unvisited_direction(context: &CellContext, rng: &mut StdRand, last_direction: Option<Direction>) -> Option<Direction> {
    let directions = Direction::all();
    let mut unvisited_directions = Vec::new();
    
    for (i, direction) in directions.iter().enumerate() {
        let is_square_visited = match context.pheromone[i] {
            Some(pheromone) => pheromone == VISITED_PHEROMONE,
            None => false,
        };
        
        // Check elevation difference is safe (within 1)
        let elevation_diff = (context.elevation[i] - context.elevation[8]).abs();
        
        if !is_square_visited && elevation_diff <= 1 {
            unvisited_directions.push(*direction);
        }
    }
    
    // If we have a last direction, prefer not to backtrack
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
                
                if !is_square_visited && elevation_diff <= 1 {
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
    let mut safe_directions = Vec::new();
    
    for (i, direction) in directions.iter().enumerate() {
        let elevation_diff = (context.elevation[i] - context.elevation[8]).abs();
        if elevation_diff <= 1 {
            safe_directions.push(*direction);
        }
    }
    
    // Prefer not to backtrack if possible
    if let Some(last_dir) = last_direction {
        let opposite = opposite_direction(last_dir);
        safe_directions.retain(|&dir| dir != opposite);
        
        // If we filtered out all directions, add them back
        if safe_directions.is_empty() {
            for (i, direction) in directions.iter().enumerate() {
                let elevation_diff = (context.elevation[i] - context.elevation[8]).abs();
                if elevation_diff <= 1 {
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
    let mut rng = StdRand::seed(seed);
    let last_direction = get_last_direction(&blob_state);
    let mut new_memory = blob_state.memory;
    
    // Priority 1: Mark current square as visited if not already
    if !is_visited(&cell_context) {
        let capnp_vec = cell_action_to_capnp(&CellAction::SetPheromone(VISITED_PHEROMONE))
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 2: If there's energy here and we're not at max energy, eat it
    if has_energy_here(&cell_context) && blob_state.energy < blob_state.max_energy {
        let capnp_vec = cell_action_to_capnp(&CellAction::Eat)
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 3: If at max energy and there's energy here (plant), split
    if blob_state.energy >= blob_state.max_energy && has_energy_here(&cell_context) {
        // Find a safe direction to split
        if let Some(direction) = find_random_safe_direction(&cell_context, &mut rng, last_direction) {
            let split_energy = blob_state.energy / 2;
            let capnp_vec = cell_action_to_capnp(&CellAction::Split(
                direction,
                split_energy as i32,
                blob_state.marker,
                new_memory,
            )).map_err(|e| Error::msg(e.to_string()))?;
            return Ok(capnp_vec);
        }
    }
    
    // Priority 4: Attack enemies (blobs with unrecognized markers)
    if let Some(direction) = find_enemy_direction(&cell_context, blob_state.marker) {
        let capnp_vec = cell_action_to_capnp(&CellAction::Attack(direction))
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 5: Move to unvisited squares with energy
    if let Some(direction) = find_unvisited_energy_direction(&cell_context) {
        set_last_direction(&mut new_memory, Some(direction));
        let capnp_vec = cell_action_to_capnp(&CellAction::Move(direction))
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 6: Move to any unvisited square
    if let Some(direction) = find_unvisited_direction(&cell_context, &mut rng, last_direction) {
        set_last_direction(&mut new_memory, Some(direction));
        let capnp_vec = cell_action_to_capnp(&CellAction::Move(direction))
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Priority 7: All squares visited, move randomly (avoid backtracking)
    if let Some(direction) = find_random_safe_direction(&cell_context, &mut rng, last_direction) {
        set_last_direction(&mut new_memory, Some(direction));
        let capnp_vec = cell_action_to_capnp(&CellAction::Move(direction))
            .map_err(|e| Error::msg(e.to_string()))?;
        return Ok(capnp_vec);
    }
    
    // Fallback: Do nothing
    let capnp_vec = cell_action_to_capnp(&CellAction::DoNothing)
        .map_err(|e| Error::msg(e.to_string()))?;
    Ok(capnp_vec)
}
