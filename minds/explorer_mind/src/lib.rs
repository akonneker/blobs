use extism_pdk::*;
use tinyrand::{StdRand, Seeded, Rand, RandRange};

use blob_interface::cell::{BlobState, CellAction, CellContext};
use blob_interface::mind_output_converter::mind_output_to_capnp;
use blob_interface::mind_input_converter::capnp_to_mind_input;
use blob_interface::types::{Direction, Pheromone};

// Constants for the explorer mind
const PLANT_PHEROMONE: Pheromone = 100;
const TRAIL_PHEROMONE: Pheromone = 50;
const EXPLORATION_PHEROMONE: Pheromone = 25;
const MIN_ENERGY_TO_EXPLORE: u32 = 30;
const SPLIT_ENERGY_AMOUNT: i32 = 80;

// Memory layout (using first 16 bytes of memory)
// Bytes 0-1: Number of plants found (u16)
// Bytes 2-3: Current exploration target X (u16) 
// Bytes 4-5: Current exploration target Y (u16)
// Bytes 6-7: Last plant X found (u16)
// Bytes 8-9: Last plant Y found (u16)
// Bytes 10-11: Exploration mode (u16: 0=random, 1=systematic, 2=trail_making)

// Helper functions
fn read_u16_from_memory(memory: &[u8], offset: usize) -> u16 {
    if offset + 1 < memory.len() {
        u16::from_le_bytes([memory[offset], memory[offset + 1]])
    } else {
        0
    }
}

fn write_u16_to_memory(memory: &mut [u8], offset: usize, value: u16) {
    if offset + 1 < memory.len() {
        let bytes = value.to_le_bytes();
        memory[offset] = bytes[0];
        memory[offset + 1] = bytes[1];
    }
}

fn has_energy_here(context: &CellContext) -> bool {
    context.energy[8] > 0
}

fn is_on_plant(context: &CellContext) -> bool {
    context.energy[8] > 50 // Assume plants have more energy
}

fn find_best_energy_direction(context: &CellContext) -> Option<Direction> {
    let directions = Direction::all();
    let mut best_direction = None;
    let mut best_energy = 0u32;
    
    for (i, direction) in directions.iter().enumerate() {
        if context.energy[i] > best_energy {
            best_energy = context.energy[i];
            best_direction = Some(*direction);
        }
    }
    
    best_direction
}

fn find_safe_move_direction(context: &CellContext, rng: &mut StdRand) -> Option<Direction> {
    let directions = Direction::all();
    let center_elevation = context.elevation[8];
    let mut safe_directions = Vec::new();
    
    for (i, direction) in directions.iter().enumerate() {
        let elevation_diff = (context.elevation[i] - center_elevation).abs();
        if elevation_diff <= 1 {
            safe_directions.push(*direction);
        }
    }
    
    if safe_directions.is_empty() {
        None
    } else {
        let index = rng.next_range(0..safe_directions.len());
        Some(safe_directions[index])
    }
}

struct ExplorerMemory {
    plants_found: u16,
    target_x: u16,
    target_y: u16,
    last_plant_x: u16,
    last_plant_y: u16,
    exploration_mode: u16,
}

impl ExplorerMemory {
    fn from_blob_memory(memory: &[u8]) -> Self {
        ExplorerMemory {
            plants_found: read_u16_from_memory(memory, 0),
            target_x: read_u16_from_memory(memory, 2),
            target_y: read_u16_from_memory(memory, 4),
            last_plant_x: read_u16_from_memory(memory, 6),
            last_plant_y: read_u16_from_memory(memory, 8),
            exploration_mode: read_u16_from_memory(memory, 10),
        }
    }
    
    fn write_to_blob_memory(&self, memory: &mut [u8]) {
        write_u16_to_memory(memory, 0, self.plants_found);
        write_u16_to_memory(memory, 2, self.target_x);
        write_u16_to_memory(memory, 4, self.target_y);
        write_u16_to_memory(memory, 6, self.last_plant_x);
        write_u16_to_memory(memory, 8, self.last_plant_y);
        write_u16_to_memory(memory, 10, self.exploration_mode);
    }
}

fn explorer_strategy(blob_state: &BlobState, context: &CellContext, seed: u64) -> (CellAction, [u8; 2048]) {
    let mut rng = StdRand::seed(seed ^ blob_state.age as u64);
    let mut memory = ExplorerMemory::from_blob_memory(&blob_state.memory);
    
    // Calculate dynamic threshold based on max_energy
    let min_energy_to_split = (blob_state.max_energy * 4) / 10; // 40% of max energy
    
    // Priority 1: If we're on a plant, mark it and set pheromone
    if is_on_plant(context) {
        memory.plants_found = memory.plants_found.saturating_add(1);
        memory.last_plant_x = (blob_state.age % 50) as u16; // Rough position estimate
        memory.last_plant_y = (blob_state.age / 50) as u16;
        
        let mut updated_memory = blob_state.memory;
        memory.write_to_blob_memory(&mut updated_memory);
        return (CellAction::SetPheromone(PLANT_PHEROMONE), updated_memory);
    }
    
    // Priority 2: Eat if there's energy here and we're not at max
    if has_energy_here(context) && blob_state.energy < blob_state.max_energy {
        let mut updated_memory = blob_state.memory;
        memory.write_to_blob_memory(&mut updated_memory);
        return (CellAction::Eat, updated_memory);
    }
    
    // Priority 3: If we have high energy and found plants, consider splitting
    if blob_state.energy > min_energy_to_split && memory.plants_found > 0 {
        if let Some(direction) = find_safe_move_direction(context, &mut rng) {
            // Create child with exploration knowledge
            let mut child_memory = [0u8; 2048];
            memory.exploration_mode = 1; // Set child to systematic exploration
            memory.write_to_blob_memory(&mut child_memory);
            
            let mut updated_memory = blob_state.memory;
            memory.write_to_blob_memory(&mut updated_memory);
            return (CellAction::Split(direction, SPLIT_ENERGY_AMOUNT, 1, child_memory), updated_memory);
        }
    }
    
    // Priority 4: If we have low energy, try to find energy
    if blob_state.energy < MIN_ENERGY_TO_EXPLORE {
        // Move towards nearby energy
        if let Some(direction) = find_best_energy_direction(context) {
            let mut updated_memory = blob_state.memory;
            memory.write_to_blob_memory(&mut updated_memory);
            return (CellAction::Move(direction), updated_memory);
        }
    }
    
    // Priority 5: Exploration and trail making
    let action = match memory.exploration_mode {
        0 => random_exploration(context, &mut rng), // Random exploration
        1 => systematic_exploration(context, &mut rng, &memory), // Systematic exploration
        2 => trail_making(context, &mut rng, &memory), // Trail making between plants
        _ => random_exploration(context, &mut rng),
    };
    
    let mut updated_memory = blob_state.memory;
    memory.write_to_blob_memory(&mut updated_memory);
    (action, updated_memory)
}

fn random_exploration(context: &CellContext, rng: &mut StdRand) -> CellAction {
    // Place exploration pheromone occasionally
    if rng.next_range(0usize..10) == 0 {
        return CellAction::SetPheromone(EXPLORATION_PHEROMONE);
    }
    
    // Move to unexplored areas (avoid existing pheromones)
    let directions = Direction::all();
    let mut unexplored_directions = Vec::new();
    
    for (i, direction) in directions.iter().enumerate() {
        if context.pheromone[i].is_none() {
            unexplored_directions.push(*direction);
        }
    }
    
    if !unexplored_directions.is_empty() {
        let index = rng.next_range(0..unexplored_directions.len());
        return CellAction::Move(unexplored_directions[index]);
    }
    
    // If everywhere has pheromones, just move safely
    if let Some(direction) = find_safe_move_direction(context, rng) {
        CellAction::Move(direction)
    } else {
        CellAction::DoNothing
    }
}

fn systematic_exploration(context: &CellContext, rng: &mut StdRand, _memory: &ExplorerMemory) -> CellAction {
    // Try to move in a systematic pattern (prefer consistent directions)
    let preferred_directions = [Direction::North, Direction::East, Direction::South, Direction::West];
    
    for direction in preferred_directions.iter() {
        let dir_index = direction.index();
        // Check if this direction is safe and has low/no pheromone
        let elevation_diff = (context.elevation[dir_index] - context.elevation[8]).abs();
        if elevation_diff <= 1 && context.pheromone[dir_index].unwrap_or(0) < TRAIL_PHEROMONE {
            return CellAction::Move(*direction);
        }
    }
    
    // Fallback to random exploration
    random_exploration(context, rng)
}

fn trail_making(context: &CellContext, rng: &mut StdRand, memory: &ExplorerMemory) -> CellAction {
    // If we found plants, create trails between them
    if memory.plants_found > 1 {
        // Place trail pheromone
        if rng.next_range(0usize..3) == 0 {
            return CellAction::SetPheromone(TRAIL_PHEROMONE);
        }
        
        // Move towards areas with plant pheromones
        let directions = Direction::all();
        for (i, direction) in directions.iter().enumerate() {
            if let Some(pheromone) = context.pheromone[i] {
                if pheromone >= PLANT_PHEROMONE {
                    return CellAction::Move(*direction);
                }
            }
        }
    }
    
    // Continue systematic exploration
    systematic_exploration(context, rng, memory)
}

#[plugin_fn]
pub fn mind_function(input: Vec<u8>) -> FnResult<Vec<u8>> {
    let (blob_state, cell_context, seed) = capnp_to_mind_input(&input)?;
    
    let (action, updated_memory) = explorer_strategy(&blob_state, &cell_context, seed);
    
    let capnp_vec = mind_output_to_capnp(&action, &updated_memory).map_err(|e| Error::msg(e.to_string()))?;
    Ok(capnp_vec)
}
