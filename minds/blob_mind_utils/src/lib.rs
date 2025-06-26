use blob_interface::cell::{BlobState, CellAction, CellContext};
use blob_interface::types::{Direction, Pheromone};
use tinyrand::{Rand, StdRand, RandRange, Seeded};

// Shared utility functions that all minds can use

/// Find the direction with the most energy in the neighborhood
pub fn find_best_energy_direction(context: &CellContext) -> Option<Direction> {
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

/// Check if there's energy available in the current position
pub fn has_energy_here(context: &CellContext) -> bool {
    context.energy[8] > 0 // Center position
}

/// Check if there's energy in any neighboring position
pub fn has_energy_nearby(context: &CellContext) -> bool {
    context.energy[0..8].iter().any(|&energy| energy > 0)
}

/// Find a safe direction to move (no steep elevation changes)
pub fn find_safe_move_direction(context: &CellContext, rng: &mut StdRand) -> Option<Direction> {
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

/// Check if a position is relatively high (good for visibility)
pub fn is_high_ground(context: &CellContext) -> bool {
    let center_elevation = context.elevation[8];
    let avg_neighbor_elevation: i32 = context.elevation[0..8].iter().sum::<i32>() / 8;
    center_elevation > avg_neighbor_elevation
}

/// Check if we're on a plant (high energy at current position)
pub fn is_on_plant(context: &CellContext) -> bool {
    context.energy[8] > 50 // Assume plants have more energy
}

/// Find direction with lowest elevation (for exploration)
pub fn find_lowest_elevation_direction(context: &CellContext) -> Option<Direction> {
    let directions = Direction::all();
    let mut best_direction = None;
    let mut lowest_elevation = i32::MAX;
    
    for (i, direction) in directions.iter().enumerate() {
        if context.elevation[i] < lowest_elevation {
            lowest_elevation = context.elevation[i];
            best_direction = Some(*direction);
        }
    }
    
    best_direction
}

/// Calculate Manhattan distance-like heuristic for pathfinding
pub fn estimate_distance_to_coordinate(from_x: usize, from_y: usize, to_x: usize, to_y: usize, world_width: usize, world_height: usize) -> u32 {
    let dx = if from_x > to_x { from_x - to_x } else { to_x - from_x };
    let dy = if from_y > to_y { from_y - to_y } else { to_y - from_y };
    
    // Consider wrapping around the world
    let dx_wrapped = world_width - dx;
    let dy_wrapped = world_height - dy;
    
    let best_dx = if dx < dx_wrapped { dx } else { dx_wrapped };
    let best_dy = if dy < dy_wrapped { dy } else { dy_wrapped };
    
    (best_dx + best_dy) as u32
}

/// Fisher-Yates shuffle implementation using tinyrand
pub fn shuffle_actions<R: Rand + RandRange<usize>>(
    actions: &mut Vec<fn(&BlobState, &CellContext) -> Option<CellAction>>,
    rng: &mut R
) {
    for i in (1..actions.len()).rev() {
        let j = rng.next_range(0..i+1);
        actions.swap(i, j);
    }
}

/// Simple strategy functions for basic behaviors
pub fn try_send_message(_blob_state: &BlobState, _context: &CellContext) -> Option<CellAction> {
    None
}

pub fn try_defend(blob_state: &BlobState, _context: &CellContext) -> Option<CellAction> {
    if blob_state.energy < 50 {
        Some(CellAction::Defend)
    } else {
        None
    }
}

pub fn try_attack(_blob_state: &BlobState, _context: &CellContext) -> Option<CellAction> {
    None
}

pub fn try_lift_terrain(_blob_state: &BlobState, _context: &CellContext) -> Option<CellAction> {
    None
}

pub fn try_dump_terrain(_blob_state: &BlobState, _context: &CellContext) -> Option<CellAction> {
    None
}

pub fn try_set_pheromone(_blob_state: &BlobState, _context: &CellContext) -> Option<CellAction> {
    None
}

pub fn try_move(blob_state: &BlobState, context: &CellContext) -> Option<CellAction> {
    if blob_state.energy > 20 {
        let mut rng = StdRand::seed(blob_state.age as u64);
        if let Some(direction) = find_safe_move_direction(context, &mut rng) {
            Some(CellAction::Move(direction))
        } else {
            None
        }
    } else {
        None
    }
}

pub fn try_split(_blob_state: &BlobState, _context: &CellContext) -> Option<CellAction> {
    None
}

pub fn try_eat(_blob_state: &BlobState, context: &CellContext) -> Option<CellAction> {
    if has_energy_here(context) {
        Some(CellAction::Eat)
    } else {
        None
    }
}
