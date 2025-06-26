use extism_pdk::*;
use tinyrand::{StdRand, Seeded, Rand, RandRange};

use blob_interface::cell::{BlobState, CellAction, CellContext};
use blob_interface::action_converter::cell_action_to_capnp;
use blob_interface::mind_input_converter::capnp_to_mind_input;
use blob_interface::types::Direction;

// Helper functions
fn has_energy_here(context: &CellContext) -> bool {
    context.energy[8] > 0
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

// Strategy functions
fn try_defend(blob_state: &BlobState, _context: &CellContext) -> Option<CellAction> {
    if blob_state.energy < 50 {
        Some(CellAction::Defend)
    } else {
        None
    }
}

fn try_eat(_blob_state: &BlobState, context: &CellContext) -> Option<CellAction> {
    if has_energy_here(context) {
        Some(CellAction::Eat)
    } else {
        None
    }
}

fn try_move(blob_state: &BlobState, context: &CellContext) -> Option<CellAction> {
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

#[plugin_fn]
pub fn mind_function(input: Vec<u8>) -> FnResult<Vec<u8>> {
    let (blob_state, cell_context, seed) = capnp_to_mind_input(&input)?;
    
    let mut actions: Vec<fn(&BlobState, &CellContext) -> Option<CellAction>> = vec![
        try_defend,
        try_eat,
        try_move,
    ];

    let mut rand = StdRand::seed(seed);
    
    // Randomize the action order
    for i in (1..actions.len()).rev() {
        let j = rand.next_range(0..i+1);
        actions.swap(i, j);
    }

    let mut output_action = CellAction::DoNothing;
    for action_fn in actions {
        if let Some(action) = action_fn(&blob_state, &cell_context) {
            output_action = action;
            break;
        }
    }

    let capnp_vec = cell_action_to_capnp(&output_action).map_err(|e| Error::msg(e.to_string()))?;
    Ok(capnp_vec)
}
