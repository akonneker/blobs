/*
INPUT
pub struct Cell {
    pub id: CellId,
    pub team_id: TeamId,
    pub energy: u32, 
    pub min_energy: u32, // Cost to exist, dropped on death
    pub marker: u32,
    pub loaded: bool,
    pub age: u32,
    pub memory: [u8; 2048],
    pub message_queue: Vec<CellMessage>,
    pub defending: bool,
    // Add other cell properties like senses, memory, etc.
}

pub struct CellContext {
    pub elevation: [i32; 9],
    pub energy: [u32; 9],
    pub pheromone: [Option<Pheromone>; 9],
    pub markers: [Option<u32>; 8],
}

OUTPUT

pub enum CellAction {
    SendMessage(Direction, CellMessage),
    Defend,
    Attack(Direction),
    LiftTerrain,
    DumpTerrain,
    SetPheromone(Pheromone),
    Move(Direction),
    Split(Direction, i32, u32, [u8; 2048]),
    Eat,
    DoNothing,
}
pub enum Direction {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
    // Center is a valid direction but usually not for targeted actions
}

pub type Pheromone = u32;

pub struct CellMessage {
    data, // Expected to be [u8; 512]
}
*/

// The `cell` and `context` parameters are passed by the game.
// `cell` provides info like `cell.id`, `cell.energy`, `cell.min_energy`, etc.
// `context` provides info about the cell's surroundings.

fn mind(input_cell, input_context) {
    print(`Cell: ${input_cell}`);
    print(`Context: ${input_context}`);
    let action_idx = rand(0..10); // Now 10 possible actions (0-9)

    // Get a random direction (excluding Center for most actions)
    // Note: Direction module is expected to be registered from Rust
    let directions = [
        Direction::north(), Direction::south(), Direction::east(), Direction::west(),
        Direction::north_east(), Direction::north_west(), Direction::south_east(), Direction::south_west()
    ];
    let dir_idx = rand(directions.len());
    let chosen_direction = directions[dir_idx];

    // Accessing cell properties (assuming registered getters or direct access)
    // let current_energy = cell.energy;
    // let min_required_energy = cell.min_energy;

    print(`Action index: ${action_idx}`);
    
    if action_idx == 0 { // SendMessage
        let message_data = [];
        for _i in 0..512 { message_data.push(0); } // Create an array of 512 zeros
        // Call the registered Rust function to create a CellMessage
        let msg_obj = new_cell_message(message_data);
        return CellAction::send_message(chosen_direction, msg_obj)
    } else if action_idx == 1 { // Defend
        return CellAction::defend()
    } else if action_idx == 2 { // Attack
        return CellAction::attack(chosen_direction)
    } else if action_idx == 3 { // LiftTerrain
        return CellAction::lift_terrain()
    } else if action_idx == 4 { // DumpTerrain
        return CellAction::dump_terrain()
    } else if action_idx == 5 { // SetPheromone
        let pheromone_val = rand(0..1000); // Random u32 value (as i64 in Rhai)
        return CellAction::set_pheromone(pheromone_val)
    } else if action_idx == 6 { // Move
        return CellAction::move_action(chosen_direction) // Uses move_action from the module
    } else if action_idx == 7 { // Split
        let energy_for_child = cell.min_energy + 5; // min_energy is i64 from Rust getter
        let marker = rand(0..256);          // Random u32 marker (as i64 in Rhai)
        let memory_data = [];
        for _i in 0..2048 { memory_data.push(0); } // Create an array of 2048 zeros
        
        // Call the registered global Rust function to create the Split action
        return create_split_action(chosen_direction, energy_for_child, marker, memory_data)
    } else if action_idx == 8 { // Eat Action
        return CellAction::eat()
    } else { // DoNothing (action_idx == 9 or fallback)
        return CellAction::do_nothing()
    }
}
