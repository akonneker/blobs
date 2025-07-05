use crate::cell::CellAction;
use crate::types::{Direction, CellMessage};
use crate::mind_output_capnp::{mind_output, action, Direction as CapnpDirection};
use capnp::serialize;

// Note: The Cap'n Proto Direction enum doesn't have a 'Center' variant.
// We handle this by panicking if we encounter it during conversion.
fn to_capnp_direction(dir: Direction) -> CapnpDirection {
    match dir {
        Direction::North => CapnpDirection::North,
        Direction::South => CapnpDirection::South,
        Direction::East => CapnpDirection::East,
        Direction::West => CapnpDirection::West,
        Direction::NorthEast => CapnpDirection::NorthEast,
        Direction::NorthWest => CapnpDirection::NorthWest,
        Direction::SouthEast => CapnpDirection::SouthEast,
        Direction::SouthWest => CapnpDirection::SouthWest,
    }
}

fn from_capnp_direction(dir: CapnpDirection) -> Direction {
    match dir {
        CapnpDirection::North => Direction::North,
        CapnpDirection::South => Direction::South,
        CapnpDirection::East => Direction::East,
        CapnpDirection::West => Direction::West,
        CapnpDirection::NorthEast => Direction::NorthEast,
        CapnpDirection::NorthWest => Direction::NorthWest,
        CapnpDirection::SouthEast => Direction::SouthEast,
        CapnpDirection::SouthWest => Direction::SouthWest,
    }
}

pub fn mind_output_to_capnp(cell_action: &CellAction, memory: &[u8; 2048]) -> capnp::Result<Vec<u8>> {
    let mut message = ::capnp::message::Builder::new_default();
    let mut output_builder = message.init_root::<mind_output::Builder>();
    
    // Set the memory
    output_builder.set_memory(memory);
    
    // Set the action
    let mut action_builder = output_builder.init_action();
    
    match cell_action {
        CellAction::SendMessage(dir, msg) => {
            let mut sm_builder = action_builder.reborrow().init_send_message();
            sm_builder.set_direction(to_capnp_direction(*dir));
            sm_builder.set_contents(&msg.data);
        }
        CellAction::Defend => {
            action_builder.set_defend(());
        }
        CellAction::Attack(dir) => {
            action_builder.set_attack(to_capnp_direction(*dir));
        }
        CellAction::LiftTerrain => {
            action_builder.set_lift_terrain(());
        }
        CellAction::DumpTerrain => {
            action_builder.set_dump_terrain(());
        }
        CellAction::SetPheromone(pheromone) => {
            action_builder.set_set_pheromone(*pheromone);
        }
        CellAction::Move(dir) => {
            action_builder.set_move(to_capnp_direction(*dir));
        }
        CellAction::Split(dir, energy, marker, split_memory) => {
            let mut sd_builder = action_builder.reborrow().init_split();
            sd_builder.set_direction(to_capnp_direction(*dir));
            sd_builder.set_energy(*energy as u32);
            sd_builder.set_marker(*marker);
            sd_builder.set_memory(split_memory);
        }
        CellAction::Eat => {
            action_builder.set_eat(());
        }
        CellAction::DoNothing => {
            action_builder.set_do_nothing(());
        }
    }

    let mut a = Vec::new();
    serialize::write_message(&mut a, &message)?;
    Ok(a)
}

pub fn capnp_to_mind_output(data: &[u8]) -> capnp::Result<(CellAction, [u8; 2048])> {
    let message_reader = serialize::read_message(&mut data.as_ref(), ::capnp::message::ReaderOptions::new())?;
    let output_reader = message_reader.get_root::<mind_output::Reader>()?;
    
    // Extract memory
    let memory_reader = output_reader.get_memory()?;
    let mut memory = [0u8; 2048];
    let len = std::cmp::min(memory_reader.len(), 2048);
    memory[..len].copy_from_slice(&memory_reader[..len]);
    
    // Extract action
    let action_reader = output_reader.get_action()?;
    let cell_action = match action_reader.which()? {
        action::Which::SendMessage(reader) => {
            let msg_reader = reader?;
            let dir = from_capnp_direction(msg_reader.get_direction()?);
            let contents = msg_reader.get_contents()?;
            let msg_data = contents.to_vec();
            CellAction::SendMessage(dir, CellMessage { data: msg_data })
        }
        action::Which::Defend(()) => CellAction::Defend,
        action::Which::Attack(dir) => CellAction::Attack(from_capnp_direction(dir?)),
        action::Which::LiftTerrain(()) => {
            CellAction::LiftTerrain
        }
        action::Which::DumpTerrain(()) => {
            CellAction::DumpTerrain
        }
        action::Which::SetPheromone(pheromone) => {
            CellAction::SetPheromone(pheromone)
        }
        action::Which::Move(dir) => CellAction::Move(from_capnp_direction(dir?)),
        action::Which::Split(reader) => {
            let sd_reader = reader?;
            let dir = from_capnp_direction(sd_reader.get_direction()?);
            let energy = sd_reader.get_energy() as i32;
            let marker = sd_reader.get_marker();
            let memory_reader = sd_reader.get_memory()?;
            let mut split_memory = [0u8; 2048];
            let len = std::cmp::min(memory_reader.len(), 2048);
            split_memory[..len].copy_from_slice(&memory_reader[..len]);
            CellAction::Split(dir, energy, marker, split_memory)
        }
        action::Which::Eat(()) => {
            CellAction::Eat
        }
        action::Which::DoNothing(()) => CellAction::DoNothing,
    };

    Ok((cell_action, memory))
} 