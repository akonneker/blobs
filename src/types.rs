use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: usize,
    pub y: usize,
    pub z: usize, // Elevation
}

pub type Pheromone = u32;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Direction {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
    Center,
}

impl Direction {
    pub fn index(&self) -> usize {
        match self {
            Direction::NorthWest => 0,
            Direction::North => 1,
            Direction::NorthEast => 2,
            Direction::West => 3,
            Direction::Center => 4,
            Direction::East => 5,
            Direction::SouthWest => 6,
            Direction::South => 7,
            Direction::SouthEast => 8,
        }
    }

    pub fn all() -> [Direction; 9] {
        [
            Direction::NorthWest,
            Direction::North,
            Direction::NorthEast,
            Direction::West,
            Direction::Center,
            Direction::East,
            Direction::SouthWest,
            Direction::South,
            Direction::SouthEast,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellMessage {
    pub data: [u8; 512],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct TeamId(pub usize);


#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CellId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, Deserialize)]
pub enum EnergyDistribution {
    Constant(u32),
    Uniform(u32, u32),
    PseudoNormal(u32, u32),
    Binomial(u32, f32),
}

// ---- Rhai Export Modules ----
pub mod rhai_exports {
    use super::{Direction, CellMessage, CellAction, Pheromone};
    use rhai::plugin::*;

    #[export_module]
    pub mod rhai_direction_module {
        pub fn north() -> Direction { Direction::North }
        pub fn south() -> Direction { Direction::South }
        pub fn east() -> Direction { Direction::East }
        pub fn west() -> Direction { Direction::West }
        pub fn north_east() -> Direction { Direction::NorthEast }
        pub fn north_west() -> Direction { Direction::NorthWest }
        pub fn south_east() -> Direction { Direction::SouthEast }
        pub fn south_west() -> Direction { Direction::SouthWest }
        pub fn center() -> Direction { Direction::Center }
    }

    #[export_module]
    pub mod rhai_cell_action_module {
        // For SendMessage, Rhai script will call new_cell_message from main.rs context
        // then pass it here.
        pub fn send_message(dir: Direction, msg: CellMessage) -> CellAction {
            CellAction::SendMessage(dir, msg)
        }
        pub fn defend() -> CellAction { CellAction::Defend }
        pub fn attack(dir: Direction) -> CellAction { CellAction::Attack(dir) }
        pub fn lift_terrain() -> CellAction { CellAction::LiftTerrain }
        pub fn dump_terrain() -> CellAction { CellAction::DumpTerrain }
        pub fn set_pheromone(value: Pheromone) -> CellAction { CellAction::SetPheromone(value) }
        pub fn move_action(dir: Direction) -> CellAction { CellAction::Move(dir) } // Renamed to avoid conflict with 'move' keyword
        
        // For Split, Rhai script will create an array for memory.
        // We need a helper in main.rs to convert Rhai array to [u8; 2048]
        // or accept Vec<u8> here and convert.
        // Let's assume a helper function `new_memory_array` will be registered in main.rs
        // that takes a rhai::Array and returns [u8; 2048], or this function takes Vec<u8>.
        // For simplicity, let's say script calls a registered Rust fn `create_split_action`
        // Alternatively, if CellAction::Split took Vec<u8> it would be simpler here.
        // Given current CellAction::Split([u8;2048]), this is tricky for direct module export.
        // It's often easier to register global Rust functions that construct complex enum variants.
        // So, I will OMIT Split from here, and it should be constructed via a global registered function in main.rs
        pub fn eat() -> CellAction { CellAction::Eat }
        pub fn do_nothing() -> CellAction { CellAction::DoNothing }
    }
}