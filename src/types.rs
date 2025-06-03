use rune::Any;
use rune::alloc::clone::TryClone;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Any)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Any)]
pub struct Position {
    pub x: usize,
    pub y: usize,
    pub z: usize, // Elevation
}

pub type Pheromone = u32;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Any, TryClone)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Any, TryClone)]
pub struct CellMessage {
    #[rune(get)]
    pub data: [u8; 512],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Any, TryClone)]
pub struct TeamId(pub usize);


#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Any, TryClone)]
pub struct CellId(pub usize);

#[derive(Debug, Clone, Any, PartialEq, Eq)]
pub enum CellAction {
    SendMessage(Direction, CellMessage),
    Defend,
    Attack(Direction),
    LiftTerrain,
    DumpTerrain,
    SetPheromone(Pheromone),
    Move(Direction),
    Split(Direction, i32, u32, [u8; 2048]),
    DoNothing,
}

#[derive(Debug, Clone, Deserialize)] // Added Clone
pub enum EnergyDistribution {
    Constant(u32),
    Uniform(u32, u32),       // Min, Max
    PseudoNormal(u32, u32),  // Mean, StdDev
    Binomial(u32, f32),      // N, P
}