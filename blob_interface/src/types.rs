use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub x: usize,
    pub y: usize,
    pub z: usize, // Elevation
}

pub type Pheromone = u32;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Direction {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
}

impl Direction {
    pub fn index(&self) -> usize {
        match self {
            Direction::NorthWest => 0,
            Direction::North => 1,
            Direction::NorthEast => 2,
            Direction::West => 3,
            Direction::East => 4,
            Direction::SouthWest => 5,
            Direction::South => 6,
            Direction::SouthEast => 7,
        }
    }

    pub fn all() -> [Direction; 8] {
        [
            Direction::NorthWest,
            Direction::North,
            Direction::NorthEast,
            Direction::West,
            Direction::East,
            Direction::SouthWest,
            Direction::South,
            Direction::SouthEast,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellMessage {
    pub data: Vec<u8>, // Changed from [u8; 512] to Vec<u8>
}

impl CellMessage {
    pub fn new(data: [u8; 512]) -> Self {
        CellMessage { data: data.to_vec() }
    }
    
    pub fn as_array(&self) -> [u8; 512] {
        let mut array = [0u8; 512];
        let len = self.data.len().min(512);
        array[..len].copy_from_slice(&self.data[..len]);
        array
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub struct TeamId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct CellId(pub usize);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum EnergyDistribution {
    Constant(u32),
    Uniform(u32, u32),
    PseudoNormal(u32, u32),
    Binomial(u32, f32),
}