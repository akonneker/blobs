use crate::types::{CellId, TeamId, CellMessage, Pheromone, Direction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub id: CellId,
    pub team_id: TeamId,
    pub energy: u32, 
    pub min_energy: u32, // Cost to exist, dropped on death
    pub max_energy: u32, // Maximum energy this cell can hold
    pub marker: u32,
    pub loaded: bool,
    pub age: u32,
    pub memory: Vec<u8>,
    pub message_queue: Vec<CellMessage>,
    pub defending: bool,
    // Add other cell properties like senses, memory, etc.
}

impl Cell {
    pub fn new(id: CellId, team_id: TeamId, initial_energy: u32, min_energy: u32, max_energy: u32) -> Self {
        Cell {
            id,
            team_id,
            energy: initial_energy,
            min_energy,
            max_energy,
            marker: 0,
            loaded: false,
            age: 0,
            memory: vec![0; 2048],
            message_queue: Vec::new(),
            defending: false,
        }
    }
    
    /// Get memory as a fixed-size array for compatibility with existing code
    pub fn memory_array(&self) -> [u8; 2048] {
        let mut array = [0u8; 2048];
        let len = self.memory.len().min(2048);
        array[..len].copy_from_slice(&self.memory[..len]);
        array
    }
    
    /// Set memory from a fixed-size array
    pub fn set_memory_array(&mut self, memory: [u8; 2048]) {
        self.memory = memory.to_vec();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CellContext {
    pub elevation: [i32; 9],
    pub energy: [u32; 9],
    pub pheromone: [Option<Pheromone>; 9],
    pub markers: [Option<u32>; 8],
}

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

#[derive(Debug, Clone)]
pub struct BlobState {
    pub energy: u32,
    pub min_energy: u32,
    pub max_energy: u32,
    pub marker: u32,
    pub loaded: bool,
    pub age: u32,
    pub memory: [u8; 2048],
    pub message_queue: Vec<CellMessage>,
}

