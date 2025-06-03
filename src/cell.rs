use crate::types::{CellId, TeamId, CellMessage};

#[derive(Debug, Clone)]
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

impl Cell {
    pub fn new(id: CellId, team_id: TeamId, initial_energy: u32, min_energy: u32) -> Self {
        Cell {
            id,
            team_id,
            energy: initial_energy,
            min_energy,
            marker: 0,
            loaded: false,
            age: 0,
            memory: [0; 2048],
            message_queue: Vec::new(),
            defending: false,
        }
    }
} 