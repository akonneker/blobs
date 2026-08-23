use crate::types::{CellId, CellMessage, TeamId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    pub id: CellId,
    pub team_id: TeamId,
    pub energy: u32,
    pub min_energy: u32, // Cost to exist, dropped on death
    pub marker: u32,
    pub loaded: bool,
    pub age: u32,
    pub memory: Vec<u8>,
    pub message_queue: Vec<CellMessage>,
    pub defending: bool,
    /// Engine-private random lineage. Never serialized into a mind invocation.
    pub random_lineage: u64,
    /// Number of decisions already requested from this cell.
    pub decision_sequence: u64,
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
            memory: vec![0; 2048],
            message_queue: Vec::new(),
            defending: false,
            random_lineage: id.0 as u64,
            decision_sequence: 0,
        }
    }

    /// Get memory as a fixed-size array for fixed-size host consumers.
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
