use crate::types::{Coordinate, Pheromone, Direction};
use serde::{Serialize, Deserialize};


#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergySource {
    Scattered(u32), // Amount of energy
    Plant { rate: u32, current_energy: u32, max_energy: u32 }, // Rate of production, current and max energy
}

/*
 0 1 2
 3 8 4
 5 6 7
*/


pub struct Neighborhood {
    pub elevation: [i32; 9],
    pub energy: [u32; 9],
    pub pheromone: [Option<Pheromone>; 9],
}

impl Neighborhood {
    pub fn new() -> Self {
        Neighborhood {
            elevation: [0; 9],
            energy: [0; 9],
            pheromone: [None; 9],
        }

    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct World {
    pub dimensions: (usize, usize), // Width and height
    pub elevation: Vec<i32>,
    pub energy: Vec<Option<EnergySource>>,
    pub pheromone: Vec<Option<Pheromone>>,
}




pub fn relative_position(position: Coordinate, direction: Option<Direction>, dimensions: (usize, usize)) -> Coordinate {
    let (width, height) = dimensions;
    match direction {
        Some(Direction::North) => Coordinate { 
            x: position.x, 
            y: if position.y == 0 { height - 1 } else { position.y - 1 }
        },
        Some(Direction::South) => Coordinate { 
            x: position.x, 
            y: (position.y + 1) % height 
        },
        Some(Direction::East) => Coordinate { 
            x: (position.x + 1) % width, 
            y: position.y 
        },
        Some(Direction::West) => Coordinate { 
            x: if position.x == 0 { width - 1 } else { position.x - 1 }, 
            y: position.y 
        },
        Some(Direction::NorthEast) => Coordinate { 
            x: (position.x + 1) % width,
            y: if position.y == 0 { height - 1 } else { position.y - 1 }
        },
        Some(Direction::NorthWest) => Coordinate { 
            x: if position.x == 0 { width - 1 } else { position.x - 1 },
            y: if position.y == 0 { height - 1 } else { position.y - 1 }
        },
        Some(Direction::SouthEast) => Coordinate { 
            x: (position.x + 1) % width,
            y: (position.y + 1) % height
        },
        Some(Direction::SouthWest) => Coordinate { 
            x: if position.x == 0 { width - 1 } else { position.x - 1 },
            y: (position.y + 1) % height
        },
        None => position,
    }
}

// pub fn relative_distance(position: Coordinate, other: Coordinate, bounds: (usize, usize)) -> u32 {
//     let (width, height) = bounds;
    
//     // Calculate x distance considering wrap-around
//     let x_diff_direct = (position.x as i32 - other.x as i32).abs();
//     let x_diff_wrap = (width as i32) - x_diff_direct;
//     let x_diff = x_diff_direct.min(x_diff_wrap);

//     // Calculate y distance considering wrap-around
//     let y_diff_direct = (position.y as i32 - other.y as i32).abs();
//     let y_diff_wrap = (height as i32) - y_diff_direct;
//     let y_diff = y_diff_direct.min(y_diff_wrap);

//     x_diff.max(y_diff) as u32
// }

impl World {
    pub fn new(width: usize, height: usize) -> Self {
        // Initialize with default tiles
        let elevation = vec![0; width * height];
        let energy = vec![None; width * height];
        let pheromone = vec![None; width * height];
        World {
            dimensions: (width, height),
            elevation,
            energy,
            pheromone,
        }
    }

    pub fn pheromone_at(&self, position: Coordinate) -> Option<Pheromone> {
        let x_wrapped = position.x % self.dimensions.0;
        let y_wrapped = position.y % self.dimensions.1;
        self.pheromone[y_wrapped * self.dimensions.0 + x_wrapped]
    }

    pub fn set_pheromone_at(&mut self, position: Coordinate, pheromone: Option<Pheromone>) {
        let x_wrapped = position.x % self.dimensions.0;
        let y_wrapped = position.y % self.dimensions.1;
        self.pheromone[y_wrapped * self.dimensions.0 + x_wrapped] = pheromone;
    }

    pub fn energy_at(&self, position: Coordinate) -> u32 {
        let x_wrapped = position.x % self.dimensions.0;
        let y_wrapped = position.y % self.dimensions.1;
        match self.energy[y_wrapped * self.dimensions.0 + x_wrapped] {
            Some(EnergySource::Scattered(energy)) => energy,
            Some(EnergySource::Plant { rate: _, current_energy, max_energy: _ }) => current_energy,
            None => 0,
        }
    }
    
    pub fn set_energy_at(&mut self, position: Coordinate, energy: Option<EnergySource>) {
        let x_wrapped = position.x % self.dimensions.0;
        let y_wrapped = position.y % self.dimensions.1;
        self.energy[y_wrapped * self.dimensions.0 + x_wrapped] = energy; 
    }

    pub fn elevation_at(&self, position: Coordinate) -> i32 {
        let x_wrapped = position.x % self.dimensions.0;
        let y_wrapped = position.y % self.dimensions.1;
        self.elevation[y_wrapped * self.dimensions.0 + x_wrapped]
    }
    
    pub fn set_elevation_at(&mut self, position: Coordinate, elevation: i32) {
        let x_wrapped = position.x % self.dimensions.0;
        let y_wrapped = position.y % self.dimensions.1;
        self.elevation[y_wrapped * self.dimensions.0 + x_wrapped] = elevation;
    }

    pub fn get_neighborhood(&self, position: Coordinate) -> Neighborhood {
        let mut neighborhood = Neighborhood::new();
        for direction in Direction::all() {
            let neighbor_position = relative_position(position, Some(direction), self.dimensions);
            neighborhood.elevation[direction.index()] = self.elevation_at(neighbor_position);
            neighborhood.energy[direction.index()] = self.energy_at(neighbor_position);
            neighborhood.pheromone[direction.index()] = self.pheromone_at(neighbor_position);
        }
        neighborhood.elevation[8] = self.elevation_at(position);
        neighborhood.energy[8] = self.energy_at(position);
        neighborhood.pheromone[8] = self.pheromone_at(position);
        neighborhood
    }
} 