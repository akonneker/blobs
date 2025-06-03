use crate::cell::{Cell};
use crate::world::{World, relative_position, EnergySource};
use crate::types::{Coordinate, CellId, TeamId, CellAction, Pheromone, CellMessage, Direction};
use crate::config::CellConfig;
use std::collections::HashMap;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

use rhai::{Engine, AST, Scope};

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct CellContext {
    pub elevation: [i32; 9],
    pub energy: [u32; 9],
    pub pheromone: [Option<Pheromone>; 9],
    pub markers: [Option<u32>; 8],
}

pub struct Game {
    pub world: World,
    pub teams: HashMap<TeamId, AST>,
    pub cells: HashMap<CellId, Cell>,
    pub coordinate_map: HashMap<Coordinate, CellId>,
    pub inv_coordinate_map: HashMap<CellId, Coordinate>,
    pub iteration: u64,
    pub max_iterations: u64,
    pub cell_config: CellConfig,
    rng: StdRng,
    engine: Engine,
    next_cell_id: usize,
}

#[derive(Debug, Clone)]
pub enum CellInteraction {
    SendMessage(CellId, CellMessage),
    Attack(CellId, u32, CellId),
    Split(CellId, u32, Coordinate, i32, u32, [u8; 2048]),
    SetPheromone(CellId, Coordinate, Pheromone),
    LiftTerrain(CellId, Coordinate),
    DumpTerrain(CellId, Coordinate),
    Move(CellId, u32, Coordinate),
    Eat(CellId, Coordinate),
}

impl CellInteraction {
    fn discriminant(&self) -> i32 {
        match self {
            CellInteraction::SendMessage(..) => 0,
            CellInteraction::Attack(..) => 1,
            CellInteraction::Split(..) => 2,
            CellInteraction::SetPheromone(..) => 3,
            CellInteraction::LiftTerrain(..) => 4,
            CellInteraction::DumpTerrain(..) => 5,
            CellInteraction::Move(..) => 6,
            CellInteraction::Eat(..) => 7,
        }
    }

    fn energy_key(&self) -> u32 {
        match self {
            CellInteraction::Attack(_, energy, _) => *energy,
            CellInteraction::Split(_, energy, _, _, _, _) => *energy,
            CellInteraction::Move(_, energy, _) => *energy,
            _ => 0,
        }
    }
}

impl PartialEq for CellInteraction {
    fn eq(&self, other: &Self) -> bool {
        if self.discriminant() != other.discriminant() || self.energy_key() != other.energy_key() {
            return false;
        }
        // At this point, discriminants and energy_keys are the same.
        // Now compare the actual content of the variants.
        match (self, other) {
            (CellInteraction::SendMessage(s1, sm), CellInteraction::SendMessage(o1, om)) => s1 == o1 && sm == om,
            (CellInteraction::Attack(s1, _, s3), CellInteraction::Attack(o1, _, o3)) => s1 == o1 && s3 == o3,
            (CellInteraction::Split(s1, _, s3, s4, s5, s6), CellInteraction::Split(o1, _, o3, o4, o5, o6)) => 
                s1 == o1 && s3 == o3 && s4 == o4 && s5 == o5 && s6 == o6,
            (CellInteraction::SetPheromone(s1, s2, s_ph), CellInteraction::SetPheromone(o1, o2, o_ph)) => 
                s1 == o1 && s2 == o2 && s_ph == o_ph,
            (CellInteraction::LiftTerrain(s1, s2), CellInteraction::LiftTerrain(o1, o2)) => 
                s1 == o1 && s2 == o2,
            (CellInteraction::DumpTerrain(s1, s2), CellInteraction::DumpTerrain(o1, o2)) => 
                s1 == o1 && s2 == o2,
            (CellInteraction::Move(s1, _, s3), CellInteraction::Move(o1, _, o3)) => s1 == o1 && s3 == o3,
            (CellInteraction::Eat(s1, s2), CellInteraction::Eat(o1, o2)) => s1 == o1 && s2 == o2,
            // This case should ideally not be reached if discriminants are the same and all variants are covered above.
            // However, it acts as a safeguard if a new variant is added and not updated here.
            _ => self.discriminant() == other.discriminant(), // True if same variant, false otherwise (already covered by initial check)
        }
    }
}

impl Eq for CellInteraction {}

impl PartialOrd for CellInteraction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CellInteraction {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.discriminant().cmp(&other.discriminant()) {
            Ordering::Equal => {
                self.energy_key().cmp(&other.energy_key())
            }
            other_ordering => other_ordering,
        }
    }
}

impl Game {
    pub fn new(world_width: usize, world_height: usize, max_iterations: u64, cell_config: CellConfig, seed: Option<u64>, engine: Engine) -> Self {
        let actual_seed = seed.unwrap_or(0); // Use 0 as a default seed if None
        let rng = StdRng::seed_from_u64(actual_seed);

        Game {
            world: World::new(world_width, world_height),
            teams: HashMap::new(),
            cells: HashMap::new(),
            coordinate_map: HashMap::new(),
            inv_coordinate_map: HashMap::new(),
            iteration: 0,
            max_iterations,
            cell_config,
            rng, 
            engine: engine,
            next_cell_id: 0,
        }
    }

    pub fn add_team(&mut self, team_id: TeamId, mind_path: &PathBuf) -> Result<(), String> {
        if !mind_path.exists() {
            return Err(format!("Script file '{}' not found.", mind_path.display()));
        }
        
        let ast_result = self.engine.compile_file(mind_path.clone());
        
        let mind_function: AST;
        match ast_result {
            Ok(ast) => mind_function = ast,
            Err(e) => return Err(format!("Failed to compile mind script '{}': {}. See diagnostics above.", mind_path.display(), e)),
        }
        self.teams.insert(team_id, mind_function);
        // Place starting cells for this team
        for _ in 0..self.cell_config.starting_cells_per_team {
            let mut attempts = 0;
            loop {
                let x = self.rng.random_range(0..self.world.dimensions.0);
                let y = self.rng.random_range(0..self.world.dimensions.1);
                let coord = Coordinate { x, y };
                if !self.coordinate_map.contains_key(&coord) {
                    let cell_id_val = self.next_cell_id;
                    self.next_cell_id += 1;
                    let new_cell_id = CellId(cell_id_val);
                    let initial_energy_for_cell = self.cell_config.initial_energy.min(self.cell_config.max_energy);
                    let cell = Cell::new(
                        new_cell_id,
                        team_id,
                        initial_energy_for_cell,
                        self.cell_config.min_energy,
                    );
                    self.cells.insert(new_cell_id, cell);
                    self.coordinate_map.insert(coord, new_cell_id);
                    self.inv_coordinate_map.insert(new_cell_id, coord);
                    println!("Placed starting cell {:?} for team {:?} at {:?}", new_cell_id, team_id, coord);
                    break;
                }
                attempts += 1;
                if attempts > self.world.dimensions.0 * self.world.dimensions.1 { // Avoid infinite loop in crowded worlds
                    eprintln!("Could not place starting cell for team {:?} due to lack of space.", team_id);
                    break;
                }
            }
        }
        Ok(())
    }

    pub fn run(&mut self) {
        while self.iteration < self.max_iterations {
            self.iteration += 1;
            self.tick();
        }
    }

    pub fn get_markers_at(&self, position: Coordinate) -> [Option<u32>; 8] {
        let mut markers = [None; 8];
        for direction in Direction::all() {
            if direction == Direction::Center {
                continue;
            }
            let neighbor_position = relative_position(position, direction, self.world.dimensions);
            let neighbor = self.get_cell_at(neighbor_position);
            if let Some(marker) = neighbor.map(|c| c.marker) {
                markers[marker as usize] = Some(marker);
            }
        }
        markers
    }

    pub fn get_cell_at(&self, position: Coordinate) -> Option<&Cell> {
        self.cells.get(&self.coordinate_map[&position])
    }

    pub fn get_cell_context(&self, coordinate: Coordinate) -> CellContext {
        let neighborhood = self.world.get_neighborhood(coordinate);
        let markers = self.get_markers_at(coordinate);
        CellContext {
            elevation: neighborhood.elevation,
            energy: neighborhood.energy,
            pheromone: neighborhood.pheromone,
            markers,
        }
    }

    pub fn get_cell_interactions(&mut self) -> Vec<CellInteraction> {
        let mut interactions: Vec<CellInteraction> = Vec::new();
        for (coordinate, cell_id) in &self.coordinate_map {
            let cell_copy = self.cells.get(cell_id).unwrap().clone();
            let cell_position = *coordinate;
            let cell_energy = cell_copy.energy;
            let cell_team_id = cell_copy.team_id;
            let context = self.get_cell_context(*coordinate);
            let mut scope = Scope::new();
            scope.push("cell", cell_copy);
            scope.push("context", context);
            let mind_result = self.engine.eval_ast_with_scope::<CellAction>(&mut scope, &self.teams[&cell_team_id]);
            match mind_result {
                Ok(output_value) => {
                    match output_value {
                        CellAction::Split(direction, energy, marker, memory) => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                            let target_coordinate = relative_position(cell_position, direction, self.world.dimensions);
                            interactions.push(CellInteraction::Split(*cell_id, cell_energy, target_coordinate, energy, marker, memory));
                        }
                        CellAction::Defend => {
                            self.cells.get_mut(cell_id).unwrap().defending = true;
                        }
                        CellAction::LiftTerrain => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                            interactions.push(CellInteraction::LiftTerrain(*cell_id, cell_position));
                        }
                        CellAction::DumpTerrain => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                            interactions.push(CellInteraction::DumpTerrain(*cell_id, cell_position));
                        }
                        CellAction::SetPheromone(pheromone) => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                            interactions.push(CellInteraction::SetPheromone(*cell_id, cell_position, pheromone));
                        }
                        CellAction::Attack(direction) => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                            let target_coordinate = relative_position(cell_position, direction, self.world.dimensions);
                            let target_cell = self.get_cell_at(target_coordinate);
                            match target_cell {
                                Some(target_cell) => {
                                    interactions.push(CellInteraction::Attack(*cell_id, cell_energy, target_cell.id));
                                }
                                None => {}
                            }
                        }
                        CellAction::SendMessage(direction, message) => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                            let target_coordinate = relative_position(cell_position, direction, self.world.dimensions);
                            let target_cell = self.get_cell_at(target_coordinate);
                            match target_cell {
                                Some(target_cell) => {
                                    interactions.push(CellInteraction::SendMessage(target_cell.id, message));
                                }
                                None => {}
                            }
                        }
                        CellAction::Move(direction) => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                            let target_coordinate = relative_position(cell_position, direction, self.world.dimensions);
                            interactions.push(CellInteraction::Move(*cell_id, cell_energy, target_coordinate));
                        }
                        CellAction::Eat => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                            interactions.push(CellInteraction::Eat(*cell_id, cell_position));
                        }
                        CellAction::DoNothing => {
                            self.cells.get_mut(cell_id).unwrap().defending = false;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error parsing mind result for cell {:?}: {}", cell_id, e);
                }
            }
        }
        interactions.sort();
        interactions
    }

    pub fn tick(&mut self) {
        let interactions = self.get_cell_interactions();

        for interaction in interactions {
            match interaction {
                CellInteraction::SendMessage(receiver_id, message) => {
                    if let Some(cell) = self.cells.get_mut(&receiver_id) {
                        cell.message_queue.push(message);
                    }
                }
                CellInteraction::Attack(attacker_id, attacker_energy_snapshot, target_id) => {
                    if !self.cells.contains_key(&attacker_id) || !self.cells.contains_key(&target_id) {
                        continue; // Attacker or target (or both) no longer exist
                    }

                    let attacker_pos_opt = self.inv_coordinate_map.get(&attacker_id).copied();
                    let target_pos_opt = self.inv_coordinate_map.get(&target_id).copied();

                    if let (Some(attacker_pos), Some(target_pos)) = (attacker_pos_opt, target_pos_opt) {
                        let elevation_attacker = self.world.elevation_at(attacker_pos);
                        let elevation_target = self.world.elevation_at(target_pos);

                        if (elevation_attacker - elevation_target).abs() > 1 {
                            // println!("Attack by {:?} on {:?} failed: elevation difference too high ({}) vs ({}) at {:?} vs {:?}", attacker_id, target_id, elevation_attacker, elevation_target, attacker_pos, target_pos);
                            continue; // Elevation difference too high
                        }

                        // Proceed with attack logic (already implemented by user/previous steps)
                        if let Some(target_cell_mut) = self.cells.get_mut(&target_id) {
                            // New Attack Power Calculation
                            let cfg = &self.cell_config;
                            let mut calculated_damage = if cfg.max_energy_for_attack_scaling == 0 { // Avoid division by zero
                                cfg.max_attack_power // or min_attack_power, depending on desired behavior for 0 scaling energy
                            } else if attacker_energy_snapshot >= cfg.max_energy_for_attack_scaling {
                                cfg.max_attack_power
                            } else {
                                let lerp_factor = attacker_energy_snapshot as f32 / cfg.max_energy_for_attack_scaling as f32;
                                (cfg.min_attack_power as f32 + lerp_factor * (cfg.max_attack_power - cfg.min_attack_power) as f32).round() as u32
                            };

                            calculated_damage = calculated_damage.max(cfg.min_attack_power); // Ensure it's at least min_attack_power

                            if target_cell_mut.defending {
                                calculated_damage /= 2; 
                            }

                            if calculated_damage > target_cell_mut.energy { 
                                target_cell_mut.energy = 0;
                            } else {
                                target_cell_mut.energy -= calculated_damage;
                            }

                            if target_cell_mut.energy == 0 {
                                println!("Cell {:?} (energy snapshot {}) killed cell {:?} with {} damage (raw {}). Target was {}defending.", 
                                    attacker_id, attacker_energy_snapshot, target_id, calculated_damage, 
                                    // to see raw before defense: re-calculate or store intermediate before /2
                                    if attacker_energy_snapshot >= cfg.max_energy_for_attack_scaling { cfg.max_attack_power } 
                                    else { (cfg.min_attack_power as f32 + (attacker_energy_snapshot as f32 / cfg.max_energy_for_attack_scaling as f32) * (cfg.max_attack_power - cfg.min_attack_power) as f32).round() as u32 }, 
                                    if target_cell_mut.defending { "" } else { "not "}
                                );
                                if let Some(killed_target_pos) = self.inv_coordinate_map.remove(&target_id) {
                                    self.coordinate_map.remove(&killed_target_pos);
                                    let killed_cell_min_energy = self.cells.get(&target_id).map_or(0, |c| c.min_energy);
                                    let killed_cell_loaded = self.cells.get(&target_id).map_or(false, |c| c.loaded);

                                    if killed_cell_min_energy > 0 {
                                        self.world.set_energy_at(killed_target_pos, Some(EnergySource::Scattered(killed_cell_min_energy)));
                                        println!("Cell {:?} dropped {} energy at {:?} upon death.", target_id, killed_cell_min_energy, killed_target_pos);
                                    }
                                    if killed_cell_loaded {
                                        self.world.set_elevation_at(killed_target_pos, self.world.elevation_at(killed_target_pos) + 1);
                                        println!("Cell {:?} dropped terrain at {:?} upon death.", target_id, killed_target_pos);
                                    }
                                }
                                self.cells.remove(&target_id); // Remove after getting min_energy and loaded status
                            }
                        }
                    } else {
                        // Attacker or target position not found, likely one was killed earlier
                        continue;
                    }
                }
                CellInteraction::Split(parent_id, _parent_energy_snapshot, target_coordinate, energy_to_child_i32, child_marker, child_memory) => {
                    if !self.cells.contains_key(&parent_id) {
                        continue; // Parent cell no longer exists
                    }
                    let parent_pos_opt = self.inv_coordinate_map.get(&parent_id).copied();

                    if let Some(parent_pos) = parent_pos_opt {
                        let elevation_parent = self.world.elevation_at(parent_pos);
                        let elevation_target = self.world.elevation_at(target_coordinate);

                        if (elevation_parent - elevation_target).abs() > 1 {
                            // println!("Split by {:?} to {:?} failed: elevation difference too high ({}) vs ({}) at {:?} vs {:?}", parent_id, target_coordinate, elevation_parent, elevation_target, parent_pos, target_coordinate);
                            continue; // Elevation difference too high
                        }

                        // Proceed with split logic (already implemented by user/previous steps)
                        if energy_to_child_i32 <= 0 {
                            continue; 
                        }
                        let energy_for_child = energy_to_child_i32 as u32;
                        if energy_for_child <= self.cell_config.min_energy { // Child must have more than min_energy
                            println!("Split by {:?} failed: energy for child ({}) not greater than min_energy ({}).", parent_id, energy_for_child, self.cell_config.min_energy);
                            continue;
                        }

                        if self.coordinate_map.contains_key(&target_coordinate) {
                            continue; 
                        }
                        if let Some(parent_cell) = self.cells.get_mut(&parent_id) {
                            if parent_cell.energy > energy_for_child { 
                                parent_cell.energy -= energy_for_child;
                                let child_id_val = self.next_cell_id;
                                self.next_cell_id += 1;
                                let child_id = CellId(child_id_val);
                                let new_cell = Cell::new(
                                    child_id,
                                    parent_cell.team_id, 
                                    energy_for_child,
                                    self.cell_config.min_energy, // New child also has min_energy requirement
                                );
                                self.cells.insert(child_id, new_cell);
                                self.coordinate_map.insert(target_coordinate, child_id);
                                self.inv_coordinate_map.insert(child_id, target_coordinate);
                                if let Some(child_cell_mut) = self.cells.get_mut(&child_id) {
                                    child_cell_mut.marker = child_marker;
                                    child_cell_mut.memory.copy_from_slice(&child_memory);
                                }
                                println!("Cell {:?} split, creating child {:?} at {:?} with energy {}", parent_id, child_id, target_coordinate, energy_for_child);
                            }
                        }
                    } else {
                        // Parent position not found
                        continue;
                    }
                }
                CellInteraction::SetPheromone(cell_id, coordinate, pheromone) => {
                    if self.cells.contains_key(&cell_id) { // Check if cell still exists
                        self.world.set_pheromone_at(coordinate, Some(pheromone));
                    }
                }
                CellInteraction::LiftTerrain(cell_id, coordinate) => {
                    if self.world.elevation_at(coordinate) > 0 {
                        if let Some(cell) = self.cells.get_mut(&cell_id) {
                            if !cell.loaded { // Can only lift if not already loaded
                                cell.loaded = true;
                                self.world.set_elevation_at(coordinate, self.world.elevation_at(coordinate) - 1);
                            }
                        }
                    }
                }
                CellInteraction::DumpTerrain(cell_id, coordinate) => {
                    if let Some(cell) = self.cells.get_mut(&cell_id) {
                        if cell.loaded { // Can only dump if loaded
                            cell.loaded = false;
                            self.world.set_elevation_at(coordinate, self.world.elevation_at(coordinate) + 1);
                        }
                    }
                }
                CellInteraction::Move(cell_id, _actor_energy_snapshot, new_coordinate) => {
                    if !self.cells.contains_key(&cell_id) {
                        continue; // Cell no longer exists
                    }
                    let origin_pos_opt = self.inv_coordinate_map.get(&cell_id).copied();

                    if let Some(origin_pos) = origin_pos_opt {
                        let elevation_origin = self.world.elevation_at(origin_pos);
                        let elevation_target = self.world.elevation_at(new_coordinate);

                        if (elevation_origin - elevation_target).abs() > 1 {
                            // println!("Move by {:?} from {:?} to {:?} failed: elevation difference too high ({}) vs ({})", cell_id, origin_pos, new_coordinate, elevation_origin, elevation_target);
                            continue; // Elevation difference too high
                        }

                        // Proceed with move logic (already implemented by user/previous steps)
                        if self.coordinate_map.contains_key(&new_coordinate) {
                            continue; 
                        }
                        if let Some(_cell_obj) = self.cells.get_mut(&cell_id) { // Renamed to _cell_obj as it's not used directly after this
                             // The old_position is origin_pos, which we already have.
                            self.coordinate_map.remove(&origin_pos);
                            // If Cell struct had a position field, it would be updated here: _cell_obj.position = new_coordinate;
                            self.coordinate_map.insert(new_coordinate, cell_id);
                            self.inv_coordinate_map.insert(cell_id, new_coordinate); // Update the reverse map
                        } else {
                             // This else should ideally not be reached if cells.contains_key(&cell_id) passed
                            eprintln!("Error: Cell {:?} expected but not found in cells map during move, despite earlier check.", cell_id);
                        }
                    } else {
                        // Origin position not found for cell_id
                         eprintln!("Error: Cell {:?} has no position in inv_coordinate_map during move.", cell_id);
                        continue;
                    }
                }
                CellInteraction::Eat(cell_id, position) => {
                    if let Some(cell) = self.cells.get_mut(&cell_id) {
                        // Check if cell is already at max energy
                        if cell.energy >= self.cell_config.max_energy {
                            println!("Cell {:?} attempt to eat at {:?} failed: already at max energy ({}/{})", cell_id, position, cell.energy, self.cell_config.max_energy);
                            continue; // Do not eat, leave tile energy as is
                        }

                        let world_idx = position.y * self.world.dimensions.0 + position.x;
                        let mut energy_gained = 0;
                        let mut original_source_present = false;
                        let mut source_to_restore: Option<EnergySource> = None; // To store original if eat fails partially

                        if world_idx < self.world.energy.len() {
                            if let Some(energy_source_at_pos) = self.world.energy[world_idx].clone() { // Clone to inspect
                                original_source_present = true;
                                source_to_restore = Some(energy_source_at_pos.clone()); // Keep a copy

                                match energy_source_at_pos {
                                    EnergySource::Scattered(amount) => {
                                        energy_gained = amount;
                                        if cell.energy + energy_gained > self.cell_config.max_energy {
                                            energy_gained = self.cell_config.max_energy - cell.energy;
                                        }
                                        if energy_gained > 0 { // if any energy can be gained
                                           println!("Cell {:?} eating {} scattered energy (available {}) at {:?}. Energy {} -> {}", 
                                                cell_id, energy_gained, amount, position, cell.energy, cell.energy + energy_gained);
                                           cell.energy += energy_gained;
                                           if amount == energy_gained { // Consumed all
                                               self.world.energy[world_idx] = None; 
                                           } else { // Consumed partially
                                               self.world.energy[world_idx] = Some(EnergySource::Scattered(amount - energy_gained));
                                           }
                                        } else {
                                            println!("Cell {:?} cannot eat more scattered energy at {:?}. Already at {}/{}", cell_id, position, cell.energy, self.cell_config.max_energy);
                                            // No energy gained, source remains as is (no need to restore self.world.energy[world_idx] = source_to_restore)
                                        }
                                    }
                                    EnergySource::Plant { rate, current_energy, max_energy } => {
                                        energy_gained = current_energy;
                                        if cell.energy + energy_gained > self.cell_config.max_energy {
                                            energy_gained = self.cell_config.max_energy - cell.energy;
                                        }
                                        if energy_gained > 0 { // if any energy can be gained
                                            println!("Cell {:?} eating {} energy from plant (available {}) at {:?}. Energy {} -> {}", 
                                                cell_id, energy_gained, current_energy, position, cell.energy, cell.energy + energy_gained);
                                            cell.energy += energy_gained;
                                            self.world.energy[world_idx] = Some(EnergySource::Plant {
                                                rate,
                                                current_energy: current_energy - energy_gained, 
                                                max_energy,
                                            });
                                        } else {
                                            println!("Cell {:?} cannot eat more plant energy at {:?}. Already at {}/{}", cell_id, position, cell.energy, self.cell_config.max_energy);
                                            // No energy gained, source remains as is
                                        }
                                    }
                                }
                            } else { // No energy source at position
                                println!("Cell {:?} attempt to eat at {:?} failed: no energy source found.", cell_id, position);
                            }
                        } else {
                            eprintln!("Error: Eat action for cell {:?} at {:?} - position is out of world bounds.", cell_id, position);
                        }
                    } else {
                        // Cell might have died before its eat action is processed
                    }
                }
            }
        }
        // TODO: Update cell ages, apply plant growth, pheromone decay, etc.
    }
    
    //Action resolution rules:
    //Actions are resolved in order of priority. If a cell is killed before its slow action can be completed, the action is cancelled.
    //The following actions can have conflicts:
    //Move, Split
    //In these cases, the cell with the lower energy wins. In case of a tie, a random cell is chosen.


}