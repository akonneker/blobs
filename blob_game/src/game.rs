use blob_interface::cell::{Cell};
use blob_interface::world::{World, relative_position, EnergySource};
use blob_interface::cell::{CellContext, CellAction};
use blob_interface::types::{Coordinate, CellId, TeamId, Pheromone, CellMessage, Direction};
use blob_interface::action_converter::{capnp_to_cell_action};
use blob_interface::mind_input_converter::cell_to_mind_input_capnp;
use crate::config::{CellConfig, MemoryConfig, StateConfig};
use std::collections::HashMap;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

use std::cmp::Ordering;

use extism::{Wasm, Manifest, Plugin, PluginBuilder, Error};
use extism_manifest::MemoryOptions;
use image::{ImageBuffer, Rgb, RgbImage};
use serde::{Serialize, Deserialize};
use std::fs;
use std::io::{Read, Write};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use bincode;


pub struct Game {
    pub world: World,
    pub teams: HashMap<TeamId, Plugin>,
    pub seeds: HashMap<TeamId, u64>,
    pub cells: HashMap<CellId, Cell>,
    pub coordinate_map: HashMap<Coordinate, CellId>,
    pub inv_coordinate_map: HashMap<CellId, Coordinate>,
    pub iteration: u64,
    pub max_iterations: u64,
    pub cell_config: CellConfig,
    pub memory_config: MemoryConfig,
    rng: StdRng,
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

/// Serializable representation of the game state
/// This excludes non-serializable data like WebAssembly plugins
#[derive(Serialize, Deserialize)]
pub struct GameState {
    pub world: World,
    pub cells: HashMap<CellId, Cell>,
    pub coordinate_map: HashMap<Coordinate, CellId>,
    pub inv_coordinate_map: HashMap<CellId, Coordinate>,
    pub iteration: u64,
    pub max_iterations: u64,
    pub cell_config: CellConfig,
    pub memory_config: MemoryConfig,
    pub next_cell_id: usize,
    pub seeds: HashMap<TeamId, u64>,
    // Note: teams (WebAssembly plugins) and rng are not serialized
    // They need to be reconstructed when loading
}

impl Game {
    pub fn new(world_width: usize, world_height: usize, max_iterations: u64, cell_config: CellConfig, seed: Option<u64>, memory_config: MemoryConfig) -> Self {
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
            memory_config,
            rng, 
            next_cell_id: 0,
            seeds: HashMap::new(),
        }
    }

    pub fn add_team(&mut self, team_id: TeamId, mind_path: &PathBuf) -> Result<(), String> {
        self.add_team_with_start_id(team_id, mind_path, 0)
    }

    pub fn add_team_with_start_id(&mut self, team_id: TeamId, mind_path: &PathBuf, start_id: u32) -> Result<(), String> {
        if !mind_path.exists() {
            return Err(format!("Script file '{}' not found.", mind_path.display()));
        }

        let wasm_file = Wasm::file(mind_path.clone());
        let manifest = Manifest::new([wasm_file])
            .with_memory_options(
            MemoryOptions::new()
            .with_max_pages(self.memory_config.max_pages)
            .with_max_var_bytes(self.memory_config.max_var_bytes)
        ).disallow_all_hosts()
        .with_timeout(self.memory_config.timeout_duration());
        
        let plugin_result = PluginBuilder::new(manifest).build();
        
        let mind_function: Plugin;
        match plugin_result {
            Ok(plugin) => mind_function = plugin,
            Err(e) => return Err(format!("Failed to compile mind script '{}': {}. See diagnostics above.", mind_path.display(), e)),
        }
        self.teams.insert(team_id, mind_function);
        self.seeds.insert(team_id, self.rng.random_range(0..u64::MAX));
        
        // Place starting cells for this team in a cluster with the specified start_id
        self.place_team_cluster_with_start_id(team_id, start_id)?;
        
        Ok(())
    }

    /// Place a team's starting cells in a spiral checkerboard cluster with a central plant
    fn place_team_cluster(&mut self, team_id: TeamId) -> Result<(), String> {
        self.place_team_cluster_with_start_id(team_id, 0)
    }

    /// Place a team's starting cells in a spiral checkerboard cluster with a central plant
    fn place_team_cluster_with_start_id(&mut self, team_id: TeamId, start_id: u32) -> Result<(), String> {
        let num_teams = self.teams.len();
        let team_index = team_id.0;
        
        // Calculate cluster center based on team index for even spacing
        let cluster_center = self.calculate_cluster_center(team_index, num_teams);
        
        // Find the maximum plant values from the energy config
        let max_plant_rate = self.get_max_plant_rate();
        let max_plant_energy = self.get_max_plant_energy();
        
        // Place a central plant at the cluster center
        self.world.set_energy_at(cluster_center, Some(EnergySource::Plant {
            rate: max_plant_rate,
            current_energy: max_plant_energy,
            max_energy: max_plant_energy,
        }));
        
        // Generate spiral checkerboard positions around the center
        let cell_positions = self.generate_spiral_checkerboard_positions(
            cluster_center, 
            self.cell_config.starting_cells_per_team
        );
        
        // Place cells at the generated positions
        for position in cell_positions {
            // Check if position is valid and unoccupied
            if !self.coordinate_map.contains_key(&position) {
                let cell_id_val = self.next_cell_id;
                self.next_cell_id += 1;
                let new_cell_id = CellId(cell_id_val);
                let initial_energy_for_cell = self.cell_config.initial_energy.min(self.cell_config.max_energy);
                let mut cell = Cell::new(
                    new_cell_id,
                    team_id,
                    initial_energy_for_cell,
                    self.cell_config.min_energy,
                    self.cell_config.max_energy,
                );
                // Set the marker to the start_id
                cell.marker = start_id;
                self.cells.insert(new_cell_id, cell);
                self.coordinate_map.insert(position, new_cell_id);
                self.inv_coordinate_map.insert(new_cell_id, position);
                println!("Placed starting cell {:?} for team {:?} at {:?} with marker {}", new_cell_id, team_id, position, start_id);
            } else {
                eprintln!("Warning: Position {:?} already occupied when placing team {:?}", position, team_id);
            }
        }
        
        Ok(())
    }

    /// Calculate the center position for a team's cluster based on even spacing
    fn calculate_cluster_center(&self, team_index: usize, num_teams: usize) -> Coordinate {
        let world_width = self.world.dimensions.0;
        let world_height = self.world.dimensions.1;
        
        // Calculate grid dimensions for team placement
        let teams_per_row = (num_teams as f64).sqrt().ceil() as usize;
        let teams_per_col = (num_teams + teams_per_row - 1) / teams_per_row; // Ceiling division
        
        let row = team_index / teams_per_row;
        let col = team_index % teams_per_row;
        
        // Calculate spacing to evenly distribute clusters
        let spacing_x = world_width / teams_per_row;
        let spacing_y = world_height / teams_per_col;
        
        // Center each cluster within its allocated space
        let center_x = col * spacing_x + spacing_x / 2;
        let center_y = row * spacing_y + spacing_y / 2;
        
        Coordinate {
            x: center_x.min(world_width - 1),
            y: center_y.min(world_height - 1),
        }
    }

    /// Generate positions in a spiral checkerboard pattern around a center point
    /// Following the pattern:
    /// 3x3: 3X4    5x5: 10X11X12
    ///      X0X          X3X4X
    ///      2X1          9X0X5
    ///                   X2X1X
    ///                   8X7X6
    fn generate_spiral_checkerboard_positions(&self, center: Coordinate, num_cells: usize) -> Vec<Coordinate> {
        let mut positions = Vec::new();
        
        if num_cells == 0 {
            return positions;
        }
        
        let mut ring = 0; // Start with ring 0 (the center)
        
        while positions.len() < num_cells {
            // Generate positions for current ring
            let ring_positions = self.generate_ring_checkerboard_positions(center, ring);
            
            for pos in ring_positions {
                if positions.len() >= num_cells {
                    break;
                }
                // Only add valid positions within world bounds
                if pos.x < self.world.dimensions.0 && pos.y < self.world.dimensions.1 {
                    positions.push(pos);
                }
            }
            
            ring += 1;
            if ring > 20 { // Safety limit to prevent infinite loops
                break;
            }
        }
        
        positions
    }

    /// Generate checkerboard positions for a specific ring around the center
    fn generate_ring_checkerboard_positions(&self, center: Coordinate, ring: usize) -> Vec<Coordinate> {
        let mut positions = Vec::new();
        
        if ring == 0 {
            // Ring 0 is just the center
            positions.push(center);
            return positions;
        }
        
        let ring_i = ring as i32;
        
        // Calculate the center's checkerboard parity
        let center_parity = (center.x + center.y) % 2;
        
        // Generate all positions in the ring
        for dx in -ring_i..=ring_i {
            for dy in -ring_i..=ring_i {
                // Check if this position is on the ring boundary
                if dx.abs() == ring_i || dy.abs() == ring_i {
                    let x = center.x as i32 + dx;
                    let y = center.y as i32 + dy;
                    
                    // Check bounds
                    if x >= 0 && y >= 0 {
                        let coord = Coordinate {
                            x: x as usize,
                            y: y as usize,
                        };
                        
                        // Apply checkerboard pattern (match the center's parity)
                        if (coord.x + coord.y) % 2 == center_parity {
                            positions.push(coord);
                        }
                    }
                }
            }
        }
        
        positions
    }

    /// Get the maximum plant rate from the energy configuration  
    fn get_max_plant_rate(&self) -> u32 {
        // Since we don't have direct access to the energy config here,
        // we'll use a reasonable maximum based on typical config values
        // This could be made configurable in the future
        50 // High rate for central cluster plants
    }

    /// Get the maximum plant energy from the energy configuration
    fn get_max_plant_energy(&self) -> u32 {
        // Since we don't have direct access to the energy config here,
        // we'll use a reasonable maximum based on typical config values
        // This could be made configurable in the future
        300 // High energy for central cluster plants
    }

    pub fn run(&mut self, verbose: bool) -> Result<(), Error> {
        while self.iteration < self.max_iterations {
            if verbose {
                println!("Iteration {}", self.iteration);
            }
            self.tick(verbose)?;
        }
        Ok(())
    }

    /// Run a specified number of game steps (iterations)
    /// Returns the number of steps actually executed (may be less if max_iterations is reached)
    pub fn step(&mut self, steps: u64, verbose: bool) -> Result<u64, Error> {
        let mut executed_steps = 0;
        for _ in 0..steps {
            if self.iteration >= self.max_iterations {
                break;
            }
            if verbose {
                println!("Iteration {}", self.iteration);
            }
            self.tick(verbose)?;
            executed_steps += 1;
        }
        Ok(executed_steps)
    }

    pub fn get_markers_at(&self, position: Coordinate) -> [Option<u32>; 8] {
        let mut markers = [None; 8];
        for direction in Direction::all().iter() {
            let neighbor_position = relative_position(position, Some(*direction), self.world.dimensions);
            let neighbor = self.get_cell_at(neighbor_position);
            if let Some(marker) = neighbor.map(|c| c.marker) {
                markers[direction.index()] = Some(marker);
            }
        }
        markers
    }

    pub fn get_cell_at(&self, position: Coordinate) -> Option<&Cell> {
        match self.coordinate_map.get(&position) {
            Some(cell_id) => self.cells.get(cell_id),
            None => None,
        }
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

            let seed = self.seeds.get(&cell_team_id).unwrap();

            let cell_input_res = cell_to_mind_input_capnp(&cell_copy, &context, *seed);

            let cell_input: Vec<u8>;
            match cell_input_res {
                Ok(value) => {
                    cell_input = value;
                }
                Err(e) => {
                    eprintln!("Error converting cell input for cell {:?}: {}", cell_id, e);
                    continue;
                }
            }

            let mind_result = if let Some(val) = self.teams.get_mut(&cell_team_id) { 
                val.call::<&Vec<u8>, Vec<u8>>("mind_function", &cell_input) 
            } else {
                Err(Error::msg("Team not found"))
            };

            let action: CellAction;
            match mind_result {
                Ok(output_value) => {
                    let action_result = capnp_to_cell_action(&output_value);
                    match action_result {
                        Ok(parsed_action) => {
                            action = parsed_action;
                        }
                        Err(e) => {
                            eprintln!("Error parsing action for cell {:?}: {}", cell_id, e);
                            action = CellAction::DoNothing;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error parsing mind result for cell {:?}: {}", cell_id, e);
                    action = CellAction::DoNothing;
                }
            }

            match action {
                CellAction::Split(direction, energy, marker, memory) => {
                    self.cells.get_mut(cell_id).unwrap().defending = false;
                    let target_coordinate = relative_position(cell_position, Some(direction), self.world.dimensions);
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
                    let target_coordinate = relative_position(cell_position, Some(direction), self.world.dimensions);
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
                    let target_coordinate = relative_position(cell_position, Some(direction), self.world.dimensions);
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
                    let target_coordinate = relative_position(cell_position, Some(direction), self.world.dimensions);
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
        interactions.sort();
        interactions
    }

    pub fn tick(&mut self, verbose: bool) -> Result<(), Error> {

        for (_team_id, plugin) in &mut self.teams {
            plugin.reset()?;
        }

        let interactions = self.get_cell_interactions();

        for interaction in interactions {
            if verbose {
                println!("Interaction: {:?}", interaction);
            }
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
                                    self.cell_config.max_energy, // New child also has max_energy limit
                                );
                                self.cells.insert(child_id, new_cell);
                                self.coordinate_map.insert(target_coordinate, child_id);
                                self.inv_coordinate_map.insert(child_id, target_coordinate);
                                if let Some(child_cell_mut) = self.cells.get_mut(&child_id) {
                                    child_cell_mut.marker = child_marker;
                                    child_cell_mut.set_memory_array(child_memory);
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
                        let mut energy_gained: u32;

                        if world_idx < self.world.energy.len() {
                            if let Some(energy_source_at_pos) = self.world.energy[world_idx].clone() { // Clone to inspect

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
        // Increment age for all living cells
        for (_cell_id, cell) in self.cells.iter_mut() {
            cell.age = cell.age.saturating_add(1);
        }

        // Apply plant growth
        for energy_source_option in self.world.energy.iter_mut() {
            if let Some(energy_source) = energy_source_option {
                if let EnergySource::Plant { rate, current_energy, max_energy } = energy_source {
                    let growth = *rate;
                    if *current_energy < *max_energy { // Only grow if not already at max
                        *current_energy = (*current_energy + growth).min(*max_energy);
                    }
                }
            }
        }

        // Increment iteration counter
        self.iteration += 1;
        
        Ok(())

        // TODO: Apply pheromone decay, etc.
    }
    
    /// Enhanced tick method that includes automatic state saving
    pub fn tick_with_auto_save(&mut self, verbose: bool, state_config: &StateConfig) -> Result<(), Error> {
        self.tick(verbose)?;
        
        // Auto-save state if configured
        if let Err(e) = self.auto_save_state(state_config) {
            eprintln!("Warning: Failed to auto-save state: {}", e);
        }
        
        Ok(())
    }
    
    /// Generate an image representation of the current game state
    /// 
    /// Creates a PNG image showing:
    /// - Terrain elevation (darker = higher elevation)
    /// - Energy sources (green for scattered energy, bright green for plants)
    /// - Cells (colored by team ID, brightness based on energy)
    /// - Pheromones (blue overlay)
    /// 
    /// Returns the image data as RgbImage for use in GUI applications.
    /// Optionally saves to disk if save_to_disk is true and filename is provided.
    pub fn generate_board_image(&self, scale: u32, filename: Option<&str>, save_to_disk: bool) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let width = self.world.dimensions.0 as u32;
        let height = self.world.dimensions.1 as u32;
        let img_width = width * scale;
        let img_height = height * scale;
        
        let mut img: RgbImage = ImageBuffer::new(img_width, img_height);
        
        // Find min/max elevation for normalization
        let min_elevation = self.world.elevation.iter().min().copied().unwrap_or(0);
        let max_elevation = self.world.elevation.iter().max().copied().unwrap_or(0);
        let elevation_range = max_elevation - min_elevation;
        
        // Define team colors (cycling through a palette)
        let team_colors = [
            [255, 100, 100], // Red
            [100, 100, 255], // Blue  
            [255, 255, 100], // Yellow
            [255, 100, 255], // Magenta
            [100, 255, 255], // Cyan
            [255, 165, 0],   // Orange
            [128, 0, 128],   // Purple
            [0, 128, 0],     // Dark Green
            [165, 42, 42],   // Brown
            [255, 20, 147],  // Deep Pink
        ];
        
        for y in 0..height {
            for x in 0..width {
                let coord = Coordinate { x: x as usize, y: y as usize };
                let world_idx = y as usize * self.world.dimensions.0 + x as usize;
                
                // Base color: terrain elevation (grayscale)
                let elevation = self.world.elevation[world_idx];
                let normalized_elevation = if elevation_range > 0 {
                    ((elevation - min_elevation) as f32 / elevation_range as f32 * 128.0) as u8 + 64
                } else {
                    128
                };
                
                let mut base_color = [normalized_elevation, normalized_elevation, normalized_elevation];
                
                // Energy sources overlay
                if let Some(energy_source) = &self.world.energy[world_idx] {
                    match energy_source {
                        EnergySource::Scattered(amount) => {
                            // Green tint for scattered energy
                            let intensity = (*amount as f32 / 200.0).min(1.0); // Assume max ~200 energy
                            base_color[1] = (base_color[1] as f32 + intensity * 100.0).min(255.0) as u8;
                        }
                        EnergySource::Plant { current_energy, max_energy, .. } => {
                            // Bright green for plants, intensity based on current energy
                            let intensity = (*current_energy as f32 / *max_energy as f32).min(1.0);
                            base_color[0] = (base_color[0] as f32 * (1.0 - intensity * 0.5)) as u8;
                            base_color[1] = (255.0 * intensity + base_color[1] as f32 * (1.0 - intensity)) as u8;
                            base_color[2] = (base_color[2] as f32 * (1.0 - intensity * 0.5)) as u8;
                        }
                    }
                }
                
                // Pheromone overlay (blue tint)
                if let Some(pheromone) = self.world.pheromone[world_idx] {
                    let intensity = (pheromone as f32 / 255.0).min(1.0);
                    base_color[2] = (base_color[2] as f32 + intensity * 80.0).min(255.0) as u8;
                }
                
                // Cell overlay (dominant color)
                if let Some(cell_id) = self.coordinate_map.get(&coord) {
                    if let Some(cell) = self.cells.get(cell_id) {
                        // Get team color
                        let team_index = cell.team_id.0 % team_colors.len();
                        let team_color = team_colors[team_index];
                        
                        // Energy-based brightness (0.3 to 1.0 based on energy ratio)
                        let energy_ratio = (cell.energy as f32 / self.cell_config.max_energy as f32).min(1.0);
                        let brightness = 0.3 + energy_ratio * 0.7;
                        
                        // Apply team color with energy-based brightness
                        base_color[0] = (team_color[0] as f32 * brightness) as u8;
                        base_color[1] = (team_color[1] as f32 * brightness) as u8;
                        base_color[2] = (team_color[2] as f32 * brightness) as u8;
                        
                        // Add visual indicators for special states
                        if cell.defending {
                            // Add white border for defending cells
                            base_color = [255, 255, 255];
                        } else if cell.loaded {
                            // Add dark border for cells carrying terrain
                            base_color[0] = (base_color[0] as f32 * 0.7) as u8;
                            base_color[1] = (base_color[1] as f32 * 0.7) as u8;
                            base_color[2] = (base_color[2] as f32 * 0.7) as u8;
                        }
                    }
                }
                
                // Apply the color to all pixels in the scaled area
                for dy in 0..scale {
                    for dx in 0..scale {
                        let pixel_x = x * scale + dx;
                        let pixel_y = y * scale + dy;
                        if pixel_x < img_width && pixel_y < img_height {
                            img.put_pixel(pixel_x, pixel_y, Rgb(base_color));
                        }
                    }
                }
            }
        }
        
        // Optionally save the image to disk
        if save_to_disk {
            if let Some(file_name) = filename {
                img.save(file_name)?;
                println!("Board image saved to: {}", file_name);
            } else {
                return Err("Filename must be provided when save_to_disk is true".into());
            }
        }
        
        Ok(img)
    }
    
    /// Generate an image for the current iteration with automatic filename
    /// 
    /// Creates a PNG image with filename format: "board_iteration_{iteration}.png"
    /// This is useful for creating animations or tracking game progress over time.
    /// Returns the image data and optionally saves to disk.
    pub fn generate_iteration_image(&self, scale: u32, save_to_disk: bool) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let filename = format!("board_iteration_{:06}.png", self.iteration);
        self.generate_board_image(scale, Some(&filename), save_to_disk)
    }
    
    /// Save the current game state to a file
    /// 
    /// Serializes the game state (excluding WebAssembly plugins) and optionally compresses it
    pub fn save_state(&self, filename: &str, compress: bool) -> Result<(), Box<dyn std::error::Error>> {
        let state = GameState {
            world: self.world.clone(),
            cells: self.cells.clone(),
            coordinate_map: self.coordinate_map.clone(),
            inv_coordinate_map: self.inv_coordinate_map.clone(),
            iteration: self.iteration,
            max_iterations: self.max_iterations,
            cell_config: self.cell_config.clone(),
            memory_config: self.memory_config.clone(),
            next_cell_id: self.next_cell_id,
            seeds: self.seeds.clone(),
        };
        
        let serialized = bincode::serialize(&state)?;
        
        if compress {
            let file = fs::File::create(filename)?;
            let mut encoder = GzEncoder::new(file, Compression::default());
            encoder.write_all(&serialized)?;
            encoder.finish()?;
        } else {
            fs::write(filename, serialized)?;
        }
        
        println!("Game state saved to: {}", filename);
        Ok(())
    }
    
    /// Load a game state from a file
    /// 
    /// Deserializes the game state and returns it. The caller must reconstruct
    /// the WebAssembly plugins using `from_state_with_teams`.
    pub fn load_state(filename: &str, compressed: bool) -> Result<GameState, Box<dyn std::error::Error>> {
        let data = if compressed {
            let file = fs::File::open(filename)?;
            let mut decoder = GzDecoder::new(file);
            let mut buffer = Vec::new();
            decoder.read_to_end(&mut buffer)?;
            buffer
        } else {
            fs::read(filename)?
        };
        
        let state: GameState = bincode::deserialize(&data)?;
        println!("Game state loaded from: {}", filename);
        Ok(state)
    }
    
    /// Create a new Game instance from a saved GameState
    /// 
    /// This reconstructs the non-serializable parts (RNG, plugins) and allows
    /// continuing simulation from a saved state.
    pub fn from_state_with_teams(
        state: GameState,
        team_mind_paths: &[(TeamId, PathBuf)],
        seed_override: Option<u64>,
    ) -> Result<Self, String> {
        // Create RNG with seed
        let actual_seed = seed_override.unwrap_or(42);
        let rng = StdRng::seed_from_u64(actual_seed);
        
        let mut game = Game {
            world: state.world,
            teams: HashMap::new(),
            seeds: state.seeds,
            cells: state.cells,
            coordinate_map: state.coordinate_map,
            inv_coordinate_map: state.inv_coordinate_map,
            iteration: state.iteration,
            max_iterations: state.max_iterations,
            cell_config: state.cell_config,
            memory_config: state.memory_config,
            rng,
            next_cell_id: state.next_cell_id,
        };
        
        // Reconstruct WebAssembly plugins without placing new starting cells
        for (team_id, mind_path) in team_mind_paths {
            if !mind_path.exists() {
                return Err(format!("Script file '{}' not found.", mind_path.display()));
            }

            let wasm_file = Wasm::file(mind_path.clone());
            let manifest = Manifest::new([wasm_file])
                .with_memory_options(
                MemoryOptions::new()
                .with_max_pages(game.memory_config.max_pages)
                .with_max_var_bytes(game.memory_config.max_var_bytes)
            ).disallow_all_hosts()
            .with_timeout(game.memory_config.timeout_duration());
            
            let plugin_result = PluginBuilder::new(manifest).build();
            
            match plugin_result {
                Ok(plugin) => {
                    game.teams.insert(*team_id, plugin);
                    println!("Reconstructed team {:?} from {:?}", team_id, mind_path);
                }
                Err(e) => {
                    return Err(format!("Failed to compile mind script '{}': {}. See diagnostics above.", mind_path.display(), e));
                }
            }
        }
        
        Ok(game)
    }
    
    /// Save state automatically based on configuration
    /// 
    /// Uses the state configuration to determine filename, compression, etc.
    pub fn auto_save_state(&self, state_config: &StateConfig) -> Result<(), Box<dyn std::error::Error>> {
        if self.iteration % state_config.save_interval == 0 {
            // Create state directory if it doesn't exist
            fs::create_dir_all(&state_config.state_directory)?;
            
            let filename = format!(
                "{}/game_state_{:06}.{}",
                state_config.state_directory,
                self.iteration,
                if state_config.compress_states { "gz" } else { "bin" }
            );
            
            self.save_state(&filename, state_config.compress_states)?;
            
            // Clean up old states if auto_cleanup is enabled
            if state_config.auto_cleanup {
                self.cleanup_old_states(state_config)?;
            }
        }
        Ok(())
    }
    
    /// Clean up old state files based on configuration
    fn cleanup_old_states(&self, state_config: &StateConfig) -> Result<(), Box<dyn std::error::Error>> {
        let dir = fs::read_dir(&state_config.state_directory)?;
        let mut state_files: Vec<_> = dir
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.is_file() {
                    let filename = path.file_name()?.to_str()?;
                    if filename.starts_with("game_state_") {
                        Some((path.clone(), filename.to_string()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by filename (which includes iteration number)
        state_files.sort_by(|a, b| a.1.cmp(&b.1));
        
        // Remove oldest files if we exceed the limit
        if state_files.len() > state_config.max_disk_states {
            let files_to_remove = state_files.len() - state_config.max_disk_states;
            for (path, _) in state_files.iter().take(files_to_remove) {
                fs::remove_file(path)?;
                println!("Cleaned up old state file: {:?}", path);
            }
        }
        
        Ok(())
    }
    
    //Action resolution rules:
    //Actions are resolved in order of priority. If a cell is killed before its slow action can be completed, the action is cancelled.
    //The following actions can have conflicts:
    //Move, Split
    //In these cases, the cell with the lower energy wins. In case of a tie, a random cell is chosen.


}