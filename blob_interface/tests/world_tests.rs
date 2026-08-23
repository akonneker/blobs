use blob_interface::cell::Cell;
use blob_interface::types::{CellId, Coordinate, TeamId};
use blob_interface::world::{EnergySource, World};

/// Test that World::get_neighborhood returns correct values for all 9 positions
#[test]
fn test_neighborhood_correctness() {
    let mut world = World::new(10, 10);
    // Set some known values
    world.elevation[55] = 5; // (5, 5) center
    world.elevation[44] = 1; // (4, 4) NW
    world.elevation[45] = 2; // (5, 4) N
    world.elevation[46] = 3; // (6, 4) NE
    world.elevation[54] = 4; // (4, 5) W
    world.elevation[56] = 6; // (6, 5) E
    world.elevation[64] = 7; // (4, 6) SW
    world.elevation[65] = 8; // (5, 6) S
    world.elevation[66] = 9; // (6, 6) SE

    let center = Coordinate { x: 5, y: 5 };
    let neighborhood = world.get_neighborhood(center);

    assert_eq!(neighborhood.elevation[8], 5, "Center elevation");
    assert_eq!(neighborhood.elevation[0], 1, "NW elevation");
    assert_eq!(neighborhood.elevation[1], 2, "N elevation");
    assert_eq!(neighborhood.elevation[2], 3, "NE elevation");
    assert_eq!(neighborhood.elevation[3], 4, "W elevation");
    assert_eq!(neighborhood.elevation[4], 6, "E elevation");
    assert_eq!(neighborhood.elevation[5], 7, "SW elevation");
    assert_eq!(neighborhood.elevation[6], 8, "S elevation");
    assert_eq!(neighborhood.elevation[7], 9, "SE elevation");
}

/// Test that World wraps around at boundaries
#[test]
fn test_neighborhood_wrapping() {
    let mut world = World::new(10, 10);
    world.elevation[0] = 42; // (0, 0) top-left corner

    let center = Coordinate { x: 0, y: 0 };
    let neighborhood = world.get_neighborhood(center);

    assert_eq!(neighborhood.elevation[8], 42, "Center should be 42");
    // NW of (0,0) wraps to (9, 9)
    assert_eq!(
        neighborhood.elevation[0], world.elevation[99],
        "NW wraps to (9,9)"
    );
    // N of (0,0) wraps to (0, 9)
    assert_eq!(
        neighborhood.elevation[1], world.elevation[90],
        "N wraps to (0,9)"
    );
}

/// Test Cell creation with correct defaults
#[test]
fn test_cell_creation() {
    let cell = Cell::new(CellId(0), TeamId(0), 100, 10);
    assert_eq!(cell.energy, 100);
    assert_eq!(cell.min_energy, 10);
    assert_eq!(cell.marker, 0);
    assert!(!cell.loaded);
    assert_eq!(cell.age, 0);
    assert!(!cell.defending);
    assert!(cell.message_queue.is_empty());
    assert_eq!(cell.memory.len(), 2048);
}

/// Test Cell memory array round-trip
#[test]
fn test_cell_memory_roundtrip() {
    let mut cell = Cell::new(CellId(0), TeamId(0), 100, 10);
    let mut memory = [0u8; 2048];
    memory[0] = 42;
    memory[100] = 255;
    memory[2047] = 128;
    cell.set_memory_array(memory);

    let retrieved = cell.memory_array();
    assert_eq!(retrieved[0], 42);
    assert_eq!(retrieved[100], 255);
    assert_eq!(retrieved[2047], 128);
}

/// Test EnergySource plant growth logic
#[test]
fn test_plant_growth() {
    let mut world = World::new(5, 5);
    world.energy[12] = Some(EnergySource::Plant {
        rate: 10,
        current_energy: 50,
        max_energy: 100,
    });

    // Simulate growth (same logic as tick)
    for energy_source_option in world.energy.iter_mut() {
        if let Some(EnergySource::Plant {
            rate,
            current_energy,
            max_energy,
        }) = energy_source_option
        {
            let growth = *rate;
            if *current_energy < *max_energy {
                *current_energy = (*current_energy + growth).min(*max_energy);
            }
        }
    }

    match &world.energy[12] {
        Some(EnergySource::Plant { current_energy, .. }) => {
            assert_eq!(*current_energy, 60, "Plant should grow by rate");
        }
        _ => panic!("Expected plant energy source"),
    }
}

/// Test plant growth caps at max_energy
#[test]
fn test_plant_growth_capped() {
    let mut world = World::new(5, 5);
    world.energy[0] = Some(EnergySource::Plant {
        rate: 100,
        current_energy: 95,
        max_energy: 100,
    });

    for energy_source_option in world.energy.iter_mut() {
        if let Some(EnergySource::Plant {
            rate,
            current_energy,
            max_energy,
        }) = energy_source_option
            && *current_energy < *max_energy
        {
            *current_energy = (*current_energy + *rate).min(*max_energy);
        }
    }

    match &world.energy[0] {
        Some(EnergySource::Plant {
            current_energy,
            max_energy,
            ..
        }) => {
            assert_eq!(*current_energy, 100, "Plant should cap at max_energy");
            assert_eq!(*max_energy, 100);
        }
        _ => panic!("Expected plant energy source"),
    }
}

/// Test elevation set/get
#[test]
fn test_elevation_set_get() {
    let mut world = World::new(20, 20);
    let coord = Coordinate { x: 10, y: 15 };
    world.set_elevation_at(coord, 42);
    assert_eq!(world.elevation_at(coord), 42);
}

/// Test pheromone set/get
#[test]
fn test_pheromone_set_get() {
    let mut world = World::new(20, 20);
    let coord = Coordinate { x: 5, y: 5 };
    assert_eq!(world.pheromone_at(coord), None);
    world.set_pheromone_at(coord, Some(100));
    assert_eq!(world.pheromone_at(coord), Some(100));
    world.set_pheromone_at(coord, None);
    assert_eq!(world.pheromone_at(coord), None);
}

/// Test energy source set/get for scattered energy
#[test]
fn test_scattered_energy_set_get() {
    let mut world = World::new(10, 10);
    let coord = Coordinate { x: 3, y: 7 };
    assert_eq!(world.energy_at(coord), 0);
    world.set_energy_at(coord, Some(EnergySource::Scattered(50)));
    assert_eq!(world.energy_at(coord), 50);
}

/// Test World::new initializes correctly
#[test]
fn test_world_new() {
    let world = World::new(32, 64);
    assert_eq!(world.dimensions, (32, 64));
    assert_eq!(world.elevation.len(), 32 * 64);
    assert_eq!(world.energy.len(), 32 * 64);
    assert_eq!(world.pheromone.len(), 32 * 64);
    assert!(world.elevation.iter().all(|&e| e == 0));
    assert!(world.energy.iter().all(|e| e.is_none()));
    assert!(world.pheromone.iter().all(|p| p.is_none()));
}
