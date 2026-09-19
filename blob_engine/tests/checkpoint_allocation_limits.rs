use blob_engine::resolution::{
    CellKey, CheckpointLimits, ReferenceCheckpoint, ReferenceRuleset, ReferenceSimulation,
    CHECKPOINT_FORMAT_VERSION,
};
use sha2::{Digest, Sha256};

fn rehash(bytes: &mut [u8]) {
    let n = bytes.len() - 32;
    let domain = b"blob.simulation.checkpoint";
    let mut hash = Sha256::new();
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain);
    hash.update(CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    hash.update((n as u64).to_le_bytes());
    hash.update(&bytes[..n]);
    bytes[n..].copy_from_slice(&hash.finalize());
}

#[test]
fn checkpoint_rejects_unbounded_dimensions_before_allocation() {
    let simulation = ReferenceSimulation::new(1, 1, ReferenceRuleset::default()).unwrap();
    let mut bytes = ReferenceCheckpoint::from_simulation(&simulation).to_bytes();
    bytes[10..18].copy_from_slice(&(1u64 << 60).to_le_bytes());
    rehash(&mut bytes);
    let limits = CheckpointLimits {
        max_tiles: 1,
        max_cells: 1,
        ..Default::default()
    };
    assert!(ReferenceCheckpoint::from_bytes_with_limits(&bytes, limits).is_err());
}

#[test]
fn checkpoint_rejects_unbounded_cell_keys_before_allocation() {
    let rules = ReferenceRuleset::default();
    let mut simulation = ReferenceSimulation::new(1, 1, rules.clone()).unwrap();
    simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let mut state = simulation.canonical_state();
    let sentinel = 0xabcdu64;
    state.cells[0].0 = CellKey(sentinel);
    state.tiles[0].occupant = Some(CellKey(sentinel));
    state.next_cell_key = sentinel + 1;
    let simulation = ReferenceSimulation::from_canonical_state(1, 1, rules, state).unwrap();
    let mut bytes = ReferenceCheckpoint::from_simulation(&simulation).to_bytes();
    let offsets: Vec<_> = bytes
        .windows(8)
        .enumerate()
        .filter_map(|(i, w)| (w == sentinel.to_le_bytes()).then_some(i))
        .collect();
    assert_eq!(offsets.len(), 2);
    for i in offsets {
        bytes[i..i + 8].copy_from_slice(&(u64::MAX - 1).to_le_bytes());
    }
    rehash(&mut bytes);
    let limits = CheckpointLimits {
        max_tiles: 1,
        max_cells: 1,
        ..Default::default()
    };
    assert!(ReferenceCheckpoint::from_bytes_with_limits(&bytes, limits).is_err());
}

#[test]
fn sparse_old_survivor_uses_separate_lifetime_limit() {
    let rules = ReferenceRuleset::default();
    let mut simulation = ReferenceSimulation::new(1, 1, rules.clone()).unwrap();
    simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let mut state = simulation.canonical_state();
    state.cells[0].0 = CellKey(43980);
    state.tiles[0].occupant = Some(CellKey(43980));
    state.next_cell_key = 43981;
    let simulation = ReferenceSimulation::from_canonical_state(1, 1, rules, state).unwrap();
    let mut bytes = ReferenceCheckpoint::from_simulation(&simulation).to_bytes();
    let limits = CheckpointLimits {
        max_tiles: 1,
        max_cells: 1,
        max_cell_slots: 43981,
        ..Default::default()
    };
    assert!(ReferenceCheckpoint::from_bytes_with_limits(&bytes, limits).is_ok());
    assert!(ReferenceCheckpoint::from_bytes_with_limits(
        &bytes,
        CheckpointLimits {
            max_cell_slots: 43980,
            ..limits
        }
    )
    .is_err());
    let offsets: Vec<_> = bytes
        .windows(8)
        .enumerate()
        .filter_map(|(i, w)| (w == 43981u64.to_le_bytes()).then_some(i))
        .collect();
    assert_eq!(offsets.len(), 1);
    let i = offsets[0];
    bytes[i..i + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    rehash(&mut bytes);
    assert!(ReferenceCheckpoint::from_bytes_with_limits(&bytes, limits).is_err());
}
