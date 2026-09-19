#[path = "row.rs"]
mod row;
use blob_rl::deployed_artifact::read_bounded;
use blob_rl::deployed_artifact::sha;
pub use row::{read, Row};
use serde_json::{json, Value};
use std::path::Path;
pub struct Corpus {
    pub combat: Vec<Row>,
    pub feeding: Vec<Row>,
    pub audit: Value,
}
pub fn load(root: &Path, seed: u64) -> Corpus {
    let dir = root.join(seed.to_string());
    assert_eq!(read(&dir.join("status.json"))["complete"], true);
    let mut combat = vec![];
    let mut feeding = vec![];
    let mut audit = serde_json::Map::new();
    for file in [
        "combat.json",
        "feeding-Line.json",
        "feeding-Checkerboard.json",
        "feeding-Ring.json",
        "feeding-LooseRandom.json",
        "feeding-Random.json",
    ] {
        let bytes = read_bounded(&dir.join(file), 128 * 1024 * 1024).unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let rows: Vec<Row> = serde_json::from_value(v["rows"].clone()).unwrap();
        for r in &rows {
            assert_eq!(r.context, "Interaction");
            assert_eq!(r.hidden.len(), 128);
            assert_eq!(r.legal_kinds.len(), 10);
            assert!(r.legal_kinds[r.selected_kind] && r.legal_kinds[r.teacher_kind]);
        }
        audit.insert(file.into(), json!({"sha256":sha(&bytes),"rows":rows.len()}));
        if file == "combat.json" {
            combat.extend(rows);
        } else {
            feeding.extend(rows);
        }
    }
    assert!(!combat.is_empty() && !feeding.is_empty());
    Corpus {
        combat,
        feeding,
        audit: json!(audit),
    }
}
