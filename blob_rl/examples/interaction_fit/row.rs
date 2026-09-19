use blob_rl::deployed_artifact::read_bounded;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
#[derive(Clone, Deserialize)]
pub struct Row {
    pub context: String,
    pub selected_kind: usize,
    pub teacher_kind: usize,
    pub legal_kinds: Vec<bool>,
    pub observation: Vec<f32>,
    pub memory: Vec<f32>,
    pub hidden: Vec<f32>,
    pub context_delta: Vec<f32>,
    pub slot_delta: Vec<f32>,
    pub base_kind: Vec<f32>,
}
pub fn read(path: &Path) -> Value {
    serde_json::from_slice(&read_bounded(path, 128 * 1024 * 1024).unwrap()).unwrap()
}
