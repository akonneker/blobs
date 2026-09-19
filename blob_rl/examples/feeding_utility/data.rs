use blob_rl::{
    action::{
        compose_policy_action, decompose_policy_action, HierarchicalActionChoice, PolicyActionKind,
        NUM_POLICY_TARGETS,
    },
    demonstration::{load_demonstrations, verify_feeding_correction_pair},
    observation::{
        HEADER_FEATURES, OBS_RANDOMNESS_FEATURE_END, OBS_RANDOMNESS_FEATURE_START, SLOT_FEATURES,
    },
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

pub const LAYOUTS: [&str; 5] = ["line", "checkerboard", "ring", "loose-random", "random"];
#[derive(Clone)]
pub struct Row {
    pub seed: u64,
    pub layout: usize,
    pub local: Vec<f32>,
    pub legal: [bool; NUM_POLICY_TARGETS],
    pub geometry_order: Vec<usize>,
    pub word: u64,
    pub control: usize,
    pub treatment: usize,
    pub teacher: Option<usize>,
    pub active: bool,
}

pub fn load(root: &Path) -> Result<(Vec<Row>, Value), String> {
    let mut rows = Vec::new();
    let mut corpora = Vec::new();
    let mut all_rows = 0;
    let mut active = 0;
    for (layout, name) in LAYOUTS.iter().enumerate() {
        let control = load_demonstrations(&root.join(format!("{name}-control")))?;
        let treatment = load_demonstrations(&root.join(format!("{name}-treatment")))?;
        let changed = verify_feeding_correction_pair(&control, &treatment)?;
        active += changed;
        corpora.push(json!({"layout":name,"source_config_sha256":treatment.manifest.source_config_sha256,"effective_config_sha256":treatment.manifest.effective_config_sha256,"semantic_ruleset_hash":treatment.manifest.semantic_ruleset_hash,"compiled_ruleset_hash":treatment.manifest.compiled_ruleset_hash,"control_manifest_sha256":control.manifest_sha256,"treatment_manifest_sha256":treatment.manifest_sha256,"control_payload_sha256":control.manifest.payload_sha256,"treatment_payload_sha256":treatment.manifest.payload_sha256,"collection":treatment.manifest.collection,"seeds":treatment.manifest.seeds,"samples":treatment.payload.samples.len(),"active":changed}));
        all_rows += control.payload.samples.len();
        for (c, t) in control
            .payload
            .samples
            .iter()
            .zip(&treatment.payload.samples)
        {
            let c_choice =
                decompose_policy_action(usize::from(c.action)).ok_or("invalid control action")?;
            if c_choice.kind != PolicyActionKind::Move.index() {
                continue;
            }
            let t_choice =
                decompose_policy_action(usize::from(t.action)).ok_or("invalid treatment action")?;
            if t_choice.kind != c_choice.kind || t_choice.effort != c_choice.effort {
                return Err("pair changes more than Move target".into());
            }
            let teacher = decompose_policy_action(usize::from(
                t.correction_teacher_action.ok_or("missing teacher")?,
            ))
            .ok_or("invalid teacher action")?;
            let mut local = t.observation[HEADER_FEATURES..].to_vec();
            for slot in local.chunks_exact_mut(SLOT_FEATURES) {
                slot[2] *= 127.;
                slot[3] *= 127.;
            }
            let legal = std::array::from_fn(|target| {
                let action = compose_policy_action(HierarchicalActionChoice {
                    kind: c_choice.kind,
                    target,
                    effort: c_choice.effort,
                })
                .unwrap();
                t.action_mask[action]
            });
            if !legal[c_choice.target] || !legal[t_choice.target] {
                return Err("label violates frozen-effort target mask".into());
            }
            let mut geometry_order: Vec<_> = (0..NUM_POLICY_TARGETS)
                .filter(|slot| local[slot * SLOT_FEATURES] > 0.)
                .collect();
            geometry_order.sort_by(|&a, &b| {
                local[a * SLOT_FEATURES + 3]
                    .total_cmp(&local[b * SLOT_FEATURES + 3])
                    .then_with(|| {
                        local[a * SLOT_FEATURES + 2].total_cmp(&local[b * SLOT_FEATURES + 2])
                    })
            });
            let mut bytes = [0; 32];
            for (byte, &encoded) in bytes
                .iter_mut()
                .zip(&t.observation[OBS_RANDOMNESS_FEATURE_START..OBS_RANDOMNESS_FEATURE_END])
            {
                *byte = (encoded * 255.).round() as u8;
                if (f32::from(*byte) / 255.).to_bits() != encoded.to_bits() {
                    return Err("private byte projection is not exactly invertible".into());
                }
            }
            rows.push(Row {
                seed: t.source_seed,
                layout,
                local,
                legal,
                geometry_order,
                word: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                control: c_choice.target,
                treatment: t_choice.target,
                teacher: (teacher.kind == PolicyActionKind::Move.index()).then_some(teacher.target),
                active: t.active_correction,
            });
        }
    }
    let mut groups = BTreeMap::<(usize, u64), [usize; 4]>::new();
    for row in &rows {
        let g = groups.entry((row.layout, row.seed)).or_default();
        g[0] += 1;
        g[1] += usize::from(row.active);
        g[2] += usize::from(Some(row.treatment) == row.teacher);
        g[3] += usize::from(row.legal.iter().filter(|v| **v).count() == 1);
    }
    let groups:Vec<_>=groups.into_iter().map(|((layout,seed),counts)|json!({"layout":LAYOUTS[layout],"seed":seed,"move_rows":counts[0],"active":counts[1],"treatment_matches_teacher_target":counts[2],"singleton_legal":counts[3]})).collect();
    let audit = json!({"datasets":corpora,"all_rows":all_rows,"move_rows":rows.len(),"active":active,"groups":groups,"private_bytes_exactly_recovered":true,"fixed_effort_masks_legal":true,"non_target_rows_retained":all_rows-rows.len()});
    Ok((rows, audit))
}
