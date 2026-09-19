//! Shared diagnostic scoring and independent scalar evaluation, not a runtime ABI.
use crate::{data::Row, model, scalar};
use blob_policy::{observation::OBS_DIM, runtime::FrozenPolicy};
use burn::{backend::NdArray, prelude::*};
use serde_json::{json, Value};
pub fn context<'a>(arm: &str, index: usize, rows: &'a [Row], donors: &[usize]) -> &'a [f32] {
    match arm {
        "observation" => &[0.; 128],
        "memory" => &rows[index].memory,
        "shuffled" => &rows[donors[index]].memory,
        _ => panic!("unknown diagnostic arm"),
    }
}
pub fn resolved_context<'a>(
    arm: &str,
    index: usize,
    rows: &'a [Row],
    donors: &[usize],
    overrides: Option<&'a [Vec<f32>]>,
) -> &'a [f32] {
    overrides
        .map(|values| values[index].as_slice())
        .unwrap_or_else(|| context(arm, index, rows, donors))
}
pub fn background(r: &Row) -> Vec<f32> {
    (0..10)
        .map(|k| (r.base_kind[k] + r.context_delta[k]) + r.slot_delta[k])
        .collect()
}
pub fn scores<B: Backend>(
    m: &model::Residual<B>,
    rows: &[Row],
    ids: &[usize],
    arm: &str,
    donors: &[usize],
    mask: bool,
) -> Tensor<B, 2> {
    scores_with_contexts(m, rows, ids, arm, donors, mask, None)
}
pub fn scores_with_contexts<B: Backend>(
    m: &model::Residual<B>,
    rows: &[Row],
    ids: &[usize],
    arm: &str,
    donors: &[usize],
    mask: bool,
    overrides: Option<&[Vec<f32>]>,
) -> Tensor<B, 2> {
    let d = m.head.weight.device();
    let n = ids.len();
    let obs = Tensor::from_data(
        TensorData::new(
            ids.iter()
                .flat_map(|&i| rows[i].observation.clone())
                .collect::<Vec<_>>(),
            [n, OBS_DIM],
        ),
        &d,
    );
    let ctx = Tensor::from_data(
        TensorData::new(
            ids.iter()
                .flat_map(|&i| {
                    resolved_context(arm, i, rows, donors, overrides)
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>(),
            [n, 128],
        ),
        &d,
    );
    let result = m.forward(obs, ctx)
        + Tensor::from_data(
            TensorData::new(
                ids.iter()
                    .flat_map(|&i| background(&rows[i]))
                    .collect::<Vec<_>>(),
                [n, 10],
            ),
            &d,
        );
    if mask {
        result
            + Tensor::from_data(
                TensorData::new(
                    ids.iter()
                        .flat_map(|&i| {
                            rows[i]
                                .legal_kinds
                                .iter()
                                .map(|&v| if v { 0. } else { -1e9 })
                        })
                        .collect::<Vec<_>>(),
                    [n, 10],
                ),
                &d,
            )
    } else {
        result
    }
}
pub fn choose(values: &[f32], legal: &[bool]) -> usize {
    (0..10)
        .filter(|&i| legal[i])
        .max_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
pub struct Evaluation<'a> {
    pub rows: &'a [Row],
    pub combat: usize,
    pub donors: &'a [usize],
    pub parent: &'a FrozenPolicy,
}
impl Evaluation<'_> {
    pub fn evaluate(
        &self,
        frozen: &scalar::Frozen,
        burn: Option<&model::Residual<NdArray>>,
        arm: &str,
    ) -> Value {
        self.evaluate_with_contexts(frozen, burn, arm, None)
    }
    pub fn evaluate_with_contexts(
        &self,
        frozen: &scalar::Frozen,
        burn: Option<&model::Residual<NdArray>>,
        arm: &str,
        overrides: Option<&[Vec<f32>]>,
    ) -> Value {
        let mut results = vec![];
        for (start, end) in [(0, self.combat), (self.combat, self.rows.len())] {
            let mut predictions = vec![];
            let (mut correct, mut attacks, mut hits, mut mismatches) = (0, 0, 0, 0);
            let mut max_difference = 0_f32;
            for ids in (start..end).collect::<Vec<_>>().chunks(128) {
                let trained = burn.map(|m| {
                    burn::tensor::activation::softmax(
                        if overrides.is_some() {
                            scores_with_contexts(
                                m,
                                self.rows,
                                ids,
                                arm,
                                self.donors,
                                false,
                                overrides,
                            )
                        } else {
                            scores(m, self.rows, ids, arm, self.donors, false)
                        },
                        1,
                    )
                    .clamp_min(1e-20)
                    .log()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                });
                for (j, &i) in ids.iter().enumerate() {
                    let r = &self.rows[i];
                    let delta = frozen
                        .forward(
                            &r.observation,
                            resolved_context(arm, i, self.rows, self.donors, overrides),
                        )
                        .unwrap();
                    let output = self
                        .parent
                        .forward_with_kind_delta(&r.observation, &r.memory, &delta)
                        .unwrap();
                    let k = choose(&output.kind, &r.legal_kinds);
                    let label = if i < self.combat {
                        r.teacher_kind
                    } else {
                        r.selected_kind
                    };
                    correct += usize::from(k == label);
                    attacks += usize::from(label == 4);
                    hits += usize::from(k == 4 && label == 4);
                    predictions.push(k);
                    if let Some(ref trained) = trained {
                        let values = &trained[j * 10..j * 10 + 10];
                        mismatches += usize::from(k != choose(values, &r.legal_kinds));
                        for (&a, &b) in values.iter().zip(&output.kind) {
                            max_difference = max_difference.max((a - b).abs());
                        }
                    }
                }
            }
            results.push(json!({"rows":end-start,"correct":correct,"agreement":correct as f64/(end-start) as f64,"attack_labels":attacks,"attack_correct":hits,"attack_agreement":if attacks==0 {1.} else {hits as f64/attacks as f64},"scalar_burn_mismatches":if burn.is_some(){json!(mismatches)}else{Value::Null},"max_log_probability_difference":if burn.is_some(){json!(max_difference)}else{Value::Null},"predictions":predictions}));
        }
        json!({"combat":results[0],"feeding":results[1],"training_fit_thresholds_met":results[0]["agreement"].as_f64().unwrap()>=0.95 && results[0]["attack_agreement"].as_f64().unwrap()>=0.9 && results[1]["agreement"].as_f64().unwrap()>=0.99})
    }
}
