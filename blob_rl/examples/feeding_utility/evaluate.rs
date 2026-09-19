use super::{
    data::{Row, LAYOUTS},
    model::{batch, Utility},
};
use blob_policy::sampling::CategoricalDistribution;
use blob_rl::observation::SLOT_FEATURES;
use burn::backend::NdArray;
use serde_json::{json, Value};

pub fn predict(scores: &[f32], row: &Row) -> usize {
    let ordered: Vec<_> = row.geometry_order.iter().map(|i| scores[*i]).collect();
    let legal: Vec<_> = row.geometry_order.iter().map(|i| row.legal[*i]).collect();
    row.geometry_order[CategoricalDistribution::from_logits(&ordered, &legal)
        .unwrap()
        .sample(row.word)
        .unwrap()]
}
pub fn evaluate(model: &Utility<NdArray>, rows: &[Row], control: bool) -> Value {
    let device = Default::default();
    let mut counts = [[0_usize; 8]; 5];
    for chunk in rows.chunks(128) {
        let (local, legal) = batch(chunk, &device);
        let scores = model
            .forward(local.clone(), legal.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let permutation: Vec<usize> = (0..32).rev().collect();
        let mut permuted = chunk.to_vec();
        for row in &mut permuted {
            row.local = permutation
                .iter()
                .flat_map(|i| {
                    row.local[i * SLOT_FEATURES..(i + 1) * SLOT_FEATURES]
                        .iter()
                        .copied()
                })
                .collect();
            row.legal = std::array::from_fn(|i| row.legal[permutation[i]]);
        }
        let (p_local, p_legal) = batch(&permuted, &device);
        let permuted_scores = model
            .forward(p_local, p_legal)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut reversed = chunk.to_vec();
        reversed.reverse();
        let (r_local, r_legal) = batch(&reversed, &device);
        let reversed_scores = model
            .forward(r_local, r_legal)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (index, row) in chunk.iter().enumerate() {
            let logits = &scores[index * 32..(index + 1) * 32];
            for (slot, score) in logits.iter().enumerate() {
                assert!(score.is_finite());
                assert_eq!(
                    score.to_bits(),
                    permuted_scores[index * 32 + 31 - slot].to_bits()
                );
                assert_eq!(
                    score.to_bits(),
                    reversed_scores[(chunk.len() - 1 - index) * 32 + slot].to_bits()
                );
            }
            let predicted = predict(logits, row);
            assert!(row.legal[predicted]);
            let expected = if control { row.control } else { row.treatment };
            let c = &mut counts[row.layout];
            c[0] += 1;
            c[1] += usize::from(predicted == expected);
            if row.active {
                c[2] += 1;
                c[3] += usize::from(predicted == expected);
            } else {
                c[4] += 1;
                c[5] += usize::from(predicted == row.control);
            }
            c[6] += usize::from(predicted == row.treatment);
            c[7] += usize::from(
                predicted != expected
                    && [7, 8, 9, 10, 13].iter().all(|feature| {
                        row.local[predicted * SLOT_FEATURES + feature].to_bits()
                            == row.local[expected * SLOT_FEATURES + feature].to_bits()
                    }),
            );
        }
    }
    let total: Vec<usize> = (0..8).map(|i| counts.iter().map(|c| c[i]).sum()).collect();
    let rate = |a: usize, b: usize| {
        if b == 0 {
            None
        } else {
            Some(a as f64 / b as f64)
        }
    };
    let overall = rate(total[1], total[0]).unwrap();
    let active = rate(total[3], total[2]).unwrap_or(0.);
    let retention = rate(total[5], total[4]).unwrap_or(0.);
    json!({"same_visible_food_occupancy_errors":total[7],"rows":total[0],"correct":total[1],"accuracy":overall,"active_rows":total[2],"active_correct":total[3],"active_accuracy":active,"retention_rows":total[4],"retention_correct":total[5],"retention_accuracy":retention,"common_treatment_correct":total[6],"common_treatment_accuracy":rate(total[6],total[0]),"gate_pass":overall>=0.95&&active>=0.90&&retention>=0.95,"slot_permutation_bit_exact":true,"row_permutation_bit_exact":true,"by_layout":counts.iter().enumerate().map(|(i,c)|json!({"layout":LAYOUTS[i],"rows":c[0],"correct":c[1],"accuracy":rate(c[1],c[0]),"active_rows":c[2],"active_correct":c[3],"retention_rows":c[4],"retention_correct":c[5]})).collect::<Vec<_>>()})
}
