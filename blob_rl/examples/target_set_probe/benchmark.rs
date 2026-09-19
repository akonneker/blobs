//! Native inference cost and 32-slot correctness; not WASM fuel qualification.
use super::*;
use std::time::Instant;

fn inputs(batch_size: usize, slots: usize) -> Batch<NdArray> {
    let device = Default::default();
    let mut local = Vec::new();
    for _ in 0..batch_size {
        for slot in 0..slots {
            let geometry = if slots == 8 {
                *GEOMETRY.get(slot).unwrap()
            } else {
                // Distinct relative positions; these are timing tensors, not ABI fixtures.
                [(slot % 6) as f32 - 3., (slot / 6) as f32 - 3.]
            };
            local.extend([geometry[0], geometry[1], (slot % 3 + 1) as f32 / 3.]);
        }
    }
    Batch {
        local: Tensor::from_data(TensorData::new(local, [batch_size, slots, 3]), &device),
        random: Tensor::from_data(
            TensorData::new(vec![0.5; batch_size * 33], [batch_size, 1, 33]),
            &device,
        ),
        positive: Tensor::zeros([batch_size, slots, 3], &device),
        mask: Tensor::zeros([batch_size, slots], &device),
        targets: Tensor::zeros([batch_size], &device),
        marginal_targets: Tensor::zeros([batch_size, slots], &device),
    }
}

pub(super) fn run() -> Value {
    let device = Default::default();
    NdArray::<f32>::seed(&device, INITIALIZATIONS[0]);
    let mut model = SetProbe::<NdArray>::new(&device);
    model.head = nn::LinearConfig::new(64, 1).init(&device);
    model.interaction = nn::LinearConfig::new(16, 64).with_bias(false).init(&device);
    let mut results = Vec::new();
    for batch_size in [1, 32] {
        for slots in [8, 32] {
            for mode in [
                Mode::LearnedSum,
                Mode::Relational,
                Mode::RelationalProduct,
                Mode::QuantileUtility,
            ] {
                let mut times = Vec::new();
                for repetition in 0..106 {
                    let start = Instant::now();
                    let values = model
                        .forward(inputs(batch_size, slots), mode)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    std::hint::black_box(&values);
                    assert!(values.iter().all(|v| v.is_finite()));
                    if matches!(mode, Mode::QuantileUtility) {
                        let legal = vec![true; slots];
                        for row in values.chunks_exact(slots) {
                            std::hint::black_box(sampling::sample(row, &legal, 1 << 63).unwrap());
                        }
                    }
                    if repetition >= 5 {
                        times.push(start.elapsed().as_secs_f64() * 1000.);
                    }
                }
                times.sort_by(f64::total_cmp);
                results.push(json!({"batch":batch_size,"slots":slots,"mode":format!("{mode:?}"),
                    "samples":times.len(),"median_ms":times[50],"p95_ms":times[95],
                    "pair_feature_tensor_bytes":if matches!(mode,Mode::Relational | Mode::RelationalProduct) {batch_size*slots*slots*16*4} else {0},
                    "single_cell_32_slot_cost_gate":if batch_size==1 && slots==32 {Some(times[95]<10.)} else {None}}));
            }
        }
    }
    json!({"scope":"Native NdArray CPU, input tensor construction and output materialization included. Five warmups, 101 timed calls. Pair-feature bytes are one tensor, not peak RSS. 32-slot fixtures measure cost and shape support, not learned behavior or WASM fuel.","results":results})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn thirty_two_slot_relational_scores_permute_exactly() {
        for mode in [Mode::Relational, Mode::RelationalProduct] {
            check_permutation(mode);
        }
    }
    fn check_permutation(mode: Mode) {
        let device = Default::default();
        let mut model = SetProbe::<NdArray>::new(&device);
        model.head = nn::LinearConfig::new(64, 1).init(&device);
        model.interaction = nn::LinearConfig::new(16, 64).with_bias(false).init(&device);
        let original = model
            .forward(inputs(2, 32), mode)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut permuted = inputs(2, 32);
        let indices = Tensor::<NdArray, 1, Int>::from_data(
            TensorData::new((0..32).rev().collect::<Vec<i32>>(), [32]),
            &device,
        );
        permuted.local = permuted.local.select(1, indices.clone());
        permuted.positive = permuted.positive.select(1, indices.clone());
        permuted.mask = permuted.mask.select(1, indices);
        let after = model
            .forward(permuted, mode)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for row in 0..2 {
            for slot in 0..32 {
                assert_eq!(
                    original[row * 32 + slot].to_bits(),
                    after[row * 32 + 31 - slot].to_bits()
                );
            }
        }
    }
}
