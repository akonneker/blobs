//! Ordinary-ABI weight/choice parity for an explicit untrained sampling canary.
#[path = "../src/extism_compat.rs"]
mod extism_compat;
#[path = "../../blob_policy/examples/sampled_target_fixture/mod.rs"]
mod fixture;

use blob_engine::{mind_runtime::inspect_mind_artifact, resolution::MindRuntimeProfile};
use blob_interface::{
    randomness::PrivateRandom,
    reference_mind::{ReferenceMindDecision, ReferenceMindInput},
    reference_mind_converter::*,
};
use blob_policy::{
    action::{PolicyActionKind, action_is_commit_legal},
    sampling::{TARGET_SAMPLING_CONTRACT, TargetDistribution},
};
use extism::{Manifest, Wasm};
use extism_manifest::MemoryOptions;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::PathBuf,
    time::{Duration, Instant},
};

fn words(input: &ReferenceMindInput) -> Vec<u64> {
    let distribution =
        TargetDistribution::from_input(input, PolicyActionKind::Move, &fixture::logits(input))
            .unwrap();
    let mut words = BTreeSet::from([0, 1, 1 << 63, u64::MAX]);
    let total = distribution.categorical().total();
    let mut sum = 0;
    for weight in distribution.categorical().weights() {
        sum += weight;
        if sum > 0 && sum < total {
            let boundary = ((u128::from(sum) << 64).div_ceil(u128::from(total))) as u64;
            words.insert(boundary - 1);
            words.insert(boundary);
            if let Some(after) = boundary.checked_add(1) {
                words.insert(after);
            }
        }
    }
    words.into_iter().collect()
}

fn check(
    executor: &extism_compat::ExtismCompatExecutor,
    input: &ReferenceMindInput,
) -> Result<(Vec<u8>, ReferenceMindDecision), Box<dyn std::error::Error>> {
    let limits = ReferenceMindLimits::default();
    let bytes = reference_mind_input_to_capnp(input, limits)?;
    let native = fixture::decide(input)?;
    assert!(action_is_commit_legal(input, &native.action, false));
    let output = executor.call(bytes, limits.max_action_bytes)?;
    let actual = capnp_to_reference_mind_decision(&output, limits)?;
    assert_eq!(
        actual, native,
        "native/WASM target, quantized weight, or memory mismatch"
    );
    Ok((output, native))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let wasm_path = PathBuf::from(args.next().ok_or("expected canary WASM path")?);
    let output = PathBuf::from(args.next().ok_or("expected new output directory")?);
    if args.next().is_some() {
        return Err("expected exactly two arguments".into());
    }
    fs::create_dir(&output)?;
    let mut wasm = Vec::new();
    fs::File::open(&wasm_path)?
        .take(4 * 1024 * 1024 + 1)
        .read_to_end(&mut wasm)?;
    if wasm.len() > 4 * 1024 * 1024 {
        return Err("canary exceeds 4 MiB artifact budget".into());
    }
    inspect_mind_artifact(&wasm, MindRuntimeProfile::ExtismPdkDeterministicV1)?;
    let manifest = Manifest::new([Wasm::data(wasm.clone())])
        .with_memory_options(
            MemoryOptions::new()
                .with_max_pages(128)
                .with_max_var_bytes(0),
        )
        .disallow_all_hosts()
        .with_timeout(Duration::from_secs(2));
    let executor = extism_compat::ExtismCompatExecutor::new(&manifest, wasmtime::Config::new())?;
    let source_files: &[(&str, &[u8])] = &[
        (
            "blob_policy/src/sampling.rs",
            include_bytes!("../../blob_policy/src/sampling.rs"),
        ),
        (
            "blob_policy/examples/sampled_target_canary.rs",
            include_bytes!("../../blob_policy/examples/sampled_target_canary.rs"),
        ),
        (
            "blob_policy/examples/sampled_target_fixture/mod.rs",
            include_bytes!("../../blob_policy/examples/sampled_target_fixture/mod.rs"),
        ),
        (
            "blob_game/examples/verify_sampled_target.rs",
            include_bytes!("../../blob_game/examples/verify_sampled_target.rs"),
        ),
        (
            "blob_interface/src/reference_mind_converter.rs",
            include_bytes!("../../blob_interface/src/reference_mind_converter.rs"),
        ),
        (
            "blob_policy/src/action.rs",
            include_bytes!("../../blob_policy/src/action.rs"),
        ),
        ("Cargo.lock", include_bytes!("../../Cargo.lock")),
    ];
    let mut hashes = std::collections::BTreeMap::new();
    for (name, bytes) in source_files {
        hashes.insert(name, format!("{:x}", Sha256::digest(bytes)));
        let path = output.join("sources").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, bytes)?;
    }
    let plan = serde_json::json!({"contract":TARGET_SAMPLING_CONTRACT,"scope":"Untrained decoder canary; exact weights and choices over ordinary ABI. No learned checkpoint, ecological or fuel qualification.","fixture":"0..32 slots, equal/weighted/masked/unaffordable/renumbered cases; all 255 eight-slot masks; exact CDF boundary +/-1 and endpoints","source_sha256":hashes,"source_scope":"Compiled native verifier source snapshots; the supplied WASM is independently identified by its digest.","wasm_sha256":format!("{:x}",Sha256::digest(&wasm)),"abi":blob_interface::abi::reference_mind_abi_hash(),"engine_seeds_used":[]});
    fs::write(output.join("plan.json"), serde_json::to_vec_pretty(&plan)?)?;
    fs::write(output.join("canary.wasm"), wasm)?;
    let start = Instant::now();
    let mut cases = Vec::new();
    let mut digest = Sha256::new();
    let mut decisions = 0;
    let mut isolated = 0;
    for count in 0..=32 {
        for variant in 0..5 {
            let mut input = fixture::input(count);
            match variant {
                0 => {
                    for slot in &mut input.slots {
                        slot.plant_energy = Some(8);
                    }
                }
                1 => {}
                2 => {
                    input.action_space.move_targets &= 0x5555_5555;
                    for slot in &mut input.slots {
                        if slot.slot % 3 == 0 {
                            slot.reachable = false;
                        }
                    }
                }
                3 => input.self_state.assimilated_energy = 1,
                4 => {
                    input.slots.reverse();
                    for (index, slot) in input.slots.iter_mut().enumerate() {
                        slot.slot = index as u8;
                    }
                }
                _ => unreachable!(),
            }
            let words = words(&input);
            for word in &words {
                let mut random = [0xa5; 32];
                random[..8].copy_from_slice(&word.to_le_bytes());
                input.randomness = PrivateRandom::from_bytes(random);
                let (bytes, _) = check(&executor, &input)?;
                digest.update((bytes.len() as u64).to_le_bytes());
                digest.update(&bytes);
                decisions += 1;
            }
            let (before, _) = check(&executor, &input)?;
            let mut unrelated = fixture::input(1);
            unrelated.private_memory = vec![17; 73];
            check(&executor, &unrelated)?;
            let (after, _) = check(&executor, &input)?;
            assert_eq!(before, after, "unrelated invocation leaked state");
            isolated += 1;
            decisions += 3;
            cases.push(
                serde_json::json!({"slots":count,"variant":variant,"boundary_words":words.len()}),
            );
        }
    }
    let mut subset_decisions = 0;
    for mask in 1..=255 {
        let mut input = fixture::input(8);
        input.action_space.move_targets = mask;
        for slot in &mut input.slots {
            slot.plant_energy = Some(8);
        }
        for word in words(&input) {
            let mut random = [0; 32];
            random[..8].copy_from_slice(&word.to_le_bytes());
            input.randomness = PrivateRandom::from_bytes(random);
            let (bytes, _) = check(&executor, &input)?;
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
            subset_decisions += 1;
            decisions += 1;
        }
    }
    let limits = ReferenceMindLimits::default();
    assert!(
        executor
            .call(vec![0xff; 17], limits.max_action_bytes)
            .is_err()
    );
    for duplicate in [false, true] {
        let mut input = fixture::input(8);
        let (dx, dy) = if duplicate {
            (input.slots[1].dx, input.slots[1].dy)
        } else {
            (0, 0)
        };
        input.slots[0].dx = dx;
        input.slots[0].dy = dy;
        assert!(fixture::decide(&input).is_err());
        assert!(
            executor
                .call(
                    reference_mind_input_to_capnp(&input, limits)?,
                    limits.max_action_bytes
                )
                .is_err()
        );
    }
    let report = serde_json::json!({"complete":true,"plan":plan,"decisions":decisions,"subset_decisions":subset_decisions,"isolation_sequences":isolated,"invalid_inputs_rejected":3,"ordered_boundary_output_sha256":format!("{:x}",digest.finalize()),"cases":cases,"elapsed_seconds":start.elapsed().as_secs_f64(),"weight_and_choice_mismatches":0});
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{decisions} native/WASM decisions and weight vectors matched; {isolated} isolation sequences; 3 invalid inputs rejected"
    );
    Ok(())
}
