#![no_main]

use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, capnp_to_reference_mind_decision, capnp_to_reference_mind_input,
    reference_mind_decision_to_capnp, reference_mind_input_to_capnp,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let limits = ReferenceMindLimits {
        max_input_bytes: 4 * 1024,
        max_action_bytes: 4 * 1024,
        max_slots: 32,
        max_private_memory_bytes: 512,
    };

    if let Ok(input) = capnp_to_reference_mind_input(bytes, limits) {
        let encoded = reference_mind_input_to_capnp(&input, limits)
            .expect("an accepted Mind input must re-encode");
        let decoded = capnp_to_reference_mind_input(&encoded, limits)
            .expect("a re-encoded Mind input must decode");
        assert_eq!(decoded, input);
    }
    if let Ok(decision) = capnp_to_reference_mind_decision(bytes, limits) {
        let encoded = reference_mind_decision_to_capnp(&decision, limits)
            .expect("an accepted Mind decision must re-encode");
        let decoded = capnp_to_reference_mind_decision(&encoded, limits)
            .expect("a re-encoded Mind decision must decode");
        assert_eq!(decoded, decision);
    }
});
