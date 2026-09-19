#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| blob_fuzz::movement::check(data));
