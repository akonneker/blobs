//! Canonical quantized cell-private policy memory.
const POLICY_MEMORY_MAGIC: [u8; 4] = *b"BRM1";
const POLICY_MEMORY_HEADER_BYTES: usize = 8;

pub fn policy_memory_bytes(recurrent_size: usize) -> Option<usize> {
    recurrent_size
        .checked_mul(std::mem::size_of::<i16>())
        .and_then(|bytes| bytes.checked_add(POLICY_MEMORY_HEADER_BYTES))
}

/// Decode only this policy's exact versioned memory format. Empty or malformed
/// memory deterministically initializes a fresh zero state.
pub fn decode_policy_memory(bytes: &[u8], recurrent_size: usize) -> Vec<f32> {
    let Some(expected) = policy_memory_bytes(recurrent_size) else {
        return vec![0.0; recurrent_size];
    };
    if bytes.len() != expected || bytes[..4] != POLICY_MEMORY_MAGIC {
        return vec![0.0; recurrent_size];
    }
    let declared = u32::from_le_bytes(bytes[4..8].try_into().expect("fixed memory header"));
    if usize::try_from(declared).ok() != Some(recurrent_size) {
        return vec![0.0; recurrent_size];
    }
    bytes[POLICY_MEMORY_HEADER_BYTES..]
        .chunks_exact(2)
        .map(|chunk| {
            f32::from(i16::from_le_bytes(
                chunk.try_into().expect("two-byte memory element"),
            )) / f32::from(i16::MAX)
        })
        .collect()
}

pub fn encode_policy_memory(memory: &[f32]) -> Vec<u8> {
    let declared = u32::try_from(memory.len()).expect("validated recurrent state fits u32");
    let mut bytes = Vec::with_capacity(
        policy_memory_bytes(memory.len()).expect("validated recurrent state byte size"),
    );
    bytes.extend_from_slice(&POLICY_MEMORY_MAGIC);
    bytes.extend_from_slice(&declared.to_le_bytes());
    for value in memory {
        let value = if value.is_finite() { *value } else { 0.0 };
        let quantized = (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    bytes
}
