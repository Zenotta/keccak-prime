//! Implements the expansion function.

use crate::fortuna::*;
use crate::prime::KeccakPrimeError;

/// Hash inputs sizes, in bytes.
pub const INPUT_HASH_SIZE: usize = 32; // 256 bits

/// Input nonce size, in bytes.
pub const NONCE_SIZE: usize = 8; // 64 bits

/// Takes a previous hash, root merkle hash and nonce as an input.
/// Outputs a byte sequence of length `output_size` (in bytes) suitable to be used in a VDF permutation function.
pub fn expand(
    prev_hash: [u8; INPUT_HASH_SIZE],
    root_hash: [u8; INPUT_HASH_SIZE],
    nonce: [u8; NONCE_SIZE],
    output_size: usize,
) -> Result<Vec<u8>, KeccakPrimeError> {
    // Derive an AES key from the previous & Merkle tree hashes.
    let derived_key = derive_aes_key(prev_hash, root_hash);
    let usage = 256 * (u64::from_be_bytes(nonce) as u128) + 256 - 1;
    let mut fortuna = Fortuna::new(&derived_key, usage)?;
    let result = fortuna.get_bytes(output_size)?;
    Ok(result)
}

/// Derives a symmetric encryption key for the AES-256 block cipher.
fn derive_aes_key(prev_hash: [u8; INPUT_HASH_SIZE], root_hash: [u8; INPUT_HASH_SIZE]) -> [u8; 32] {
    let mut xor_result = [0u8; INPUT_HASH_SIZE];
    for i in 0..INPUT_HASH_SIZE {
        xor_result[i] = prev_hash[i] ^ root_hash[i];
    }
    xor_result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify that the output has an expected size.
    #[test]
    fn verify_output_size() {
        let prev_hash = [1u8; INPUT_HASH_SIZE];
        let root_hash = [2u8; INPUT_HASH_SIZE];
        let nonce = [3u8; NONCE_SIZE];

        let output_size = 136; // 1088 bits

        let res = expand(prev_hash, root_hash, nonce, output_size).expect("expand function failed");
        dbg!(&res);
        assert_eq!(res.len(), output_size);
    }
}
