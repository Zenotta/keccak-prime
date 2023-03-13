//! Implements the Keccak-prime function.

use crate::{
    constants::{BLOCK_HEADER_SIZE, INPUT_HASH_SIZE, MAX, NONCE_SIZE},
    expansion::expand,
    keccak::Keccak,
    prf::prf,
    sloth, Hasher,
};
use std::fmt;
use std::{convert::TryInto, error::Error};

/// Keccak-prime block linking function.
///
/// ### Arguments
/// - `header_of_block_k`
/// - `prev_hash_of_prev_block`
/// - `prev_hash`: previous block hash.
/// - `root_hash`: Merkle root hash.
/// - `nonce`: block nonce.
/// - `penalty`: applied penalty (regulates a number of extra Keccak permutations).
/// - `delay`: delay parameter used in the VDF function.
///
/// ### Returns
/// - `witness` number. It can be verified using the [crate::vdf::verify] function.
/// - `output_hash`.
pub fn link_blocks(
    header_of_block_k: [u8; BLOCK_HEADER_SIZE],
    prev_hash_of_prev_block: [u8; INPUT_HASH_SIZE],
    prev_hash: [u8; INPUT_HASH_SIZE],
    root_hash: [u8; INPUT_HASH_SIZE],
    nonce: [u8; NONCE_SIZE],
    penalty: u16,
    delay: u16,
    loop_count: u16,
) -> Result<(Vec<u8>, [u8; INPUT_HASH_SIZE]), KeccakPrimeError> {
    // Expand the block header to the VDF domain.
    let vdf_input = expand(
        prev_hash,
        root_hash,
        nonce,
        penalty,
        delay,
        loop_count,
        MAX.clone(),
    );

    // Execute VDF.
    let witness = sloth::solve(vdf_input, delay);
    let witness_bytes = witness.to_bytes_be();

    // Call PRF.
    let y = prf(
        &header_of_block_k,
        &witness_bytes[0..200].try_into().unwrap(), // FIXME
    );

    // Hash the results
    let mut hash_bytes = Vec::with_capacity(y.len() + INPUT_HASH_SIZE * 3 + NONCE_SIZE);
    hash_bytes.extend_from_slice(&y);
    hash_bytes.extend_from_slice(&prev_hash);
    hash_bytes.extend_from_slice(&root_hash);
    hash_bytes.extend_from_slice(&delay.to_le_bytes());
    hash_bytes.extend_from_slice(&penalty.to_le_bytes());
    hash_bytes.extend_from_slice(&nonce);
    hash_bytes.extend_from_slice(&prev_hash_of_prev_block);

    // Construct a Keccak function with rate=1088 and capacity=512.
    let mut keccak = Keccak::new(1088 / 8);
    keccak.update(&hash_bytes);

    Ok((witness_bytes, keccak.finalize_with_penalty(penalty)))
}

/// Keccak-prime error.
#[derive(Debug)]
pub enum KeccakPrimeError {
    /// Opaque AES function failure.
    AesError(aes_gcm_siv::aead::Error),
}

impl From<aes_gcm_siv::aead::Error> for KeccakPrimeError {
    fn from(e: aes_gcm_siv::aead::Error) -> Self {
        Self::AesError(e)
    }
}

impl fmt::Display for KeccakPrimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeccakPrimeError::AesError(e) => write!(f, "AES error: {}", e),
        }
    }
}

impl Error for KeccakPrimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            KeccakPrimeError::AesError(_err) => None, // aes_gcm_siv::Error doesn't implement the Error trait
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{INPUT_HASH_SIZE, NONCE_SIZE};

    #[test]
    fn keccak_prime_test() {
        let prev_hash = [1u8; INPUT_HASH_SIZE];
        let root_hash = [2u8; INPUT_HASH_SIZE];
        let nonce = [3u8; NONCE_SIZE];

        let header_of_block_k = [4; BLOCK_HEADER_SIZE];
        let prev_hash_of_prev_block = [5; INPUT_HASH_SIZE];

        dbg!(link_blocks(
            header_of_block_k,
            prev_hash_of_prev_block,
            prev_hash,
            root_hash,
            nonce,
            100,
            100,
            100
        )
        .expect("Failed to execute Keccak-prime"));
    }
}
