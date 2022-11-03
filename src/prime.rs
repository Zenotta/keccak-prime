//! Implements the Keccak-prime function.

use crate::{
    expansion::{expand, INPUT_HASH_SIZE, NONCE_SIZE},
    keccak::Keccak,
    sloth, Hasher,
};
use num_bigint::BigUint;
use std::error::Error;
use std::fmt;

/// Size of output for the expansion function.
/// Can be changed into a parameter for `prime` if needed.
const EXPANSION_OUTPUT_SIZE: usize = 136; // 1088 bits

/// Keccak-prime function.
///
/// ### Arguments
/// - `prev_hash`: previous block hash.
/// - `root_hash`: Merkle root hash.
/// - `nonce`: block nonce.
/// - `penalty`: applied penalty (regulates a number of extra Keccak permutations).
/// - `delay`: delay parameter used in the VDF function.
/// - `vdf_iterations`: a number of VDF iterations.
pub fn prime(
    prev_hash: [u8; INPUT_HASH_SIZE],
    root_hash: [u8; INPUT_HASH_SIZE],
    nonce: [u8; NONCE_SIZE],
    penalty: usize,
    delay: u64,
    vdf_iterations: usize,
) -> Result<[u8; INPUT_HASH_SIZE], KeccakPrimeError> {
    // Expand the block.
    let block = expand(prev_hash, root_hash, nonce, EXPANSION_OUTPUT_SIZE)?;

    // Execute a chain of VDFs.
    let mut vdf_output = BigUint::from_bytes_be(&block);
    for _i in 0..vdf_iterations {
        vdf_output = sloth::solve(vdf_output, delay);
    }

    let vdf_output_bytes = vdf_output.to_bytes_be();

    // Construct a Keccak function with rate=1088 and capacity=512.
    let mut keccak = Keccak::new(1088 / 8);
    keccak.update(&vdf_output_bytes);
    Ok(keccak.finalize_with_penalty(penalty))
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
    use super::prime;
    use crate::expansion::{INPUT_HASH_SIZE, NONCE_SIZE};

    #[test]
    fn keccak_prime_test() {
        let prev_hash = [1u8; INPUT_HASH_SIZE];
        let root_hash = [2u8; INPUT_HASH_SIZE];
        let nonce = [3u8; NONCE_SIZE];

        dbg!(prime(prev_hash, root_hash, nonce, 100, 100, 10)
            .expect("Failed to execute Keccak-prime"));
    }
}
