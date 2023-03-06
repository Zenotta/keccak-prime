//! Implements the expansion function.

use crate::constants::{BLOCK_HEADER_SIZE, INPUT_HASH_SIZE, NONCE_SIZE};
use crate::prng::{self, PrngState};

use num_bigint::BigUint;

use std::convert::TryInto;

const USAGE_NUMBER: u8 = 0xff;

/// Takes a previous hash, root merkle hash and nonce as an input.
/// Outputs a value in the domain of the VDF permutation function.
pub fn expand(
    prev_hash: [u8; INPUT_HASH_SIZE],
    root_hash: [u8; INPUT_HASH_SIZE],
    nonce: [u8; NONCE_SIZE],
    penalty: u32,
    delay: u32,
    max: BigUint,
) -> BigUint {
    // integer X chosen uniformly at random from the set of integers {0, 1, ... max}.
    let mut prng = PrngState::new(
        &derive_key(prev_hash, root_hash, nonce, penalty, delay),
        USAGE_NUMBER,
    );

    loop {
        let byte_array = prng.get_bytes(168); // 1344 bits = SHAKE-128 bit rate

        let x = BigUint::new(unsafe {
            let (_prefix, digits_u32, _suffix) = byte_array.align_to::<u32>();
            digits_u32.to_owned()
        });

        if x <= max {
            return x;
        }
    }
}

/// Derives a key for the PRNG.
fn derive_key(
    prev_hash: [u8; INPUT_HASH_SIZE],
    root_hash: [u8; INPUT_HASH_SIZE],
    nonce: [u8; NONCE_SIZE],
    penalty: u32,
    delay: u32,
) -> [u8; BLOCK_HEADER_SIZE] {
    let mut key = Vec::with_capacity(prng::KEY_LEN);

    key.extend_from_slice(&prev_hash);
    key.extend_from_slice(&root_hash);
    key.extend_from_slice(&delay.to_le_bytes());
    key.extend_from_slice(&penalty.to_le_bytes());
    key.extend_from_slice(&nonce);

    key[0..BLOCK_HEADER_SIZE].try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use crate::constants::MAX;

    use super::*;

    // Verify that the output has an expected size.
    #[test]
    fn verify_output_size() {
        // prime 𝑞 = 21600 − 2273;
        // integer max = 𝑞 − 1.

        let prev_hash = [1u8; INPUT_HASH_SIZE];
        let root_hash = [2u8; INPUT_HASH_SIZE];
        let nonce = [3u8; NONCE_SIZE];

        let res = expand(prev_hash, root_hash, nonce, 0, 0, MAX.clone());

        assert!(res <= *MAX);
    }
}
