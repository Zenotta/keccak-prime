//! Implements the Keccak-prime function.

use crate::{
    constants::{INPUT_HASH_SIZE, MAX, NONCE_SIZE},
    expansion::expand,
    keccak::Keccak,
    prf::prf,
    sloth, Hasher,
};
use std::convert::TryInto;

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
    block_k_root_hash: [u8; INPUT_HASH_SIZE],
    prev_block_root_hash: [u8; INPUT_HASH_SIZE],
    prev_hash: [u8; INPUT_HASH_SIZE],
    root_hash: [u8; INPUT_HASH_SIZE],
    nonce: [u8; NONCE_SIZE],
    penalty: u16,
    delay: u16,
    loop_count: u16,
) -> (Vec<u8>, [u8; INPUT_HASH_SIZE]) {
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
        &block_k_root_hash,
        &witness_bytes[0..200].try_into().unwrap(), // FIXME
        loop_count,
    );

    // Hash the results
    let mut hash_bytes = Vec::with_capacity(y.len() + INPUT_HASH_SIZE * 3 + NONCE_SIZE);
    hash_bytes.extend_from_slice(&y);
    hash_bytes.extend_from_slice(&prev_hash);
    hash_bytes.extend_from_slice(&root_hash);
    hash_bytes.extend_from_slice(&delay.to_be_bytes());
    hash_bytes.extend_from_slice(&loop_count.to_be_bytes());
    hash_bytes.extend_from_slice(&penalty.to_be_bytes());
    hash_bytes.extend_from_slice(&nonce);
    hash_bytes.extend_from_slice(&prev_block_root_hash);

    // Construct a Keccak function with rate=1088 and capacity=512.
    let mut keccak = Keccak::new(1088 / 8);
    keccak.update(&hash_bytes);

    (witness_bytes, keccak.finalize_with_penalty(penalty))
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

        let block_k_root_hash = [4; INPUT_HASH_SIZE];
        let prev_hash_of_prev_block = [5; INPUT_HASH_SIZE];

        dbg!(link_blocks(
            block_k_root_hash,
            prev_hash_of_prev_block,
            prev_hash,
            root_hash,
            nonce,
            100,
            100,
            100
        ));
    }
}
