//! Implements the Keccak-prime function.

use crate::{
    constants::{INPUT_HASH_SIZE, MAX, NONCE_SIZE},
    expansion::expand,
    prf::prf,
    sha3::Sha3,
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
/// - `loop_count`: number of loops in PRF.
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
    hash_bytes.extend_from_slice(&loop_count.to_be_bytes());
    hash_bytes.extend_from_slice(&delay.to_be_bytes());
    hash_bytes.extend_from_slice(&penalty.to_be_bytes());
    hash_bytes.extend_from_slice(&nonce);
    hash_bytes.extend_from_slice(&prev_block_root_hash);

    // Construct a SHA-3 function with rate=1088 and capacity=512.
    let mut h = Sha3::v256();
    h.update(&hash_bytes);

    (witness_bytes, h.finalize_with_penalty(penalty))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak_prime_test() {
        let block_k_root_hash = [
            0xc0, 0x22, 0x42, 0x39, 0xa8, 0xf7, 0x14, 0xbc, 0xfc, 0x85, 0x43, 0x11, 0xc7, 0xe4,
            0x6d, 0x4e, 0x66, 0xdf, 0x62, 0xe4, 0xcb, 0x67, 0x2f, 0x8f, 0x19, 0xfe, 0x51, 0x12,
            0x31, 0xcd, 0x72, 0x48,
        ];
        let prev_block_root_hash = [
            0xcd, 0xae, 0x4e, 0x1b, 0x5f, 0x9f, 0x83, 0x23, 0x77, 0xcf, 0xa1, 0xf4, 0xe8, 0xe0,
            0x82, 0x49, 0x54, 0x7f, 0x77, 0xc0, 0x6b, 0xd5, 0x28, 0xac, 0x32, 0x4b, 0xb8, 0x48,
            0xf8, 0xbd, 0x12, 0x64,
        ];
        let prev_hash = [
            0xc6, 0xa9, 0x02, 0x52, 0xf8, 0xae, 0x2b, 0xdf, 0x79, 0xa4, 0x88, 0xa3, 0xbc, 0x6d,
            0xa5, 0x41, 0x9f, 0xb1, 0xe7, 0x75, 0x8a, 0xd3, 0x5d, 0xf2, 0x7f, 0xf6, 0xf2, 0x4d,
            0x2e, 0xa5, 0xde, 0xbe,
        ];
        let root_hash = [
            0xf3, 0x82, 0x1a, 0xb7, 0x0b, 0x4a, 0x0a, 0x5d, 0x6c, 0x6c, 0x9c, 0x10, 0x6c, 0x72,
            0x73, 0xd3, 0x9f, 0xb1, 0xad, 0xd7, 0xd9, 0x09, 0xb8, 0xbf, 0xc7, 0x2d, 0xbb, 0xa1,
            0x7b, 0xcd, 0x5a, 0xf4,
        ];
        let nonce = [0x38, 0xcc, 0x40, 0x09, 0xde, 0x41, 0x33, 0x6e, 0x00, 0x00];

        let expected = [
            0x77, 0x2e, 0xc6, 0x7a, 0x27, 0x26, 0xe3, 0x28, 0x68, 0xf0, 0x0a, 0xd3, 0xd1, 0xc8,
            0x85, 0x08, 0xf4, 0x4c, 0x5c, 0xe8, 0x4d, 0x84, 0x94, 0xf9, 0xbd, 0xc2, 0x6b, 0xb6,
            0x4f, 0xd4, 0x41, 0x93,
        ];

        let penalty = 2;
        let loop_count = 3;
        let delay = 4;

        let (witness, output_hash) = link_blocks(
            block_k_root_hash,
            prev_block_root_hash,
            prev_hash,
            root_hash,
            nonce,
            penalty,
            delay,
            loop_count,
        );

        println!(
            "output_hash: {}\n witness: {}",
            hex::encode(&output_hash),
            hex::encode(&witness)
        );

        assert_eq!(output_hash, expected);
    }
}
