//! Sloth VDF implementation.
//!
//! Implementation follows the Section 3 in paper "A random zoo" [1].
//!
//! [1] https://csrc.nist.gov/csrc/media/events/workshop-on-elliptic-curve-cryptography-standards/documents/papers/session1-wesolowski-paper.pdf

use aes::cipher::{generic_array::GenericArray, BlockEncrypt, KeyInit};
use aes::Aes128;
use num_bigint::BigUint;

/// Defines internal integer type.
/// This should be at least `2 ^ (2*k)`, where `k` is the security level.
type Int = BigUint;

/// Seed number for Sloth VDF.
pub const SEED: u128 = 31;

// Implements the Rho function as seen in Section 3.2 of the paper.
fn rho(x: Int) -> Int {
    let x1 = x.modpow(&Int::from((SEED - 3) / 4), &Int::from(SEED));

    let x2 = (x * x1.clone()) % SEED; // = x ^ ((p + 1) / 4)
    let is_even = x2.clone() % Int::from(2u8) == Int::from(0u8);

    let x3 = (x1 * x2.clone()) % SEED; // = x ^ ((p - 1) / 2)

    // Check for quadratic residue
    let quad_res = x3 <= Int::from(1u8);

    if is_even == quad_res {
        x2
    } else {
        (SEED - x2) % SEED
    }
}

// Implements the Tau function as seen in Section 3.2 of the paper.
// It composes the Rho function with the permutation function Sigma -
// which in our case is a single round of AES-128.
fn tau(x: Int) -> Int {
    let x = rho(x);

    let key = GenericArray::from([0u8; 16]);
    let cipher = Aes128::new(&key);

    let mut bytes = x.to_bytes_be();

    // pad bytes to 16 bytes boundary
    let rem_bytes = bytes.len() % 16;

    if rem_bytes > 0 {
        bytes.extend(std::iter::repeat(0).take(16 - rem_bytes));
    }

    for ref mut chunk in bytes.chunks_mut(16) {
        let block = GenericArray::from_mut_slice(chunk);
        cipher.encrypt_block(block);
    }

    Int::from_bytes_be(&bytes)
}

/// ## Arguments
/// - `w` is the security parameter.
/// - `delay` is the desired puzzle difficulty.
///
/// ## Returns
/// - Witness number.
pub fn solve(w: Int, delay: u64) -> Int {
    let mut w_iter = w;

    for _ in 0..delay {
        w_iter = tau(w_iter);
    }

    w_iter
}

#[cfg(test)]
mod tests {
    use super::{rho, solve};
    use num_bigint::BigUint;
    use std::time::Instant;

    #[test]
    fn test_rho() {
        // Check output from rho - it should correspond to expected values
        // (calculated from the formulas found in the paper with seed num = 31).
        let fixtures = [
            (0u8, 0u64),
            (1, 30),
            (2, 8),
            (3, 11),
            (11, 19),
            (15, 27),
            (30, 1),
        ];

        for (iter, expected_val) in fixtures {
            assert_eq!(rho(BigUint::from(iter)), BigUint::from(expected_val));
        }
    }

    #[test]
    fn solve_test() {
        let x = 80u64;
        let t = 9999;

        let instant = Instant::now();
        let y = solve(BigUint::from(x), t);

        println!("{}, eval: {} ms", y, instant.elapsed().as_millis());
    }
}
