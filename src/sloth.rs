//! Sloth VDF implementation.
//!
//! Implementation follows the Section 3 in paper "A random zoo" [1].
//! We use a single round of Keccak-f[1600] as a permutation function.
//!
//! [1] https://csrc.nist.gov/csrc/media/events/workshop-on-elliptic-curve-cryptography-standards/documents/papers/session1-wesolowski-paper.pdf

// TODO: implement this module using crypto_bigint once it's more stable.

// use crypto_bigint::{nlimbs, MulMod, UInt};
use std::convert::TryInto;

use lazy_static::lazy_static;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

use crate::{inverse::inverse_keccak_function, keccakf::RC, WORDS};

// Define a single-round Keccak-f and its inverse - we use it as permutation function.
inverse_keccak_function!("`inverse-keccak-f[1600, 1]`", inverse_keccakf_1, 1, RC);
keccak_function!("`keccak-f[1600, 1]`", keccakf_1, 1, RC);

/// Defines internal integer type.
/// This should be at least `2 ^ (2*k)`, where `k` is the security level.
type Int = BigUint;

// crypto_bigint alternative:
// type Int = crypto_bigint::UInt<{ nlimbs!(1600) }>;
// pub const SEED: Int = Int::from_be_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff71f");

lazy_static! {
    /// Seed number for Sloth VDF: p = 2^1600 – 2273
    static ref SEED: Int = Int::parse_bytes(b"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff71f", 16).unwrap();
    static ref SEED_SIGNED: BigInt = BigInt::from_biguint(Sign::Plus, SEED.clone());
    static ref SEED_EXPONENT: Int = (SEED.clone() - 3u32) / 4u32;
}

/// Implements the Rho function as seen in Section 3.2 of the paper.
fn rho(x: Int) -> Int {
    let x1 = x.modpow(&*SEED_EXPONENT, &*SEED);

    let x2 = (&x * &x1) % &*SEED; // = x ^ ((p + 1) / 4)
    let is_even = &x2 % Int::from(2u8) == Int::zero();

    let x3 = (&x1 * &x2) % &*SEED; // = x ^ ((p - 1) / 2)

    // Check for quadratic residue
    let quad_res = x3 <= Int::one();

    if is_even == quad_res {
        x2
    } else {
        (&*SEED - &x2) % &*SEED
    }
}

/// Inverse Rho function.
fn rho_inverse(x: Int) -> Int {
    let x = BigInt::from(x);
    let is_even = &x % 2 == BigInt::zero();

    let multiplier = BigInt::from(if is_even { 1 } else { -1 });

    ((&multiplier * &x * &x % &*SEED_SIGNED + &*SEED_SIGNED) % &*SEED_SIGNED)
        .to_biguint()
        // unwrap is fine here because `mod SEED_SIGNED` guarantees it's a positive number
        .unwrap()
}

/// Permutation function. It is implemented using a single round of Keccak.
fn sigma(x: Int) -> Int {
    let mut bytes = (x % &*SEED).to_u64_digits();

    // Ensure the `bytes` length is exactly `keccak_prime::WORDS` bytes.
    bytes.resize(WORDS, 0);

    // Convert Int into a bit string suitable for Keccak-f which is `[u64; WORDS]`.
    let mut byte_array = bytes
        .try_into()
        .expect("unexpected incorrect input for keccak-f");

    // Apply a single round of Keccak-f.
    keccakf_1(&mut byte_array);

    // Convert a bit string into an integer.
    // num_bigint expects the input to be `Vec<u32>`.
    // FIXME: get rid of unsafety?
    Int::new(unsafe {
        let (_prefix, digits_u32, _suffix) = byte_array.align_to::<u32>();
        digits_u32.to_owned()
    })
}

/// Inverse of a permutation function. In our case, it's simply a
/// single round of inverse Keccak-f[1600].
fn sigma_inverse(x: Int) -> Int {
    let mut bytes = x.to_u64_digits();

    // Ensure the `bytes` length is exactly `keccak_prime::WORDS` bytes.
    bytes.resize(WORDS, 0);

    // Convert Int into a bit string suitable for Keccak-f which is `[u64; WORDS]`.
    let mut byte_array = bytes
        .try_into()
        .expect("unexpected incorrect input for keccak-f");

    // Apply inverse Keccak-f.
    inverse_keccakf_1(&mut byte_array);

    // Convert a bit string into an integer.
    // num_bigint expects the input to be `Vec<u32>`.
    // FIXME: get rid of unsafety?
    Int::new(unsafe {
        let (_prefix, digits_u32, _suffix) = byte_array.align_to::<u32>();
        digits_u32.to_owned()
    })
}

/// Implements the Tau function as seen in Section 3.2 of the paper.
/// It composes the Rho function with the permutation function Sigma -
/// which in our case is a single round of AES-128.
fn tau(x: Int) -> Int {
    sigma(rho(x))
}

/// Implements the inverse Tau function.
fn tau_inverse(x: Int) -> Int {
    rho_inverse(sigma_inverse(x))
}

/// ## Arguments
/// - `s` is the security parameter.
/// - `delay` is the desired puzzle difficulty.
///
/// ## Returns
/// - Witness number.
pub fn solve(s: Int, delay: u64) -> Int {
    let mut w_iter = s;

    for _ in 0..delay {
        w_iter = tau(w_iter);
    }

    w_iter
}

/// ## Arguments
/// - `s` is the security parameter.
/// - `w` is the witness number obtained from `solve`.
/// - `delay` is the puzzle difficulty.
///
/// ## Returns
/// - `true` if the verification has passed.
pub fn verify(s: Int, w: Int, delay: u64) -> bool {
    let mut w_iter = w;

    for _ in 0..delay {
        w_iter = tau_inverse(w_iter);
    }

    w_iter == s
}

#[cfg(test)]
mod tests {
    use super::{solve, verify};
    use num_bigint::BigUint;
    use std::time::Instant;

    #[test]
    fn sloth() {
        let x = BigUint::from(11u64);
        let t = 100;

        // compute the sloth vdf
        let instant = Instant::now();
        let witness = solve(x.clone(), t);
        println!("{}, eval: {} ms", witness, instant.elapsed().as_millis());

        // verify the result
        let instant = Instant::now();
        assert!(verify(x, witness.clone(), t));
        println!("verified in {} ms", instant.elapsed().as_millis());
    }
}
