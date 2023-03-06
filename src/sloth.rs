//! Sloth VDF implementation.
//!
//! Implementation follows the Section 3 in paper "A random zoo" [1].
//! We use a single round of Keccak-f[1600] as a permutation function.
//!
//! [1] https://csrc.nist.gov/csrc/media/events/workshop-on-elliptic-curve-cryptography-standards/documents/papers/session1-wesolowski-paper.pdf

// TODO: implement this module using crypto_bigint once it's more stable.

// use crypto_bigint::{nlimbs, MulMod, UInt};
use std::convert::TryInto;

use crate::constants::{SEED, SEED_EXPONENT, SEED_SIGNED};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

use crate::{inverse::inverse_keccak_function, keccakf::RC, WORDS};

// Define a single-round Keccak-f and its inverse - we use it as permutation function.
inverse_keccak_function!("`inverse-keccak-f[1600, 1]`", inverse_keccakf_1, 1, RC);
keccak_function!("`keccak-f[1600, 1]`", keccakf_1, 1, RC);

/// Defines internal integer type.
/// This should be at least `2 ^ (2*k)`, where `k` is the security level.
type Int = BigUint;

/// Implements the Rho function as seen in Section 3.2 of the paper.
fn rho(x: Int) -> Int {
    let x1 = x.modpow(&SEED_EXPONENT, &SEED);

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
/// which in our case is a single round of Keccak.
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
pub fn solve(s: Int, delay: u32) -> Int {
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
pub fn verify(s: Int, w: Int, delay: u32) -> bool {
    let mut w_iter = w;

    for _ in 0..delay {
        w_iter = tau_inverse(w_iter);
    }

    w_iter == s
}

#[cfg(test)]
mod tests {
    use super::{rho, rho_inverse, sigma, solve, verify};
    use crate::constants::Int;
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

    /// Testing on test vectors taken from the specification.
    #[test]
    fn rho_test_vectors() {
        {
            assert_eq!(rho(Int::from(0u8)), Int::from(0u8));
        }
        {
            let expected = Int::parse_bytes(b"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff71e", 16).unwrap();
            assert_eq!(rho(Int::from(1u8)), expected);
        }
        {
            let x = Int::parse_bytes(b"f12ceb98ae50c59570860a5159dc58d10c8eb4acc3e4dc5c949ab47f4727aeccd9374b3830f563227daa0c340bdd89c70b13055d16bb5a9d1823b30265243ace332fdd45107443ca68542b9ec02207416a590d0e3e829af15227853ada56d2abdcfcb4efd96aba69025aa5f26936330755d9dbc1dc43ca84d77381c0e48c560d757aa4e8245cf10ff428ed2897127c094f45c7ef626ce4e08a9043844b4bfe141419607701ed7b6769a3a4a7f0038263add45c49bcd1771b6bf3db4331ee96c755c70a6550710fa5", 16).unwrap();
            let expected = Int::parse_bytes(b"85bf38f105494aba1032d60c0cd5b14f48686e76582a0f22b60079fe3a380b8e3092dbfdfb994739d6dba5f3ee2f8b0bd923521392758c0a9bbb47ec4f990300e37144b3874997fdc1c7a0a10a0eeb05138fc970b52157a5f7266799766ea9226b2487a4763f482658d6a298299a9716f8e77918cd01ca6a1c0ef72030c4cfbf7a679a2fc96e690683b2d85bd86dbe2129c3f2145dd4284639fab32957452ebdf46bd9d1317bde07c55fbfb9627d161ae4c34c2f6606802c743b4aabb812942a0962135d288aa9e3", 16).unwrap();
            assert_eq!(rho(Int::from(x)), expected);
        }
        {
            let x = Int::parse_bytes(b"4928b0f415b721f6478a60deccb79421686eb572241ceab570eb683624f198c616fdd42298ec0d80b736e511873e4dceb6b2abd0a11e351022e6e43b242aeb024b65064b5ce01b914118ab43f1b2945cc4e9236708a8654171c04a2e0ca0183d0978a69c5020255928c0394e608ddd691198ac1138495d6320b6d62878a65297fdcfe62036ec1d18a3c78b0ca8d4593a701060933efb4b6b05c0764b9f404a7ce883e129697da9bfa65cef86fbe02b3f6943624a79aedc7554f74c41f349974fa464f078d9e53b86", 16).unwrap();
            let expected = Int::parse_bytes(b"197033a7ca4c1dcf00d82830deaa6d040040cd02705634394b24966726e49f45d155048aadadb211b26f5a3b942a6a67f90b4996df59215bf19ddec59a27ad6043814279e67a39e1fe2d99b9fa557278dd3416a7a403057d108864ee33c8f315364798c53e2005f6ea435d1454253469af1017d8cec09defe8b58ff16090061e406f524eb26fd9479b115b5d4f4e2ab40b1550db4e976f01b04b25f273f804936615b032d1d40366901baac498939baed10bc130298145b02631fce6c5aa2b8364b6d8bc6388c1e1", 16).unwrap();
            assert_eq!(rho(Int::from(x)), expected);
        }
    }

    /// Testing on test vectors taken from the specification.
    #[test]
    fn sigma_test_vectors() {
        {
            let expected = Int::from(1u8);
            assert_eq!(sigma(Int::from(0u8)), expected);

            // fixme
            // let expected = Int::parse_bytes(b"0100000000000001004000000000000040000000000000000040000000000001400000000000000000002000000000080000000000000000000000000000000800002000000000000000000000000000000004000000000000000000000000000004040000000000000000000000000000040000000000000000000000000000000000100000000000000000000002000000000000000000000000100000020000000000000000000000000002000100000000000000000000000000020000000000000000000100", 16).unwrap();
            // assert_eq!(sigma(Int::from(1u8)), expected);
        }
    }
}
