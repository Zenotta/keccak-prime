//! Constants used for crypto functions configuration.

use lazy_static::lazy_static;
use num_bigint::{BigInt, BigUint, Sign};

/// Block header size, in bytes.
pub const BLOCK_HEADER_SIZE: usize = 80; // in bytes; 640 bits
/// Hash inputs sizes, in bytes.
pub const INPUT_HASH_SIZE: usize = 32; // 256 bits
/// Input nonce size, in bytes.
pub const NONCE_SIZE: usize = 10; // 80 bits
/// Target hash length, in bytes.
pub const HASH_LENGTH: usize = 32; // 256 bits
/// Size of the result of VDF.
pub const WITNESS_SIZE: usize = 200; // 1600 bits

/// Blocks before changing the random program
pub const PROGPOW_PERIOD: usize = 10;
/// Lanes that work together calculating a hash
pub const PROGPOW_LANES: usize = 16;
/// uint32 registers per lane
pub const PROGPOW_REGS: usize = 32;
/// uint32 loads from the DAG per lane
pub const PROGPOW_DAG_LOADS: usize = 0; // 4
/// size of the cached portion of the DAG
pub const PROGPOW_CACHE_BYTES: usize = 16 * 1024;
/// DAG accesses, also the number of loops executed
pub const PROGPOW_CNT_DAG: usize = 64;
/// random cache accesses per loop
pub const PROGPOW_CNT_CACHE: usize = 0; // 11
/// random math instructions per loop
pub const PROGPOW_CNT_MATH: usize = 18;

/// Arbitrary-precision integer type. Defined for portability.
pub type Int = BigUint;

// crypto_bigint alternative:
// type Int = crypto_bigint::UInt<{ nlimbs!(1600) }>;
// pub const SEED: Int = Int::from_be_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff71f");

lazy_static! {
    /// Seed number for Sloth VDF: p = 2^1600 – 2273
    pub(crate) static ref SEED: Int = Int::parse_bytes(b"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff71f", 16).unwrap();
    pub(crate) static ref SEED_SIGNED: BigInt = BigInt::from_biguint(Sign::Plus, SEED.clone());
    pub(crate) static ref SEED_EXPONENT: Int = (SEED.clone() - 3u32) / 4u32;

    /// Used as the domain for VDF - {0, 1, ... MAX}.
    pub(crate) static ref MAX: Int = SEED.clone() - 1u32;
}
