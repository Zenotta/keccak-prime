//! This is an implementation of a pseudorandom generation function based on SHAKE-128.

use crate::keccakf::KeccakF;
use crate::{bits_to_rate, KeccakState};

/// Key length in bytes.
pub(crate) const KEY_LEN: usize = 166; // 1328 bits

/// Based on `SHAKE` extendable-output functions defined in [`FIPS-202`].
///
/// [`FIPS-202`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
#[derive(Clone)]
pub struct PrngState {
    state: KeccakState<KeccakF>,
}

impl PrngState {
    const DELIM: u8 = 0x1f;

    /// Creates a new instance of the SHAKE-based CSPRNG from a provided `key` and a `usage` number.
    pub fn new(key: &[u8], usage: u8) -> PrngState {
        // initial state = key || usage || four one bits || padding 10*1 || 256 zero bits
        assert!(key.len() <= KEY_LEN); // FIXME panics

        // Fill the buffer with initial values
        let mut initial_bits = Vec::with_capacity(KEY_LEN + 1);
        initial_bits.extend_from_slice(key);
        initial_bits.push(usage);
        // bits_remainder.push(0b1111); // four 1 bits + pad10*1
        // bits_remainder.extend_from_slice(&u32::default().to_ne_bytes()); // 256 zero bits

        let mut state = KeccakState::new(bits_to_rate(128), Self::DELIM);
        state.update(&initial_bits);

        PrngState { state }
    }

    /// Generates a pseudorandom bit string of length `len`.
    pub fn get_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(len);
        result.resize(len, 0);

        self.state.squeeze(&mut result);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the `get_bytes` function works with varying lengths.
    #[test]
    fn variable_lengths() {
        // k = PrevHash || RootHash || d || p || Nonce
        let mut rng = PrngState::new(&[0; KEY_LEN], 1);

        // len == 1344 / 8

        assert_eq!(rng.get_bytes(1).len(), 1);
        assert_eq!(rng.get_bytes(4).len(), 4);
        assert_eq!(rng.get_bytes(128).len(), 128);
        assert_eq!(rng.get_bytes(1000).len(), 1000);
        assert_eq!(rng.get_bytes(4096).len(), 4096);
        assert_eq!(rng.get_bytes(2).len(), 2);
    }

    #[test]
    fn diff_keys() {
        {
            // Test PRNG initialised with different usage numbers and same keys.
            let mut rng1 = PrngState::new(&[0; KEY_LEN], 1);
            let mut rng2 = PrngState::new(&[0; KEY_LEN], 2);

            assert_ne!(rng1.get_bytes(64), rng2.get_bytes(64));
        }
        {
            // Test PRNG initialised with the same usage and same key.
            let mut rng1 = PrngState::new(&[0; KEY_LEN], 1);
            let mut rng2 = PrngState::new(&[0; KEY_LEN], 1);

            assert_eq!(rng1.get_bytes(64), rng2.get_bytes(64));
        }
    }

    /// Test counter.
    #[test]
    fn counter() {
        let mut rng = PrngState::new(&[0; KEY_LEN], 1);

        let byte1 = rng.get_bytes(1);
        let byte2 = rng.get_bytes(1);

        assert_ne!(byte1, byte2);
    }
}
