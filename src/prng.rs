//! This is an implementation of a pseudorandom generation function based on SHAKE-128.

use std::convert::TryInto;

use crate::keccakf::KeccakF;
use crate::{bits_to_rate, Buffer, KeccakState, Mode, WORDS};

/// Key length in bytes.
pub(crate) const KEY_LEN: usize = 166; // 1328 bits

/// Based on `SHAKE` extendable-output functions defined in [`FIPS-202`].
///
/// [`FIPS-202`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
#[derive(Clone, Debug)]
pub struct PrngState {
    state: KeccakState<KeccakF>,
}

impl PrngState {
    const DELIM: u8 = 0x1f;

    pub(crate) fn print_internal_state(&self) {
        dbg!(self.state.print_internal_state(), self.state.offset_bits);
    }

    /// Creates a new instance of the SHAKE-based CSPRNG from a provided `key` and a `usage` number.
    pub fn new(key: &[u8], usage: u8) -> PrngState {
        // initial state = key || usage || four one bits || padding 10*1 || 256 zero bits
        assert!(key.len() <= KEY_LEN); // FIXME panics

        // Fill the buffer with initial values
        let mut initial_bits = Vec::with_capacity(KEY_LEN + 1);
        initial_bits.extend_from_slice(key);
        initial_bits.push(usage);

        // padding
        let pad_bytes = KEY_LEN - key.len();
        initial_bits.extend(if pad_bytes > 0 {
            let mut padding = vec![0; pad_bytes + 1];
            padding[0] = 0b00011111;
            padding[pad_bytes] = 0b10000000;
            padding.into_iter()
        } else {
            // four 1 bits + pad10*1
            vec![0b10011111].into_iter()
        });
        initial_bits.extend_from_slice(&[0; 32]); // 256 zero bits

        let initial_offset = 1344;
        let initial_bits_array: [u8; WORDS * 8] = initial_bits.clone().try_into().unwrap();

        let mut state = KeccakState::new_with_buffer(
            bits_to_rate(128),
            Self::DELIM,
            Buffer::from(initial_bits_array),
            initial_offset,
        );

        // dbg!(state.print_internal_state());
        state.fill_block();
        state.mode = Mode::Squeezing;

        PrngState { state }
    }

    /// Generates a pseudorandom bit string of length `len` in bytes.
    pub fn get_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut result = vec![0; len];
        self.state.squeeze(&mut result, 0);
        result
    }

    /// Generates a pseudorandom bit string of length `len` in bits.
    pub(crate) fn get_bits(&mut self, len: usize) -> Vec<u8> {
        let mut result = vec![0; ((len + 7) & !7) / 8];
        self.state.squeeze_bits(len, &mut result);
        result
    }

    /// Generates a random 16-bit integer.
    pub fn get_integer(&mut self, max: u16) -> u16 {
        if max == 0 {
            0
        } else {
            // fixme: is there a more efficient way of doing this? ilog2 rounds number down
            let len = (max as f32 + 1f32).log2().ceil() as usize;
            loop {
                let mut x = self.get_bits(len);

                if x.len() < 2 {
                    // fixme: pad for u16 should be implemented more efficiently
                    x.insert(0, 0);
                }
                let num = u16::from_be_bytes(x.try_into().unwrap());
                if num <= max {
                    return num;
                }
            }
        }
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

    #[test]
    fn prng_test_vectors() {
        let key = hex::decode("164f659995f4ec98377c8f7b16e22be5682d115624ea429a4ed422325c151f82b3af308743907f92b8a2b25ceca5ae5b762d54ffe6b84dabfc985d6db451d44717819bdcb563a30c79e29e12115413b1f395af84310ddbe3d4c110fb1566286c8471b2cbfc6af6e7944a264308b6ca658cd2256539ae1edb0a00fe213d8e939b2f699899768e4095812ab5e463588910ecc264a20e81d21f63e3932baad311a544d1e39b7997").unwrap();
        let mut prng = PrngState::new(&key, 0xf6);

        {
            let res = prng.get_bytes(1336 / 8);
            let expected_res = hex::decode("407a8e20ac21746f7acd3762c1d39e3e87d33ec1e3f2252745c4ee02935d2f865da019bb512bfbf43160a25f0aa8898e9609e9611d8d423205fcfe74c904c6c0426956f90b7641c6fb856bfb1cb31f4fbb8dbb9c95a6f1851d55e8c48b86e96048cccc623b3e90c88dd2d08fe3dca02e273431ba66096104c59d6d73465465307aa03b6a9c5a84edd94ad2a67cec97a5a5b32a0af718cf34bfce7fc56dfacfd764065722458abb").unwrap();
            assert_eq!(res, expected_res);
        }
        {
            let res = prng.get_bits(17);
            let expected_res = hex::decode("0115c7").unwrap();
            assert_eq!(res, expected_res);
        }
        {
            let res = prng.get_bits(23);
            let expected_res = hex::decode("758758").unwrap();
            assert_eq!(res, expected_res);
        }
        {
            let res = prng.get_integer(0);
            assert_eq!(res, 0);
        }
        {
            let res = prng.get_integer(0x82);
            assert_eq!(res, 6);
        }
        {
            let res = prng.get_integer(0x05);
            assert_eq!(res, 4);
        }
    }
}
