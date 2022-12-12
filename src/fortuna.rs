//! This is a simplified implementation of the Fortuna CSPRNG.
//! The main difference from the original Fortuna is that we don't use hashes for seeding;
//! the hash is computed externally. Instead, we generate a key before the generation of
//! pseudorandom data.

use aes_gcm_siv::aead::{generic_array::GenericArray, AeadInPlace, NewAead};
use aes_gcm_siv::Aes256GcmSiv;

use crate::prime::KeccakPrimeError;

/// Initialization vector size used in the AES-GCM implementation.
pub const AES_IV_SIZE: usize = 12; // 96 bits

/// Length in bytes of the AES key.
const KEY_LEN: usize = 32;

/// The usage number is limited to 96 bits.
const USAGE_MAX_BITS: u128 = 96;

/// Simplified Fortuna CSPRNG
pub struct Fortuna {
    /// Seeded key.
    key: Aes256GcmSiv,
    /// Counter value.
    cb: u128,
    /// Remained of bits that weren't used in the latest generated bit string.
    bits_remainder: Vec<u8>,
}

impl Fortuna {
    /// Creates a new instance of the Fortuna CSPRNG from a provided `key` and a `usage` number.
    pub fn new(key: &[u8; KEY_LEN], usage: u128) -> Result<Fortuna, KeccakPrimeError> {
        let key = Self::gen_seed_key(key, usage)?;
        Ok(Fortuna {
            key,
            cb: 0,
            bits_remainder: Vec::with_capacity(128),
        })
    }

    /// Generates a pseudorandom bit string of length `len`.
    pub fn get_bytes(&mut self, mut len: usize) -> Result<Vec<u8>, KeccakPrimeError> {
        let mut result = Vec::with_capacity(len);

        if !self.bits_remainder.is_empty() {
            // Get min(len, bits_remainder.len()) bits stored as the remainder.
            let range = std::cmp::min(len, self.bits_remainder.len());
            let remainder: Vec<_> = self.bits_remainder.drain(0..range).collect();
            len -= remainder.len();
            result.extend(remainder);
        }

        while len >= 16 {
            result.extend(self.gen_block()?);
            len -= 16;
        }

        if len > 0 {
            let block = self.gen_block()?;
            result.extend(&block[0..len]);

            // Store unused bits as a remainder.
            if block.len() > len {
                self.bits_remainder.extend(&block[len..]);
            }
        }

        Ok(result)
    }

    /// Generates a next block of bits from the current counter value and increments the counter.
    fn gen_block(&mut self) -> Result<[u8; 16], KeccakPrimeError> {
        let mut cb = u128::to_be_bytes(self.cb);

        let _auth_tag = self.key.encrypt_in_place_detached(
            // We use a zero nonce as an initialization vector.
            GenericArray::from_slice(&[0; AES_IV_SIZE]),
            &[0u8; 0], // we don't have any additional data
            &mut cb,
        )?;

        self.cb = self.cb.wrapping_add(1);

        Ok(cb)
    }

    /// Generates a seed key from the provided values.
    fn gen_seed_key(key: &[u8; KEY_LEN], usage: u128) -> Result<Aes256GcmSiv, KeccakPrimeError> {
        let key = GenericArray::from_slice(key);
        let cipher = Aes256GcmSiv::new(key);

        let usage = usage & ((1u128 << USAGE_MAX_BITS) - 1); // limit the usage number to 96 bits
        let cb = u128::pow(2, 32) * usage;

        // Convert 'usage' into its binary representation.
        // This value will be used as one half of the initial key.
        let mut cb1 = u128::to_be_bytes(cb);

        // Also use the increment function to obtain the 2nd half of the key.
        let mut cb2 = u128::to_be_bytes(cb.wrapping_add(1));

        // 'encrypt_detached' means we _don't_ concatenate the authentication tag with the cipher output
        // because we want the cipher to be of a particular size (128 bits) to be used as a key.
        let _auth_tag = cipher.encrypt_in_place_detached(
            // We use a zero nonce as an initialization vector.
            GenericArray::from_slice(&[0; AES_IV_SIZE]),
            &[0u8; 0], // we don't have any additional data
            &mut cb1,
        )?;
        let _auth_tag = cipher.encrypt_in_place_detached(
            GenericArray::from_slice(&[0; AES_IV_SIZE]),
            &[0u8; 0],
            &mut cb2,
        )?;

        // Concatenate encrypted values to get the resulting key.
        let seed_key = GenericArray::clone_from_slice(&[cb1, cb2].concat());
        let seed_cipher = Aes256GcmSiv::new(&seed_key);

        Ok(seed_cipher)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the `get_bytes` function works with varying lengths.
    #[test]
    fn variable_lengths() {
        let mut fortuna = Fortuna::new(&[0; 32], 1).unwrap();

        assert_eq!(fortuna.get_bytes(1).unwrap().len(), 1);
        assert_eq!(fortuna.get_bytes(4).unwrap().len(), 4);
        assert_eq!(fortuna.get_bytes(128).unwrap().len(), 128);
        assert_eq!(fortuna.get_bytes(1000).unwrap().len(), 1000);
        assert_eq!(fortuna.get_bytes(4096).unwrap().len(), 4096);
        assert_eq!(fortuna.get_bytes(2).unwrap().len(), 2);
    }

    /// Test Fortuna with different keys.
    #[test]
    fn diff_keys() {
        let mut fortuna1 = Fortuna::new(&[0; 32], 1).unwrap();
        let mut fortuna2 = Fortuna::new(&[1; 32], 1).unwrap();

        assert_ne!(
            fortuna1.get_bytes(64).unwrap(),
            fortuna2.get_bytes(64).unwrap()
        );
    }

    /// Test counter.
    #[test]
    fn counter() {
        let mut fortuna = Fortuna::new(&[0; 32], 1).unwrap();

        let byte1 = fortuna.get_bytes(1).unwrap();
        let byte2 = fortuna.get_bytes(1).unwrap();

        assert_ne!(byte1, byte2);
    }
}
