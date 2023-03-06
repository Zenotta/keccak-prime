//! The `Keccak` hash functions.

use super::{bits_to_rate, keccakf::KeccakF, Hasher, KeccakState};

/// The `Keccak` hash functions defined in [`Keccak SHA3 submission`].
///
/// # Usage
///
/// ```toml
/// [dependencies]
/// tiny-keccak = { version = "2.0.0", features = ["keccak"] }
/// ```
///
/// [`Keccak SHA3 submission`]: https://keccak.team/files/Keccak-submission-3.pdf
#[derive(Clone)]
pub struct Keccak {
    state: KeccakState<KeccakF>,
}

impl Keccak {
    const DELIM: u8 = 0x01;

    /// Creates  new [`Keccak`] hasher with a security level of 224 bits.
    ///
    /// [`Keccak`]: struct.Keccak.html
    pub fn v224() -> Keccak {
        Keccak::new(224)
    }

    /// Creates  new [`Keccak`] hasher with a security level of 256 bits.
    ///
    /// [`Keccak`]: struct.Keccak.html
    pub fn v256() -> Keccak {
        Keccak::new(256)
    }

    /// Creates  new [`Keccak`] hasher with a security level of 384 bits.
    ///
    /// [`Keccak`]: struct.Keccak.html
    pub fn v384() -> Keccak {
        Keccak::new(384)
    }

    /// Creates  new [`Keccak`] hasher with a security level of 512 bits.
    ///
    /// [`Keccak`]: struct.Keccak.html
    pub fn v512() -> Keccak {
        Keccak::new(512)
    }

    /// Creates new [`Keccak`] hasher with a specified rate.
    pub fn new(bits: usize) -> Keccak {
        // dbg!(bits, bits_to_rate(bits));
        Keccak {
            state: KeccakState::new(bits_to_rate(bits), Self::DELIM),
        }
    }
}

impl Keccak {
    /// Squeeze the state to the output (256 bits) and apply an extra number of permutations.
    pub fn finalize_with_penalty(self, penalty: u32) -> [u8; 32] {
        self.state.finalize_with_penalty(penalty)
    }
}

impl Hasher for Keccak {
    /// Absorb additional input. Can be called multiple times.
    ///
    /// # Example
    ///
    /// ```
    /// # use keccak_prime::{Hasher, Keccak};
    ///
    /// # fn main() {
    /// #   let mut keccak = Keccak::v256();
    /// #   keccak.update(b"hello");
    /// #   keccak.update(b" world");
    /// # }
    /// ```
    fn update(&mut self, input: &[u8]) {
        self.state.update(input);
    }

    /// Pad and squeeze the state to the output.
    ///
    /// # Example
    ///
    /// ```
    /// # use keccak_prime::{Hasher, Keccak};
    /// #
    /// # fn main() {
    /// #   let keccak = Keccak::v256();
    /// #   let mut output = [0u8; 32];
    /// #   keccak.finalize(&mut output);
    /// # }
    /// #
    /// ```
    fn finalize(self, output: &mut [u8]) {
        self.state.finalize(output);
    }
}

#[cfg(test)]
mod tests {
    use super::{Hasher, Keccak};

    // Keccak without any extra penalty must be equal to the original Keccak.
    #[test]
    fn unmodified_keccak_equivalence() {
        let mut output_orig = [0; 32];

        let mut keccak_orig = Keccak::v256();
        keccak_orig.update(&[1, 2, 3]);
        keccak_orig.finalize(&mut output_orig);

        let mut keccak_mod = Keccak::v256();
        keccak_mod.update(&[1, 2, 3]);
        let output_mod = keccak_mod.finalize_with_penalty(0);

        assert_eq!(output_orig, output_mod);

        // Check that penalty > 0 differs from the orig keccak.
        let mut keccak_mod = Keccak::v256();
        keccak_mod.update(&[1, 2, 3]);
        let output_penalized = keccak_mod.finalize_with_penalty(1);

        assert_ne!(output_orig, output_penalized);
    }
}
