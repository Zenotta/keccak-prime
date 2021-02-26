use crate::{bits_to_rate, keccakf::KeccakF, Hasher, KeccakState};

/// The `SHA3` hash functions defined in [`FIPS-202`].
///
/// [`FIPS-202`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
///
/// # Usage
///
/// ```toml
/// [dependencies]
/// tiny-keccak = { version = "2.0.0", features = ["sha3"] }
/// ```
///
/// # Example
///
/// ```
/// # use keccak_prime::{Hasher, Sha3};
/// #
/// # fn main() {
/// #   let input = b"hello world";
/// #   let mut output = [0; 32];
/// #   let mut sha3 = Sha3::v256();
/// #   sha3.update(input);
/// #   sha3.finalize(&mut output);
/// # }
/// ```
#[derive(Clone)]
pub struct Sha3 {
    state: KeccakState<KeccakF>,
}

impl Sha3 {
    const DELIM: u8 = 0x06;

    /// Creates  new [`Sha3`] hasher with a security level of 224 bits.
    ///
    /// [`Sha3`]: struct.Sha3.html
    pub fn v224() -> Sha3 {
        Sha3::new(224)
    }

    /// Creates  new [`Sha3`] hasher with a security level of 256 bits.
    ///
    /// [`Sha3`]: struct.Sha3.html
    pub fn v256() -> Sha3 {
        Sha3::new(256)
    }

    /// Creates  new [`Sha3`] hasher with a security level of 384 bits.
    ///
    /// [`Sha3`]: struct.Sha3.html
    pub fn v384() -> Sha3 {
        Sha3::new(384)
    }

    /// Creates  new [`Sha3`] hasher with a security level of 512 bits.
    ///
    /// [`Sha3`]: struct.Sha3.html
    pub fn v512() -> Sha3 {
        Sha3::new(512)
    }

    fn new(bits: usize) -> Sha3 {
        Sha3 {
            state: KeccakState::new(bits_to_rate(bits), Self::DELIM),
        }
    }
}

impl Hasher for Sha3 {
    fn update(&mut self, input: &[u8]) {
        self.state.update(input);
    }

    fn finalize(self, output: &mut [u8]) {
        self.state.finalize(output);
    }
}
