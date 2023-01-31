//! KISS99 pseudorandom generator function.
//! https://en.wikipedia.org/wiki/KISS_(algorithm)
//! http://www.cse.yorku.ca/~oz/marsaglia-rng.html

#[derive(Default, Clone, Copy, Debug)]
pub struct Kiss99State {
    pub z: u32,
    pub w: u32,
    pub jsr: u32,
    pub jcong: u32,
}

pub fn kiss99(state: &mut Kiss99State) -> u32 {
    state.z = (state.z & 65535)
        .wrapping_mul(36969)
        .wrapping_add(state.z >> 16);

    state.w = (state.w & 65535)
        .wrapping_mul(18000)
        .wrapping_add(state.w >> 16);

    let mwc: u32 = (state.z << 16).wrapping_add(state.w);
    state.jsr ^= state.jsr << 17;
    state.jsr ^= state.jsr >> 13;
    state.jsr ^= state.jsr << 5;

    state.jcong = state.jcong.wrapping_mul(69069).wrapping_add(1234567);

    (mwc ^ state.jcong).wrapping_add(state.jsr)
}

// fnv1a implementation
pub const FNV_PRIME: u32 = 0x1000193;
pub const FNV_OFFSET_BASIS: u32 = 0x811c9dc5;

pub const fn fnv1a(h: u32, d: u32) -> u32 {
    return (h ^ d).wrapping_mul(FNV_PRIME);
}

#[cfg(test)]
mod test {
    use super::*;

    // Using test vectors from EIP-1057
    // https://eips.ethereum.org/assets/eip-1057/test-vectors#fnv1a
    #[test]
    fn test_fnv1a() {
        assert_eq!(fnv1a(0x811C9DC5, 0xDDD0A47B), 0xD37EE61A);
        assert_eq!(fnv1a(0xD37EE61A, 0xEE304846), 0xDEDC7AD4);
        assert_eq!(fnv1a(0xDEDC7AD4, 0x00000000), 0xA9155BBC);
    }

    #[test]
    fn test_kiss99() {
        let mut state = Kiss99State {
            z: 362436069,
            w: 521288629,
            jsr: 123456789,
            jcong: 380116160,
        };

        // Expected return values for each iteration starting from 0
        // https://eips.ethereum.org/assets/eip-1057/test-vectors#kiss99
        let expected_values = [769445856, 742012328, 2121196314, 2805620942];

        for i in 0..4 {
            assert_eq!(kiss99(&mut state), expected_values[i]);
        }

        for _ in 5..100_000 {
            // do 99 995 dummy iterations
            kiss99(&mut state);
        }

        // the 100'000th iteration has an expected value
        assert_eq!(kiss99(&mut state), 941074834);
    }
}
