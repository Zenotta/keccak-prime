//! Inverse Keccak functions.

pub(crate) const INVERSE_THETA_POS64: [u64; 5] = [
    0xDE26BC4D789AF134,
    0x09AF135E26BC4D78,
    0xEBC4D789AF135E26,
    0x7135E26BC4D789AF,
    0xCD789AF135E26BC4,
];

macro_rules! inverse_keccak_function {
    ($doc: expr, $name: ident, $rounds: expr, $rc: expr) => {
        #[doc = $doc]
        #[allow(unused_assignments)]
        #[allow(non_upper_case_globals)]
        pub fn $name(a: &mut [u64; $crate::WORDS]) {
            use crunchy::unroll;

            const fn idx(x: usize, y: usize) -> usize {
                return (x % 5 + (5 * (y % 5)));
            }

            for round_index in 0..$rounds {
                let mut array: [u64; 5] = [0; 5];

                let i = $rounds - round_index - 1; // inverse round number

                // iota (xor is inversable, so it's equivalent to iotaInverse)
                a[0] ^= $rc[i];

                // Inverse chi
                unroll! {
                    for y in 0..5 {
                        unroll! {
                            for x in 0..5 {
                                array[x] = a[idx(x, y)];
                            }
                        }

                        unroll! {
                            for x in 0..6 {
                                let x_chi = 3 * x;
                                // println!("a[{}] = array[{}] ^ (a[{}] & (!array[{}]));", idx(x_chi, y), x_chi % 5, idx(x_chi + 2, y_step), (x_chi + 1) % 5);
                                a[idx(x_chi, y)] = array[x_chi % 5] ^ (a[idx(x_chi + 2, y)] & (!array[(x_chi + 1) % 5]));
                            }
                        }
                    }
                };

                // Inverse pi and inverse rho
                let mut last = a[1];
                unroll! {
                    for y in 1..24 {
                        let x = 23 - y;

                        array[0] = a[$crate::PI[x]];
                        a[$crate::PI[x]] = last.rotate_right($crate::RHO[x + 1]);
                        last = array[0];
                    }
                }
                a[1] = last.rotate_right($crate::RHO[0]);


                // Inverse theta
                unroll! {
                    for x in 0..5 {
                        array[x] = a[idx(x, 0)];
                        unroll! {
                            for y in 1..5 {
                                array[x] ^= a[idx(x, y)];
                            }
                        }
                    }
                }

                const LANE_SIZE: usize = 64;
                let mut inverse_positions: [u64; 5] = [0; 5];

                unroll! {
                    for x in 0..5 {
                        inverse_positions[x] ^= $crate::inverse::INVERSE_THETA_POS64[x];
                    }
                }

                for _z in 0..LANE_SIZE {
                    for x_off in 0..5 {
                        unroll! {
                            for x in 0..5 {
                                unroll! {
                                    for y in 0..5 {
                                        if (inverse_positions[x_off] & 1) != 0 {
                                            // TODO: optimise array indexing.
                                            let mut array_idx: isize = x as isize - x_off as isize;
                                            array_idx %= 5;

                                            if array_idx < 0 {
                                                array_idx += 5;
                                            }

                                            a[idx(x, y)] ^= array[array_idx as usize];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for x_off in 0..5 {
                        array[x_off] = array[x_off].rotate_left(1);
                        inverse_positions[x_off] >>= 1;
                    }
                }
            }
        }
    }
}

pub(crate) use inverse_keccak_function;

#[cfg(test)]
mod tests {
    use crate::keccakf::RC;
    use crate::Buffer;

    // Define a Keccak-F and its inverse.
    keccak_function!("`keccak-f[1600, 8]`", keccakf_8, 8, RC);
    inverse_keccak_function!("`inverse-keccak-f[1600, 8]`", inverse_keccakf_8, 8, RC);

    #[test]
    fn inverse_keccak_f() {
        // Test state is all zeroes
        let mut test_buf = Buffer::default();

        // Apply 8 rounds of Keccak-F.
        keccakf_8(test_buf.words());
        println!("keccak_f_result: {:?}", test_buf.words());

        // Apply 8 rounds of the inverse Keccak-F.
        inverse_keccakf_8(test_buf.words());
        println!("inverse_keccak_f_result: {:?}", test_buf.words());

        // Verify that we've got all zeroes back as a result.
        assert!(test_buf.words().iter().all(|word| *word == 0));
    }
}
