//! VDF permutation functions.

use crate::sloth;
use crate::{Buffer, VdfPermutation};
use rug::Integer;
use std::u64;

/// Implements the Sloth VDF permutation algorithm.
pub struct SlothVdf {
    seed: Integer,
    bitrate_bits: u64,
    t_val: u64,
}

impl SlothVdf {
    /// Create a new Sloth VDF instance.
    ///
    /// ## Parameters
    /// - `seed` is the seed number `p`.
    /// - `bitrate_bits` is used to compute the `t` value (the resulting difficulty).
    /// - `t_val` is the desired puzzle difficulty.
    pub fn new(seed: Integer, bitrate_bits: u64, t_val: u64) -> SlothVdf {
        SlothVdf {
            seed,
            bitrate_bits,
            t_val,
        }
    }

    /// Create a new Sloth VDF instance with the default seed parameter.
    ///
    /// ## Parameters
    /// - `seed` is the seed number `p`.
    /// - `bitrate_bits` is used to compute the `t` value (the resulting difficulty).
    /// - `t_val` is the desired puzzle difficulty.
    pub fn default_seed(bitrate_bits: u64, t_val: u64) -> SlothVdf {
        SlothVdf {
            seed: Integer::from(sloth::SEED),
            bitrate_bits,
            t_val,
        }
    }
}

// VDF permutation. It operates on and delays(?) the passed-in buffer.
impl VdfPermutation for SlothVdf {
    fn execute(&self, buffer: &mut Buffer) {
        // Pre-calculate an exponent for quad_res: (seed - 1) // 2
        let quad_res_exp = (self.seed.clone() - 1u64).div_rem_floor(2u64.into()).0;

        // We are going to loop over the values in the state memory and randomly overwrite
        // one of them with the new updated value from the vdf (ok right now not randomly
        // eventually this will be random)
        // TODO: make this random
        let words = buffer.words();

        'outer: for x in 0..5 {
            for y_count in 0..5 {
                let y = y_count * 5;

                let mut val = Integer::from(words[x + y]);

                // println!(
                //     "x: {}, quad_res = {}",
                //     val,
                //     val.clone().pow_mod(&quad_res_exp, &self.seed).unwrap()
                // );
                let is_last = false; // i == len;

                if is_last || val.clone().pow_mod(&quad_res_exp, &self.seed).unwrap() != 1 {
                    val = val.div_rem_floor(self.seed.clone()).1;

                    // We want a family of vdfs so define many expressions for t here
                    let t = self.t_val / self.bitrate_bits; // lower rate needs higher t

                    let mod_op = sloth::eval(&self.seed, &val, t)
                        .div_rem_floor(Integer::from(u64::MAX))
                        .1
                        .to_u64_wrapping();
                    // println!("mod_op = {}", mod_op);

                    words[x + y] = mod_op;

                    break 'outer;
                }
            }
        }
    }
}
