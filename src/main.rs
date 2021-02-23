use hex;
use pow_rs::*;

pub const SEED: u128 = 73237431696005972674723595250817150843;

fn p(bitrate: usize) {
    let input = b"xyz";
    let mut output = [0; 48];
    // let vdf = SlothVdf::new(Integer::from(SEED), 6656, 0);

    let mut sha3 = Keccak::new(bitrate);
    sha3.update(input);
    sha3.finalize(&mut output);

    // println!("OUTPUT: {:?}", output.to_vec());
    // println!("OUTPUT LEN: {:?}", output.len());

    // println!("Bytes: {:?}", output);
    // println!("Byte length: {:?}", output.len());
    println!("{:}", hex::encode(&output.to_vec()));
    // println!("HEX LENGTH: {:?}", hex::encode(&output.to_vec()).len());
}

fn main() {
    p(136);

    let input = b"xyz";
    let mut sha3 = Keccak::new(136);
    sha3.update(input);
    println!("{:}", hex::encode(&sha3.finalize_with_penalty(0)));

    let mut sha3 = Keccak::new(136);
    sha3.update(input);
    println!("{:}", hex::encode(&sha3.finalize_with_penalty(1000000)));
}
