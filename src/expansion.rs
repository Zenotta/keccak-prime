//! Implements the expansion function.

use aes_gcm_siv::aead::{generic_array::GenericArray, AeadInPlace, NewAead};
use aes_gcm_siv::{Aes128GcmSiv, Aes256GcmSiv};

/// Hash inputs sizes, in bytes.
pub const INPUT_HASH_SIZE: usize = 32; // 256 bits

/// Input nonce size, in bytes.
pub const NONCE_SIZE: usize = 16; // 128 bits

/// Initialization vector size used in the AES-GCM implementation.
pub const AES_IV_SIZE: usize = 12; // 96 bits

/// Expansion function result size, in bytes.
pub const OUTPUT_SIZE: usize = 136; // 1088 bits

/// Size of output for each padding block.
const AES128_OUTPUT_SIZE: usize = 16; // 128 bits

/// Derives a symmetric encryption key for the AES-256 block cipher.
fn derive_aes_key(prev_hash: [u8; INPUT_HASH_SIZE], root_hash: [u8; INPUT_HASH_SIZE]) -> [u8; 32] {
    let mut xor_result = [0u8; INPUT_HASH_SIZE];
    for i in 0..INPUT_HASH_SIZE {
        xor_result[i] = prev_hash[i] ^ root_hash[i];
    }
    xor_result
}

/// Use AES-128 block cipher to generate padding data for the expansion function.
fn encrypt_pad_round(
    cipher: &Aes128GcmSiv,
    round_num: u8,
) -> Result<[u8; AES128_OUTPUT_SIZE], aes_gcm_siv::aead::Error> {
    // We use the round number as plaintext.
    let mut counter = [round_num; AES128_OUTPUT_SIZE];
    let _auth_tag = cipher.encrypt_in_place_detached(
        GenericArray::from_slice(&[round_num; AES_IV_SIZE]),
        &[0u8; 0], // we don't have any additional data
        &mut counter,
    );
    Ok(counter)
}

/// Takes a previous hash, root merkle hash and nonce as an input.
/// Outputs a byte sequence suitable to be used in a VDF permutation function.
pub fn expand(
    prev_hash: [u8; INPUT_HASH_SIZE],
    root_hash: [u8; INPUT_HASH_SIZE],
    nonce: [u8; NONCE_SIZE],
) -> Result<Vec<u8>, aes_gcm_siv::aead::Error> {
    // Derive an AES key from the previous & Merkle tree hashes.
    let derived_key = derive_aes_key(prev_hash, root_hash);

    let key = GenericArray::from_slice(&derived_key);
    let cipher = Aes256GcmSiv::new(&key);

    // Encrypt nonce with the derived key.
    // This becomes our new key which we'll use to encrypt further data.
    let mut ciphertext = Vec::from(nonce);

    // 'encrypt_detached' means we _don't_ concatenate the authentication tag with the cipher output
    // because we want the cipher to be of a particular size (128 bits) to be used as a key.
    let _auth_tag = cipher.encrypt_in_place_detached(
        // We use a zero nonce as an initialization vector.
        &GenericArray::from_slice(&[0; AES_IV_SIZE]),
        &[0u8; 0], // we don't have any additional data
        &mut ciphertext,
    )?;

    // Write the ciphertext we've got to the beginning of this function result.
    let mut result = Vec::with_capacity(OUTPUT_SIZE);
    result.extend_from_slice(&ciphertext[0..NONCE_SIZE]);

    // Derive an AES key for the padding round. We reuse the same key for all rounds.
    let pad_rounds_key = GenericArray::from_slice(&ciphertext[0..16]);
    let pad_cipher = Aes128GcmSiv::new(&pad_rounds_key);

    // Pad the result until it has the expected size.
    let pad_rounds_num =
        ((OUTPUT_SIZE - NONCE_SIZE) as f32 / AES128_OUTPUT_SIZE as f32).ceil() as usize;

    let mut result = (0..pad_rounds_num).try_fold(result, |mut result, round_num| {
        // We increment the round number to make IV different from the first one we used above.
        let ciphertext = encrypt_pad_round(&pad_cipher, (round_num + 1) as u8)?;
        result.extend_from_slice(&ciphertext);
        Ok(result)
    })?;

    result.truncate(OUTPUT_SIZE);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify that the output has an expected size.
    #[test]
    fn verify_output_size() {
        let prev_hash = [1u8; INPUT_HASH_SIZE];
        let root_hash = [2u8; INPUT_HASH_SIZE];
        let nonce = [3u8; NONCE_SIZE];

        let res = expand(prev_hash, root_hash, nonce).expect("expand function failed");
        // dbg!(&res);
        assert_eq!(res.len(), OUTPUT_SIZE);
    }
}
