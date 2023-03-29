//! Pseudorandom function based on the ProgPoW algorithm.
//! https://eips.ethereum.org/EIPS/eip-1057#specification

use crate::{constants::*, prng::PrngState};
use std::{convert::TryInto, error::Error, fmt};

use crate::kiss99::{fnv1a, kiss99, Kiss99State, FNV_OFFSET_BASIS};

const MERGE_MAX_R: u16 = 4;
const MATH_MAX_R: u16 = 10;

/// Error occurred during the pseudorandom function execution.
#[derive(Debug)]
pub enum PrfError {
    #[cfg(feature = "prf_vulkan")]
    /// Vulkan runtime error (such as a device initialization error).
    VulkanError(String, Option<Box<dyn std::error::Error + 'static>>),
}

impl fmt::Display for PrfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "prf_vulkan")]
            PrfError::VulkanError(description, cause) => {
                if let Some(cause) = cause {
                    write!(f, "Vulkan error: {}, caused by: {}", description, cause)
                } else {
                    write!(f, "Vulkan error: {}", description)
                }
            }
        }
    }
}

impl Error for PrfError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            #[cfg(feature = "prf_vulkan")]
            PrfError::VulkanError(_, Some(err)) => Some(&**err),
            _ => None,
        }
    }
}

#[cfg(test)]
fn debug_mix_data(mix_data: &[u32; PROGPOW_REGS * PROGPOW_LANES]) -> String {
    mix_data
        .iter()
        .map(|num| hex::encode(num.to_be_bytes()))
        .fold(String::new(), |a, b| a + &b + " ")
}

#[cfg(test)]
fn debug_mix_lane(mix_state_lane: &[u32]) -> String {
    let mix_state_hex = mix_state_lane
        .iter()
        .map(|l| hex::encode(l.to_be_bytes()))
        .fold(String::new(), |a, b| a + &b + " ");
    let digest_lane = reduce_mix(&mix_state_lane[0..32].try_into().unwrap());

    mix_state_hex + "\n\n---\ndigest: " + &hex::encode(digest_lane.to_be_bytes())
}

/// Pseudorandom function.
pub fn prf(
    block_k_root_hash: &[u8; INPUT_HASH_SIZE],
    witness: &[u8; WITNESS_SIZE],
    loop_count: u16,
) -> Result<[u8; 136], PrfError> {
    let mut mix_state = [0u32; PROGPOW_REGS * PROGPOW_LANES];

    // initialize PRNG with data from witness
    for lane_id in 0..PROGPOW_LANES {
        let seed = u64::from_be_bytes(
            witness[(lane_id * 8)..((lane_id + 1) * 8)]
                .try_into()
                .expect("unexpected witness input size"), // this basically shouldn't happen because it's a fixed-size array
        );
        fill_mix(
            seed,
            lane_id as u32,
            &mut mix_state[(lane_id * PROGPOW_REGS)..(lane_id * PROGPOW_REGS + PROGPOW_REGS)],
        );
    }

    // dbg!(debug_mix_data(&mix_state));

    // Generate and compile a program for execution on GPU.
    let updated_mix_data = if loop_count > 0 {
        let prog_source = generate_kernel(block_k_root_hash, loop_count);

        println!("{}", prog_source);

        let prog = vulkan::compile_kernel(prog_source).unwrap();

        vulkan::execute(&prog, mix_state)?
    } else {
        mix_state
    };

    // dbg!(debug_mix_data(&updated_mix_data));

    // Reduce mix data to a per-lane 32-bit digest
    let mut result = Vec::<u8>::with_capacity(136);

    for lane_id in 0..PROGPOW_LANES {
        let mix_state_lane = updated_mix_data
            [(lane_id * PROGPOW_REGS)..(lane_id * PROGPOW_REGS + PROGPOW_REGS)]
            .try_into()
            .unwrap();

        let digest_lane = reduce_mix(&mix_state_lane);

        // dbg!(debug_mix_lane(&mix_state_lane));

        // FIXME: excessive memory copying
        result.extend_from_slice(&digest_lane.to_be_bytes());
    }

    // Concat lane digests with the remainder of the witness number
    result.extend_from_slice(&witness[(PROGPOW_LANES * 8)..]);

    Ok(result.try_into().expect("unexpected output size"))
}

/// Returns a digest of mix_state.
fn reduce_mix(mix_state: &[u32; PROGPOW_REGS]) -> u32 {
    let mut digest_lane = FNV_OFFSET_BASIS;

    for lane in mix_state.iter().take(PROGPOW_REGS) {
        digest_lane = fnv1a(digest_lane, *lane);
    }

    digest_lane
}

/// Populates an array of u32 values used by each lane in the hash calculations.
// todo: use const for `mix_state` size
fn fill_mix(seed: u64, lane_id: u32, mix_state: &mut [u32]) {
    // Use FNV to expand the per-warp seed to per-lane
    // Use KISS to expand the per-lane seed to fill mix
    let mut rng_state = Kiss99State::default();
    rng_state.z = fnv1a(FNV_OFFSET_BASIS, seed as u32);
    rng_state.w = fnv1a(rng_state.z, (seed >> 32) as u32);
    rng_state.jsr = fnv1a(rng_state.w, lane_id);
    rng_state.jcong = fnv1a(rng_state.jsr, lane_id);

    for k in &mut mix_state[..] {
        *k = kiss99(&mut rng_state);
    }
}

/// Generates a single iteration of the ProgPoW loop logic.
fn generate_progpow_loop_iter(block_k_root_hash: &[u8; INPUT_HASH_SIZE]) -> String {
    let mut buffer = String::with_capacity(1024);
    let mut prng_state = PrngState::new(block_k_root_hash, 0xFE); // todo: use const for usage number

    // Helper function that returns a random number
    let mut get_rand = |max: u16| prng_state.get_integer(max);

    // Initialize the program seed and sequences.
    let mut mix_seq_dst = [0u32; PROGPOW_REGS];
    let mut mix_seq_dst_cnt = 0;

    // Create a random sequence of mix destinations for merge() and mix sources for cache reads
    // guarantees every destination merged once
    // guarantees no duplicate cache reads, which could be optimized away
    // Uses Fisher-Yates shuffle
    #[allow(clippy::needless_range_loop)] // clippy's suggestion is less readable
    for i in 0..PROGPOW_REGS {
        mix_seq_dst[i] = i as u32;
    }
    for i in (1..PROGPOW_REGS).rev() {
        let j = get_rand(i as u16);
        mix_seq_dst.swap(i, j as usize);
    }

    for _i in 0..PROGPOW_CNT_MATH {
        // TODO: add cached memory access instructions which require DAG.
        // DAG can be generated using a pseudorandom algorithm in this case.
        // For now, we can just assume that PROGPOW_CNT_CACHE is set to 0.

        // Random math - generate 2 unique sources
        let src_rnd = get_rand((PROGPOW_REGS * (PROGPOW_REGS - 1) - 1) as u16) as usize;

        let src1 = src_rnd % PROGPOW_REGS; // 0 <= src1 < PROGPOW_REGS
        let mut src2 = src_rnd / PROGPOW_REGS; // 0 <= src2 < PROGPOW_REGS - 1

        if src2 >= src1 {
            // src2 is now any reg other than src1
            src2 += 1;
        }

        let sel1 = get_rand(MATH_MAX_R);

        let dst = mix_seq_dst[mix_seq_dst_cnt % PROGPOW_REGS];
        let sel2 = get_rand(MERGE_MAX_R);

        mix_seq_dst_cnt += 1;

        // rand math function
        buffer.push_str("data = ");
        buffer.push_str(&generate_random_math_func(
            &format!("mix[{src1}]"),
            &format!("mix[{src2}]"),
            sel1 as u32,
        ));
        buffer.push_str(";\n");

        // merge
        buffer.push_str(&format!("mix[{dst}] = "));
        buffer.push_str(&generate_merge_func(
            &format!("mix[{dst}]"),
            "data",
            sel2 as u32,
        ));
        buffer.push_str(";\n");
    }

    buffer
}

/// Generates an OpenCL C kernel for PRF.
/// This can be used either in the Vulkan runtime (with the clspv compiler) or in the OpenCL one.
fn generate_kernel(block_k_root_hash: &[u8; INPUT_HASH_SIZE], loop_count: u16) -> String {
    let mut buffer = String::with_capacity(1024);

    // TODO: move fill_mix and most other functions _inside_ the kernel.
    buffer.push_str(&format!(
        "#define PROGPOW_REGS {PROGPOW_REGS}\n\
        #define PROGPOW_LANES {PROGPOW_LANES}\n\
        typedef unsigned int uint32_t;\n\
        typedef unsigned long uint64_t;\n\
        \
        static void progPowLoop(__global uint32_t *mix) {{\n\
            uint32_t data;\n",
    ));

    // Unroll the loop count.
    for _i in 0..loop_count {
        buffer.push_str(&generate_progpow_loop_iter(block_k_root_hash));
    }
    buffer.push_str("}\n");

    // TODO: utilize PROGPOW_DAG_LOADS?

    buffer.push_str(
        "__kernel void ProgPoW(__global uint32_t mix[PROGPOW_LANES * PROGPOW_REGS]) {{\n\
            #pragma unroll 1\n\
            for (uint32_t l = 0; l < PROGPOW_LANES; l++) {{\n\
                barrier(CLK_LOCAL_MEM_FENCE);\n\
                progPowLoop(&mix[l * PROGPOW_REGS]);\n\
            }}\n\
        }}\n",
    );

    buffer
}

/// Merge new data from b into the value in a.
/// Assuming A has high entropy only do ops that retain entropy even if B is low entropy.
/// (i.e. don't do A & B)
fn generate_merge_func(a: &str, b: &str, rand: u32) -> String {
    match rand % 4 {
        0 => format!("({a} * 33) + {b}"),
        1 => format!("({a} ^ {b}) * 33"),
        // prevent rotate by 0 which is a NOP
        2 => format!(
            "rotate({a}, (uint32_t)({bits})) ^ {b}",
            bits = ((rand >> 16) % 31) + 1
        ),
        3 => format!(
            "rotate({a}, (uint32_t)(32 - {bits})) ^ {b}",
            bits = ((rand >> 16) % 31) + 1
        ),
        _ => unreachable!(),
    }
}

fn generate_random_math_func(a: &str, b: &str, rand: u32) -> String {
    match rand % 11 {
        0 => format!("{a} + {b}"),
        1 => format!("{a} * {b}"),
        2 => format!("mul_hi({a}, {b})"),
        3 => format!("min({a}, {b})"),
        4 => format!("rotate({a}, (uint32_t)({b}))"),
        5 => format!("rotate({a}, (uint32_t)(32 - {b}))"),
        6 => format!("{a} & {b}"),
        7 => format!("{a} | {b}"),
        8 => format!("{a} ^ {b}"),
        9 => format!("clz({a}) + clz({b})"),
        10 => format!("popcount({a}) + popcount({b})"),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::{fill_mix, generate_kernel, prf, reduce_mix};
    use crate::constants::*;

    #[test]
    fn generate_kernel_test() {
        let kernel = generate_kernel(&[0; 32], 2);
        println!("\n===\n{}", kernel);
    }

    #[test]
    fn progpow_test_vectors() {
        {
            let expected = [
                0x9e, 0xd7, 0xe0, 0x74, 0x55, 0x42, 0xbc, 0xfe, 0xa5, 0xc8, 0x94, 0xfb, 0x79, 0xde,
                0x4c, 0x64, 0x24, 0x22, 0x42, 0xfd, 0x0b, 0x48, 0x0f, 0x0a, 0x71, 0x3c, 0x7a, 0xec,
                0x30, 0x7d, 0xec, 0xcd, 0x8e, 0x5c, 0xda, 0xe2, 0xf1, 0x4e, 0x9c, 0x9b, 0x32, 0x63,
                0xf4, 0x70, 0x76, 0x6e, 0x6a, 0xf9, 0x93, 0xad, 0x0e, 0x19, 0x20, 0x84, 0xe7, 0x7e,
                0x5b, 0x6e, 0xcc, 0x9d, 0xe1, 0xec, 0x0e, 0x8f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ];

            let res = prf(&[0; INPUT_HASH_SIZE], &[0; WITNESS_SIZE], 0).unwrap();

            assert_eq!(res, expected);
        }
        {
            let k = [
                0x9f, 0x07, 0x4d, 0x72, 0xe0, 0xc1, 0xbb, 0x5b, 0xdd, 0x26, 0x75, 0xe2, 0x36, 0xdb,
                0xbb, 0x5a, 0xae, 0xef, 0x08, 0x9c, 0x67, 0xb6, 0xe3, 0x5f, 0xe7, 0x41, 0x80, 0xe3,
                0x46, 0x1f, 0x10, 0x4f,
            ];
            let witness = [
                0xce, 0xbe, 0x6e, 0xfc, 0xb4, 0x4c, 0xa2, 0xa2, 0x28, 0xae, 0x7e, 0xd4, 0xda, 0xe5,
                0xd1, 0xd2, 0xcb, 0x3f, 0xa2, 0x15, 0x37, 0xb1, 0xb5, 0x2d, 0x88, 0x75, 0x9c, 0x74,
                0x41, 0x2f, 0x68, 0xb2, 0x8a, 0xe1, 0xaa, 0x4f, 0x82, 0x25, 0x80, 0x20, 0xcb, 0xe9,
                0x66, 0xd0, 0x0c, 0xe8, 0x61, 0x5b, 0x2a, 0x23, 0x45, 0xf6, 0xfd, 0xda, 0xbd, 0x83,
                0x03, 0x36, 0xe9, 0x39, 0x4a, 0xbd, 0x59, 0x71, 0x04, 0xc1, 0xf8, 0xee, 0xca, 0x5d,
                0xb7, 0x61, 0x26, 0x30, 0x79, 0x3f, 0x14, 0xed, 0xfb, 0x70, 0x94, 0xd5, 0xdc, 0x6e,
                0x4a, 0x27, 0xbd, 0xe1, 0xef, 0x80, 0x73, 0x7a, 0x78, 0xac, 0xf1, 0x64, 0xbe, 0xbc,
                0x2c, 0x99, 0x08, 0xbf, 0x0a, 0x02, 0x5f, 0x03, 0xbe, 0x57, 0x40, 0x76, 0xdd, 0xd4,
                0x6b, 0x66, 0x3d, 0x03, 0x8b, 0xd0, 0x6f, 0x7c, 0xe7, 0xe4, 0x60, 0x8d, 0x31, 0xb1,
                0x2d, 0xd9, 0x60, 0x7b, 0x5e, 0x37, 0xa6, 0x6b, 0x4a, 0x5c, 0x35, 0x4e, 0x16, 0xd0,
                0x5f, 0x35, 0x7a, 0x11, 0x40, 0x22, 0x47, 0xd4, 0xb5, 0x15, 0x63, 0x91, 0xd8, 0x9b,
                0x22, 0x50, 0x9a, 0x20, 0x51, 0x68, 0xed, 0x11, 0xe0, 0x57, 0x45, 0x2d, 0xfa, 0xb9,
                0xf6, 0xcc, 0x6c, 0x77, 0x99, 0x61, 0x4a, 0x51, 0xf4, 0xdc, 0x6a, 0x34, 0x25, 0x3a,
                0x69, 0xb7, 0x47, 0x4a, 0xa6, 0x08, 0x95, 0x70, 0x6e, 0x9b, 0xc2, 0x85, 0xbe, 0xd7,
                0x72, 0x08, 0x15, 0x35,
            ];
            let expected = [
                0x7d, 0x32, 0x1e, 0xe5, 0x82, 0xc9, 0xf3, 0x1f, 0x1f, 0xfe, 0xf3, 0x64, 0xcc, 0x1c,
                0x6e, 0x2d, 0x4b, 0x1c, 0x6a, 0x4f, 0x06, 0x12, 0x06, 0xf2, 0xad, 0x36, 0x65, 0x46,
                0xe0, 0x26, 0x10, 0x05, 0x1d, 0x8f, 0x5e, 0x76, 0xb0, 0xb9, 0xa9, 0xa6, 0x7f, 0xe8,
                0x74, 0xfb, 0x75, 0x62, 0xa9, 0x7a, 0xc9, 0x98, 0xc2, 0xa7, 0x24, 0x2d, 0x84, 0xba,
                0x87, 0x6b, 0xea, 0x8e, 0x29, 0x85, 0xaf, 0x77, 0x60, 0x7b, 0x5e, 0x37, 0xa6, 0x6b,
                0x4a, 0x5c, 0x35, 0x4e, 0x16, 0xd0, 0x5f, 0x35, 0x7a, 0x11, 0x40, 0x22, 0x47, 0xd4,
                0xb5, 0x15, 0x63, 0x91, 0xd8, 0x9b, 0x22, 0x50, 0x9a, 0x20, 0x51, 0x68, 0xed, 0x11,
                0xe0, 0x57, 0x45, 0x2d, 0xfa, 0xb9, 0xf6, 0xcc, 0x6c, 0x77, 0x99, 0x61, 0x4a, 0x51,
                0xf4, 0xdc, 0x6a, 0x34, 0x25, 0x3a, 0x69, 0xb7, 0x47, 0x4a, 0xa6, 0x08, 0x95, 0x70,
                0x6e, 0x9b, 0xc2, 0x85, 0xbe, 0xd7, 0x72, 0x08, 0x15, 0x35,
            ];

            let res = prf(&k, &witness, 2).unwrap();

            assert_eq!(res, expected);
        }
    }

    #[test]
    fn reduce_mix_test_vectors() {
        {
            let mix = [0; PROGPOW_REGS];
            let digest_lane = reduce_mix(&mix);
            assert_eq!(digest_lane, 0x0b2ae445);
        }
        {
            let mix: [u32; PROGPOW_REGS] = [
                0x40ca735e, 0x05a48bcd, 0xe07cd0c1, 0xee169196, 0xfa1e2409, 0xb2dc2c30, 0x311ed967,
                0xbeb4feec, 0xe9294c6e, 0xd819adad, 0x680e4e2f, 0xfefd9704, 0x9b51fde0, 0x7404259f,
                0x274f2928, 0xbfa205f8, 0x9a8bf385, 0x3cdcfd9d, 0xd563a604, 0x3729c15d, 0x37df2b12,
                0x52d37e21, 0xd2a3cda7, 0x3f04114c, 0xd078c56e, 0xd9c4ce1d, 0x03818894, 0x7c0cd1a9,
                0xcb3f29ea, 0x1fee194d, 0x6fbdac64, 0x7f4b1803,
            ];
            let digest_lane = reduce_mix(&mix);
            assert_eq!(digest_lane, 0x7ec9241b);
        }
    }

    #[test]
    fn fill_mix_test_vectors() {
        let hash_seed = 0xEE304846DDD0A47B;

        {
            let lane_id = 0;
            let mut mix_state = [0u32; PROGPOW_REGS];
            fill_mix(hash_seed, lane_id, &mut mix_state);

            assert_eq!(mix_state[0], 0x10c02f0d);
            assert_eq!(mix_state[3], 0x43f0394d);
            assert_eq!(mix_state[5], 0xc4e89d4c);
        }

        {
            let lane_id = 13;
            let mut mix_state = [0u32; PROGPOW_REGS];
            fill_mix(hash_seed, lane_id, &mut mix_state);

            assert_eq!(mix_state[0], 0x4e46d05d);
            assert_eq!(mix_state[3], 0x70712177);
            assert_eq!(mix_state[5], 0xbef18d17);
        }
    }
}

/// GPU compute runtime implementation based on Vulkan
#[cfg(feature = "prf_vulkan")]
mod vulkan {
    use std::io;
    use vulkano::{
        buffer::{Buffer, BufferAllocateInfo, BufferContents, BufferError, BufferUsage},
        command_buffer::{
            allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BuildError,
            CommandBufferBeginError, CommandBufferExecError, CommandBufferUsage,
            PipelineExecutionError,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator, DescriptorSetCreationError,
            PersistentDescriptorSet, WriteDescriptorSet,
        },
        device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceCreationError,
            DeviceExtensions, QueueCreateInfo, QueueFlags,
        },
        instance::{Instance, InstanceCreateInfo, InstanceCreationError},
        memory::allocator::StandardMemoryAllocator,
        pipeline::{
            compute::ComputePipelineCreationError, ComputePipeline, Pipeline, PipelineBindPoint,
        },
        shader::{ShaderCreationError, ShaderModule},
        sync::{self, FlushError, GpuFuture},
        LoadingError, VulkanError, VulkanLibrary,
    };

    use super::PrfError;

    impl From<InstanceCreationError> for PrfError {
        fn from(value: InstanceCreationError) -> Self {
            PrfError::VulkanError(
                "failed to create a device instance".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<DescriptorSetCreationError> for PrfError {
        fn from(value: DescriptorSetCreationError) -> Self {
            PrfError::VulkanError(
                "failed to create a descriptor set".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<ShaderCreationError> for PrfError {
        fn from(value: ShaderCreationError) -> Self {
            PrfError::VulkanError(
                "failed to initialize a shader".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<DeviceCreationError> for PrfError {
        fn from(value: DeviceCreationError) -> Self {
            PrfError::VulkanError(
                "failed to create a Vulkan device".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<BufferError> for PrfError {
        fn from(value: BufferError) -> Self {
            PrfError::VulkanError("buffer error".to_owned(), Some(Box::new(value)))
        }
    }

    impl From<BuildError> for PrfError {
        fn from(value: BuildError) -> Self {
            PrfError::VulkanError(
                "command buffer build error".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<PipelineExecutionError> for PrfError {
        fn from(value: PipelineExecutionError) -> Self {
            PrfError::VulkanError(
                "Vulkan pipeline execution error".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<CommandBufferBeginError> for PrfError {
        fn from(value: CommandBufferBeginError) -> Self {
            PrfError::VulkanError("Vulkan buffer error".to_owned(), Some(Box::new(value)))
        }
    }

    impl From<ComputePipelineCreationError> for PrfError {
        fn from(value: ComputePipelineCreationError) -> Self {
            PrfError::VulkanError(
                "failed to create a compute pipeline".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<CommandBufferExecError> for PrfError {
        fn from(value: CommandBufferExecError) -> Self {
            PrfError::VulkanError(
                "failed to execute a command buffer".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<FlushError> for PrfError {
        fn from(value: FlushError) -> Self {
            PrfError::VulkanError(
                "failed to flush a command buffer".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<LoadingError> for PrfError {
        fn from(value: LoadingError) -> Self {
            PrfError::VulkanError(
                "failed to initialize Vulkan".to_owned(),
                Some(Box::new(value)),
            )
        }
    }

    impl From<VulkanError> for PrfError {
        fn from(value: VulkanError) -> Self {
            PrfError::VulkanError("Vulkan runtime error".to_owned(), Some(Box::new(value)))
        }
    }

    #[cfg(feature = "prf_vulkan_build_clspv")]
    pub fn compile_kernel(kern: String) -> io::Result<Vec<u32>> {
        let output = clspv_sys::compile_from_source(&kern, Default::default());
        if output.ret_code == 0 {
            Ok(output.output)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, output.log))
        }
    }

    /// Compiles an OpenCL kernel into SPIR-V.
    /// Requires a `clspv` compiler.
    ///
    /// ## Returns
    ///
    /// A compiled SPIR-V binary.
    #[cfg(not(feature = "prf_vulkan_build_clspv"))]
    pub fn compile_kernel(kern: String) -> io::Result<Vec<u32>> {
        use std::{
            convert::TryInto,
            io::Write,
            process::{Command, Stdio},
        };

        let mut clspv = Command::new("clspv")
            .arg("-o")
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "clspv compiler not found"))?;

        let mut stdin = clspv.stdin.take().expect("Failed to open stdin");

        std::thread::spawn(move || {
            stdin
                .write_all(kern.as_bytes())
                .expect("Failed to write to stdin");
        });

        let output = clspv.wait_with_output().expect("Failed to read stdout");

        if output.status.success() {
            Ok(output
                .stdout
                .chunks_exact(4)
                .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
                .collect())
        } else {
            Err(io::Error::new(io::ErrorKind::Other, output.stderr))
        }
    }

    /// ## Inputs
    /// - `compute_shader` - SPIR-V binary of a compiled program.
    pub fn execute<T: BufferContents + Sized + Clone>(
        compute_shader: &[u32],
        input_data: T,
    ) -> Result<T, PrfError> {
        let library = VulkanLibrary::new()?;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enumerate_portability: true,
                ..Default::default()
            },
        )?;

        // Choose which physical device to use
        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            khr_variable_pointers: true,
            ..DeviceExtensions::empty()
        };

        // Get a device and a compute queue.
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()?
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .ok_or_else(|| {
                PrfError::VulkanError("no valid Vulkan device has been found".to_owned(), None)
            })?;

        // Initialise the device.
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;

        // Retrieve a single compute queue.
        let queue = queues
            .next()
            .ok_or_else(|| PrfError::VulkanError("can't find a compute queue".to_owned(), None))?;

        let pipeline = {
            let shader = unsafe { ShaderModule::from_words(device.clone(), compute_shader)? };

            ComputePipeline::new(
                device.clone(),
                shader.entry_point("ProgPoW").ok_or_else(|| {
                    PrfError::VulkanError("can't find a shader entry point".to_owned(), None)
                })?,
                &(),
                None,
                |_| {},
            )?
        };

        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        // Create a Vulkan buffer and pass the input data.
        let input_buffer = {
            Buffer::from_data(
                &memory_allocator,
                BufferAllocateInfo {
                    buffer_usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                input_data,
            )?
        };

        // Create a descriptor set for the buffer.
        let layout = pipeline.layout().set_layouts().get(0).ok_or_else(|| {
            PrfError::VulkanError("descriptor set layout hasn't been found".to_owned(), None)
        })?;
        let set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, input_buffer.clone())],
        )?;

        // Build a command buffer.
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .dispatch([1, 1, 1])?; // FIXME: correct number of groups

        let command_buffer = builder.build()?;

        // Execute the command buffer.
        let future = sync::now(device)
            .then_execute(queue, command_buffer)?
            .then_signal_fence_and_flush()?;

        // Block execution until the GPU has finished the operation.
        let timeout = None;
        future.wait(timeout)?;

        // Retrieve the buffer contents - it should now contain the mixed data.
        let output_buffer_content = input_buffer.read()?;
        Ok((*output_buffer_content).clone())
    }

    // Vulkan-specific tests
    #[cfg(test)]
    mod tests {
        use crate::{
            constants::{INPUT_HASH_SIZE, PROGPOW_LANES, PROGPOW_REGS},
            prf::{generate_kernel, generate_merge_func, generate_random_math_func},
        };

        use super::*;

        #[test]
        fn test_maths_functions() {
            let inputs = [
                0x8626bb1f_u32, // a
                0xbbdfbc4e,     // b
                0x3f4bdfac,
                0xd79e414f,
                0x6d175b7e,
                0xc4e89d4c,
                0x2eddd94c,
                0x7e70cb54,
                0x61ae0e62,
                0xe0596b32,
                0x8a81e396,
                0x3f4bdfac,
                0x8a81e396,
                0x7e70cb54,
                0xa7352f36,
                0xa0eb7045,
                0xc89805af,
                0x64291e2f,
                0x760726d3,
                0x79fc6a48,
                0x75551d43,
                0x3383ba34,
                0xea260841,
                0xe92c44b7,
            ];

            let random_vals = [
                0x883e5b49_u32,
                0x36b71236,
                0x944ecabb,
                0x3f472a85,
                0x3f472a85,
                0xcec46e67,
                0xdbe71ff7,
                0x59e7b9d8,
                0x1bdc84a9,
                0xc675cac5,
                0x2863ad31,
                0xf83ffe7d,
            ];

            let expected = [
                0x4206776d_u32,
                0x4c5cb214,
                0x53e9023f,
                0x2eddd94c,
                0x61ae0e62,
                0x1e3968a8,
                0x1e3968a8,
                0xa0212004,
                0xecb91faf,
                0x0ffb4c9b,
                0x00000003,
                0x0000001b,
            ];

            // Generate a kernel from the test vectors
            let mut buffer = String::with_capacity(1024);

            buffer.push_str(&format!(
                "typedef unsigned int uint32_t;\n\
                void __kernel ProgPoW(__global uint32_t test_data[{inputs_len}]) {{\n",
                inputs_len = inputs.len()
            ));
            for (i, rand) in random_vals.iter().enumerate() {
                buffer.push_str(&format!(
                    "test_data[{res_idx}] = {maths_fn};\n",
                    res_idx = i * 2 + 1,
                    maths_fn = generate_random_math_func(
                        &format!("test_data[{}]", i * 2),
                        &format!("test_data[{}]", i * 2 + 1),
                        *rand
                    )
                ));
            }
            buffer.push_str("}\n");

            println!("{}", buffer);

            let compute_shader = compile_kernel(buffer).unwrap();
            let res = execute(&compute_shader, inputs).unwrap();

            for (idx, x) in res.iter().skip(1).step_by(2).enumerate() {
                assert_eq!(expected[idx], *x);
            }
        }

        #[test]
        fn test_merge_functions() {
            let inputs = [
                0x3b0bb37d_u32, // a
                0xa0212004,     // b
                0x10c02f0d,
                0x870fa227,
                0x24d2bae4,
                0x0ffb4c9b,
                0xda39e821,
                0x089c4008,
            ];

            let random_vals = [0x9bd26ab0_u32, 0xd4f45515, 0x7fdbc2f2, 0x8b6cd8c3];

            let expected = [0x3ca34321_u32, 0x91c1326a, 0x2eddd94c, 0x8a81e396];

            // Generate a kernel from the test vectors
            let mut buffer = String::with_capacity(1024);

            buffer.push_str(&format!(
                "typedef unsigned int uint32_t;\n\
                void __kernel ProgPoW(__global uint32_t test_data[{inputs_len}]) {{\n",
                inputs_len = inputs.len()
            ));
            for (i, rand) in random_vals.iter().enumerate() {
                buffer.push_str(&format!(
                    "test_data[{res_idx}] = {maths_fn};\n",
                    res_idx = i * 2 + 1,
                    maths_fn = generate_merge_func(
                        &format!("test_data[{}]", i * 2),
                        &format!("test_data[{}]", i * 2 + 1),
                        *rand
                    )
                ));
            }
            buffer.push_str("}\n");

            println!("{}", buffer);

            let compute_shader = compile_kernel(buffer).unwrap();
            let res = execute(&compute_shader, inputs).unwrap();

            for r in res {
                println!("{}", hex::encode(r.to_be_bytes()));
            }

            for (idx, x) in res.iter().skip(1).step_by(2).enumerate() {
                assert_eq!(expected[idx], *x);
            }
        }

        #[test]
        fn test_progpow() {
            {
                let buffer = generate_kernel(&[0; INPUT_HASH_SIZE], 0);
                println!("{}", buffer);

                let compute_shader = compile_kernel(buffer).unwrap();
                let _res = execute(&compute_shader, [0u32; 512]);
            }
            {
                let mix_lane = [0u32; PROGPOW_REGS * PROGPOW_LANES];
                let expected = [
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                    0x00000000, 0x00000000,
                ];

                let buffer = generate_kernel(&[0; INPUT_HASH_SIZE], 1);
                println!("{}", buffer);

                let compute_shader = compile_kernel(buffer).unwrap();
                let res = execute(&compute_shader, mix_lane).unwrap();

                for (idx, r) in res.iter().enumerate() {
                    assert_eq!(*r, expected[idx]);
                }
            }

            {
                let mix_lane: [u32; 512] = [
                    0xe4ff0838, 0x5a2fd8ad, 0x83e1b790, 0xbbb986ae, 0x2c27a632, 0x746a4261,
                    0x83bef2d4, 0x2cce12f1, 0x82feb5b9, 0xedba62a6, 0x2cafa540, 0x587b112c,
                    0x8ef3d22c, 0xdeaf474d, 0xa54657ec, 0xd9ad4e7b, 0x271c5f1a, 0x442da58d,
                    0xc1109959, 0xcb50e2b4, 0xedebc8bd, 0x210b24de, 0x919355db, 0x4a6c5b8e,
                    0x9d033b60, 0xd6f75e06, 0x18a0cd2d, 0x3a013bc4, 0x767df337, 0x5dffd0a6,
                    0xf8ca0553, 0x5869f818, 0x907e0a97, 0x6f169919, 0x46826846, 0x7982eace,
                    0x738b78ce, 0x13dc3ad3, 0xaa2b1ed9, 0xa2be6967, 0x018433ef, 0x32a40b8d,
                    0x230fdd4b, 0xe1e623d5, 0x72ff3385, 0x8b6808e9, 0x506bc177, 0xe7a4873b,
                    0xd83937d1, 0xdcf46bf1, 0x346e8392, 0xbdcf3882, 0x0320d7a4, 0x371325b5,
                    0x31e7ae31, 0x808ee104, 0x1be97cae, 0x34e9b37c, 0x5564abbc, 0xedf21c09,
                    0x2f363846, 0x66e3d34e, 0xd0f6b2f0, 0x6f24173e, 0xfdf66b68, 0xe9dfbeba,
                    0xd19e724b, 0x4aa06d2e, 0x52125925, 0x3924dde6, 0x35e27357, 0xaae6a1a5,
                    0xf1bf66de, 0x4664732d, 0x9ead275e, 0xdf488c92, 0x1c37e406, 0x8ef9e323,
                    0x80b5939f, 0x37285f76, 0x451f50bc, 0x74b321cb, 0xb3beb3ab, 0xad2366e0,
                    0x05e03fe2, 0xe0de060b, 0xdef4f121, 0x829ea6ad, 0xc011f1f1, 0x382d465f,
                    0x992a3ca9, 0xe810dfad, 0x8f6c5d44, 0xbe4a4d5f, 0x426ff14a, 0xec419a43,
                    0xa29d5630, 0x032a67a2, 0x5bafa59d, 0x2c6a0881, 0x15e0db63, 0xd340f1df,
                    0xef736b6c, 0xd279dc30, 0x346c8337, 0x0954ea53, 0xa0055fec, 0x48d23f62,
                    0x52e54d17, 0x3748ed3e, 0x28f50b3d, 0x7920207c, 0x2e6178f5, 0x6e9b1edd,
                    0x3abac8de, 0xd5c1a65e, 0x0dc6af1a, 0xc3e6a936, 0x3cefcddc, 0x365641d8,
                    0x453685e6, 0x2e830fa9, 0xb501e77f, 0xacc23f2b, 0xddc0b8ba, 0x9b1149a3,
                    0x228af1e1, 0x9a7b90ad, 0xec2c6787, 0xee103470, 0xa4b48730, 0x4009e01d,
                    0xafa01355, 0x688e560a, 0xee8ab0a3, 0x93ee44b2, 0x323ed6a9, 0x2681fc63,
                    0x39f60494, 0x695cd599, 0x2aa7f4b7, 0x39629b23, 0xff5acd2c, 0xd0336a16,
                    0x01d94dc1, 0xd67f9df1, 0x1d48e3e1, 0x3fab70d2, 0x9eb71325, 0x4cbb3ac9,
                    0xb1579a9f, 0x41f0f1a1, 0x6ca6cf22, 0xbf02c43d, 0xad1eb418, 0x20af6b18,
                    0x59fcd51d, 0x18b6f63e, 0x8a5c80ea, 0x63dfa350, 0xf78fa570, 0x4aab8e3d,
                    0xb58326d7, 0x17977a97, 0x6477dc65, 0x03f79de3, 0x24029f57, 0x270a25c0,
                    0x16dfe76b, 0x0f61c057, 0x6f1df29c, 0x684efd85, 0x4ef98b3f, 0xa972c8af,
                    0xeb7c7278, 0x25c5a2de, 0x745c8559, 0xdb6fa3af, 0x9075f156, 0x2daf2d8c,
                    0x3036ed8f, 0xb432d630, 0x451711f7, 0x3cbea076, 0x5a992058, 0x96ccce97,
                    0x86e1e3bf, 0x70a4a711, 0x0e06a94e, 0xee928301, 0xd65c9653, 0xdfd0471d,
                    0xec75ca17, 0xe8f290a0, 0x3ca195b7, 0xc4a43646, 0x114514e4, 0x0279fab6,
                    0xe58f1b11, 0xa891deaf, 0x61386ddc, 0x8e754afd, 0x6475965f, 0x104065a8,
                    0xfd213040, 0xf84406bb, 0x250092c9, 0x18fe3811, 0x8bdff8cf, 0x7ca73396,
                    0x2e9e971e, 0xe52ed416, 0x24e6e0e7, 0x182fcc75, 0x505cfde4, 0x0f676182,
                    0x6e0485f3, 0xd0973470, 0xead72d4b, 0xd8329c90, 0x16b50db4, 0xa2ee29f5,
                    0x4a08d181, 0x4fd2babe, 0x14ae999d, 0xb767569e, 0xa5d31326, 0x2efe735a,
                    0x34f9145c, 0x3e2b8ccc, 0xcf0ab888, 0xb734473c, 0xf4474521, 0x410a89aa,
                    0x1fede499, 0x4c6db5d0, 0x8e0fbce6, 0xd80e1a18, 0xea905e88, 0xd150b6b8,
                    0x8594e5f2, 0xb80510ee, 0x0421b0ce, 0x3f6aa279, 0x1f942b31, 0xf3a42d30,
                    0x2cc80a38, 0xe91ae295, 0xd947db06, 0x6b6203ef, 0xe4a7de80, 0x019a9671,
                    0xeaa3c2ef, 0xbcff1463, 0x38e2b04c, 0x01098529, 0x5b00adb9, 0x690ce4b6,
                    0x80c90214, 0xa8011326, 0x1b59ab3a, 0xb1adc850, 0x89a03cda, 0xf5125794,
                    0x73761787, 0x69d26e85, 0xf4eb92ac, 0x279a026f, 0x79d2781f, 0xd32732a0,
                    0x5f5176ae, 0x7c2a5d0a, 0x12a7167c, 0x0e017b4b, 0x8956741c, 0x818f05c4,
                    0xf2f0f772, 0x6298e2b1, 0x9b6bb5b2, 0xeeaab443, 0xf9bbb3d0, 0xfe209daf,
                    0xc9d52b9e, 0x91de9993, 0x8b944c79, 0x44022b51, 0x4ca17d50, 0x5d5afab9,
                    0x5e8a03cc, 0xc315ba04, 0xb86b3916, 0x3f7f1a05, 0xaeb09db4, 0x66f76496,
                    0x83d7f29d, 0xd8d748d4, 0xe42662ee, 0x1c123003, 0xa91ab0c0, 0x2d6ab6c8,
                    0x3335a437, 0x6a5e2a17, 0x1bf44ff9, 0x5282e056, 0xc569671d, 0xbb3a6e54,
                    0x056e75f3, 0xedfb4fa0, 0x2b6ac829, 0x298724ea, 0xd03082c2, 0x33a5fc9f,
                    0x94e9b33f, 0x2e22d523, 0x96c8ac80, 0x36576707, 0x123cf098, 0xb212a1e2,
                    0x16c52e18, 0x3c1ed242, 0x58f74a19, 0x7e5b8264, 0x62029d56, 0x43ffba53,
                    0x53e554bb, 0x2aef44bd, 0x2f83a22a, 0x6cfc6849, 0xa8576c14, 0x51f8c2b9,
                    0xc5bc0b8d, 0x651f2ac1, 0xb17a703a, 0xca9966af, 0x8cfd7c87, 0x1b3e161c,
                    0x2c7c9fa6, 0x640c25e8, 0xdb9d159e, 0x37979f23, 0x361275e6, 0x1c82fd9a,
                    0xf7821460, 0xd87bdcc0, 0xa169ee07, 0x65511f6a, 0x027ceba6, 0x17cde2b0,
                    0x2144229e, 0xed187730, 0xd5df490e, 0x29d9e192, 0xeb09bf89, 0xa032e368,
                    0x1ac01895, 0xa8bd83b9, 0xaa2e9671, 0x874a6b40, 0xa46e79fa, 0x857d7805,
                    0xedce972d, 0xc1b9a987, 0x1d27e4fe, 0xe6a43ec5, 0x7806ad87, 0x104b0177,
                    0xd47fdfcb, 0x66d7bb08, 0x4de6ee61, 0x62d6e740, 0x32a8c3cd, 0x669c7606,
                    0xf88559d4, 0x608bbf5e, 0x8151d0da, 0x2fab1b7d, 0x4d029883, 0x5b6a333f,
                    0x8822803f, 0x3586749f, 0xc14376e6, 0x03ccb8c9, 0x3f2fdda5, 0xdd47cd77,
                    0x8cea9335, 0xee79a357, 0x49a71353, 0xa608125e, 0x8505cd20, 0x75a20d4c,
                    0x052a1edb, 0x509860a8, 0x87684387, 0x00bec149, 0x4f88e9d6, 0x069793c0,
                    0x6a74e6ed, 0x210bb469, 0xc4ec84e3, 0x1461d9db, 0xe2bf5ade, 0xab1786b3,
                    0x6ad8bf1d, 0xc2135b37, 0x3247f321, 0xd837d456, 0x5aaaee8c, 0xd2c2d0c9,
                    0x77c9ae87, 0xebd1d145, 0xdeb627d8, 0x3ec74b66, 0x13cc1895, 0x1e0487a0,
                    0xc4bca807, 0x5921165e, 0xb15095a8, 0x0328cfe0, 0xb7f6548b, 0x1cda58b4,
                    0x45fe7c0f, 0x2953ac40, 0xc7e3f5b6, 0x91c92193, 0x2dc039bc, 0x0833aa7b,
                    0x1ec403ea, 0x00ff1cd5, 0xa6d667e7, 0xbdec18dd, 0x764ac549, 0x4188a4ae,
                    0x814255c4, 0x7af942ce, 0x1894c6f3, 0x3b80465f, 0xa06cecbc, 0xdef6008f,
                    0xb2209ea2, 0xa8c0a737, 0xeda098fe, 0xd95c36de, 0xfe7faeac, 0x8fc140c3,
                    0x6609643f, 0x1f8b9671, 0x64b567bd, 0xb9e1f6bf, 0xa9ac3280, 0xe152e050,
                    0x6cc6595f, 0x0583dd89, 0x62afeef1, 0x939e1473, 0x2a767b54, 0x48dc80b4,
                    0x235b4e1d, 0x759b99a2, 0x9e258702, 0x29d6d6af, 0xec222463, 0x87fcc1c9,
                    0xc73c975b, 0xd4dc8fc6, 0x936f65ff, 0x27d0dc5e, 0xcd6f2d2c, 0x0a675e73,
                    0x78fca12d, 0xfb7a9734, 0x5c1f5770, 0x8f7ca2a7, 0xff423074, 0xee9f09a7,
                    0x1b6a7e94, 0xde2033a0, 0x8d7ed54a, 0x948bf422, 0xb586170b, 0x4a5c214a,
                    0xc9412b0a, 0x5dfb3eb4, 0x0ad4da5a, 0x68b0536d, 0xe216d028, 0xa9170a31,
                    0x8458a12d, 0x8ffcf6b7, 0xfd92243e, 0x2a3dc754, 0xad5a13c0, 0x490a725f,
                    0xcb1bf198, 0x5c114fe0, 0xf8c6fe61, 0x88995248, 0xd76ade10, 0x61b9de42,
                    0x1722ed18, 0x11f107ea, 0x45e882be, 0x631ea379, 0xd71ac9e9, 0x03a04ff4,
                    0x4f65cc4f, 0x80e90995, 0x02c82c3f, 0xf587f885, 0x851f1ee1, 0x04daf663,
                    0xbd4db5cb, 0xaf191f1f,
                ];
                let expected = [
                    0xabd11791, 0x5a2fd8ad, 0x83e1b790, 0x32ea5c6e, 0x2c27a632, 0x72d6466e,
                    0x3de6e0a0, 0x2cce12f1, 0x8ea6846a, 0x13b90b17, 0x2cafa540, 0x587b112c,
                    0x93a3d558, 0xdeaf474d, 0x224b33aa, 0x42e60075, 0x0aa8427c, 0x442da58d,
                    0xc1109959, 0xcb50e2b4, 0xde4f3ccb, 0x210b24de, 0xa6b2dc8b, 0x4a6c5b8e,
                    0x9d033b60, 0xb8c8725a, 0x2cba72cf, 0x3a013bc4, 0xa4622a42, 0xa53bea5a,
                    0x71ca49b2, 0x80977615, 0x3319f900, 0x6f169919, 0x46826846, 0xa9e0448e,
                    0x738b78ce, 0x56e957cf, 0x5be4381e, 0xa2be6967, 0x6524aacf, 0xfbe5c5b5,
                    0x230fdd4b, 0xe1e623d5, 0xa7f387c2, 0x8b6808e9, 0x76db6f96, 0x50e81915,
                    0xdf603210, 0xdcf46bf1, 0x346e8392, 0xbdcf3882, 0x111c10b2, 0x371325b5,
                    0x79d4b133, 0x808ee104, 0x1be97cae, 0xad8b5e6c, 0x01fa233c, 0xedf21c09,
                    0xedb59ec5, 0x93d9c4a1, 0x4381ef3d, 0x8e7ae5f1, 0x7e3ff428, 0xe9dfbeba,
                    0xd19e724b, 0x9eae12ee, 0x52125925, 0x71f6920f, 0x32f863c7, 0xaae6a1a5,
                    0x4d945f32, 0xb10f3c12, 0x9ead275e, 0xdf488c92, 0x8fd43f33, 0x8ef9e323,
                    0xf37e3a5b, 0x40cde8c0, 0xe909685a, 0x74b321cb, 0xb3beb3ab, 0xad2366e0,
                    0x04582d6c, 0xe0de060b, 0x99a8b9ad, 0x829ea6ad, 0xc011f1f1, 0x23e81c94,
                    0xbe71d1c9, 0xe810dfad, 0x833d9fe7, 0x1330dcf6, 0xd8803771, 0x79c59fb0,
                    0x157d65e6, 0x032a67a2, 0x5bafa59d, 0xb9ab18a1, 0x15e0db63, 0x5642f680,
                    0x4c0c6a0f, 0xd279dc30, 0xd77a4ffd, 0x4ba20ea7, 0xa0055fec, 0x48d23f62,
                    0x257e032d, 0x3748ed3e, 0x57a373a2, 0xda854f39, 0xfa9097b7, 0x6e9b1edd,
                    0x3abac8de, 0xd5c1a65e, 0xe927843b, 0xc3e6a936, 0x88f6bc52, 0x365641d8,
                    0x453685e6, 0xa8bf4c2b, 0x553ed75f, 0xacc23f2b, 0x9299a1bb, 0x925c57d6,
                    0xd2dac954, 0x8db527b3, 0x0f1dfec4, 0xee103470, 0xa4b48730, 0x4145e3ff,
                    0xafa01355, 0x4c48df52, 0x2be26151, 0x93ee44b2, 0x8937aa8d, 0x9251b48e,
                    0x39f60494, 0x695cd599, 0x83cf6b67, 0x39629b23, 0x58d9b39c, 0xc9c92710,
                    0x3d0305ff, 0xd67f9df1, 0x1d48e3e1, 0x3fab70d2, 0xb6df5bc4, 0x4cbb3ac9,
                    0x4d1d2e2a, 0x41f0f1a1, 0x6ca6cf22, 0x404e89f9, 0x50f53719, 0x20af6b18,
                    0xf0ffcce4, 0x1c4676cf, 0x3184ca7f, 0xe5690ad3, 0x95e5bbd5, 0x4aab8e3d,
                    0xb58326d7, 0x0a86ccf3, 0x6477dc65, 0x7e228f06, 0xea6c1d07, 0x270a25c0,
                    0x04f06988, 0xd107d14e, 0x6f1df29c, 0x684efd85, 0x5941c2b0, 0xa972c8af,
                    0x4b33c668, 0x233713c4, 0xffed309c, 0xdb6fa3af, 0x9075f156, 0x2daf2d8c,
                    0x419b6c62, 0xb432d630, 0x46546932, 0x3cbea076, 0x5a992058, 0x121394bd,
                    0x631e5ba5, 0x70a4a711, 0x3c031f6b, 0x1f91217d, 0x0df06e81, 0xa9a1ac22,
                    0x3a60e157, 0xe8f290a0, 0x3ca195b7, 0x592aff27, 0x114514e4, 0x406aaf7b,
                    0x2a7795ed, 0xa891deaf, 0xc2a70ab3, 0x5c803a80, 0x6475965f, 0x104065a8,
                    0xe2ed6561, 0xf84406bb, 0x50e44327, 0x45d6af7b, 0x07df12c9, 0x7ca73396,
                    0x2e9e971e, 0xe52ed416, 0x0bcbcf48, 0x182fcc75, 0xb9a4e82e, 0x0f676182,
                    0x6e0485f3, 0x86b46d47, 0x45bcd6ad, 0xd8329c90, 0x78419d0a, 0x207443ab,
                    0x8a062f1b, 0xe45de5e5, 0x5f459bcd, 0xb767569e, 0xa5d31326, 0x0eccdf1e,
                    0x34f9145c, 0x35d4725d, 0x743a0316, 0xb734473c, 0xdc75850e, 0xcf600fc3,
                    0x1fede499, 0x4c6db5d0, 0x6d3c0b75, 0xd80e1a18, 0x44e09030, 0x2bdb5941,
                    0x3831a44d, 0xb80510ee, 0x0421b0ce, 0x3f6aa279, 0x20e6706f, 0xf3a42d30,
                    0xaac6fbc1, 0xe91ae295, 0xd947db06, 0xd88cba08, 0x79a3ae86, 0x019a9671,
                    0x24706785, 0xa3160ef1, 0x4af231f1, 0xe1122517, 0x1142b7f5, 0x690ce4b6,
                    0x80c90214, 0xa8237762, 0x1b59ab3a, 0x70c23d72, 0x16b50298, 0xf5125794,
                    0xf37bc05c, 0x8f38ed85, 0xf4eb92ac, 0x279a026f, 0x316572c5, 0xd32732a0,
                    0x3dd4a1fa, 0xb8716e96, 0x6789e61b, 0x0e017b4b, 0x8956741c, 0x818f05c4,
                    0x9db16302, 0x6298e2b1, 0xf93eb40a, 0xeeaab443, 0xf9bbb3d0, 0x4667380a,
                    0x047a9f62, 0x91de9993, 0x09225cb7, 0x8933c56f, 0xbe6d02e5, 0x838e7269,
                    0xa2191fe4, 0xc315ba04, 0xb86b3916, 0x2f625a84, 0xaeb09db4, 0x64a075b1,
                    0xafb44975, 0xd8d748d4, 0xd9f1d618, 0x72233b10, 0xa91ab0c0, 0x2d6ab6c8,
                    0xe2e84b8b, 0x6a5e2a17, 0xa6e28d2b, 0x15b05a11, 0x72964ade, 0xbb3a6e54,
                    0x056e75f3, 0xedfb4fa0, 0xaf88fb61, 0x298724ea, 0xc37f5eb9, 0x33a5fc9f,
                    0x94e9b33f, 0x70ff156a, 0x6fde3c83, 0x36576707, 0x5a2fcff0, 0x0f70f584,
                    0xfb83fe7c, 0x1f0e991d, 0x11fd1036, 0x7e5b8264, 0x62029d56, 0xc3f70471,
                    0x53e554bb, 0xe17767ad, 0x4e5f51f8, 0x6cfc6849, 0xc541feca, 0xb487ffac,
                    0xc5bc0b8d, 0x651f2ac1, 0xf3b3c7cb, 0xca9966af, 0x7058380d, 0xc9340f53,
                    0xbc10948a, 0x640c25e8, 0xdb9d159e, 0x37979f23, 0xbc583717, 0x1c82fd9a,
                    0x46412d3e, 0xd87bdcc0, 0xa169ee07, 0xc917586a, 0x521a6066, 0x17cde2b0,
                    0xf60ffd76, 0xa69bebf5, 0x3a380371, 0x0621f6dd, 0xaccd7007, 0xa032e368,
                    0x1ac01895, 0xc06dfb5d, 0xaa2e9671, 0x0c501bbf, 0x53c03188, 0x857d7805,
                    0xbeef3f1f, 0xd368ec03, 0x1d27e4fe, 0xe6a43ec5, 0x9a70d948, 0x104b0177,
                    0x05768087, 0x5bd832ac, 0x0ac4ba9d, 0x62d6e740, 0x32a8c3cd, 0x669c7606,
                    0x486071f9, 0x608bbf5e, 0x8f3c1f4e, 0x2fab1b7d, 0x4d029883, 0x174dc252,
                    0x8c728822, 0x3586749f, 0x1c7ef85c, 0xf4cb6ef0, 0x8b3a195a, 0x8196e480,
                    0x24c7d4db, 0xee79a357, 0x49a71353, 0x670a5e1e, 0x8505cd20, 0xb80e0b6a,
                    0xe7242da5, 0x509860a8, 0x93a40be6, 0xb4218597, 0x4f88e9d6, 0x069793c0,
                    0xa81b28dc, 0x210bb469, 0x82e4ebfd, 0x5dc0cfc2, 0x3aaab6b6, 0xab1786b3,
                    0x6ad8bf1d, 0xc2135b37, 0xe250b55f, 0xd837d456, 0x1b6fb84e, 0xd2c2d0c9,
                    0x77c9ae87, 0x1f33577f, 0xb57b22d9, 0x3ec74b66, 0xabc1edc4, 0xeefb7740,
                    0x3fff9aaa, 0x0c3fbed1, 0xdda2c680, 0x0328cfe0, 0xb7f6548b, 0xb8256eb0,
                    0x45fe7c0f, 0x78f07ce3, 0x284e0abd, 0x91c92193, 0x9c3a45d8, 0x68972815,
                    0x1ec403ea, 0x00ff1cd5, 0xd2d84bf3, 0xbdec18dd, 0xf24ba458, 0x9ccd4a78,
                    0xa98d0e6a, 0x7af942ce, 0x1894c6f3, 0x3b80465f, 0x12bfebf9, 0xdef6008f,
                    0xc21be2ea, 0xa8c0a737, 0xeda098fe, 0xb672d49c, 0xce75842e, 0x8fc140c3,
                    0x215f0270, 0x7f28ebd7, 0x393828c3, 0xc5bb3640, 0xcd1cdf53, 0xe152e050,
                    0x6cc6595f, 0xb5ff8f2d, 0x62afeef1, 0x7facf247, 0x3fc4ada3, 0x48dc80b4,
                    0xa6e53b3a, 0x19704ce8, 0x9e258702, 0x29d6d6af, 0xa76e8ca6, 0x87fcc1c9,
                    0x531566df, 0x01e9894a, 0x015c2604, 0x27d0dc5e, 0xcd6f2d2c, 0x0a675e73,
                    0x4e16ddd8, 0xfb7a9734, 0xf4d90256, 0x8f7ca2a7, 0xff423074, 0x1820297d,
                    0x88ba5114, 0xde2033a0, 0xe11ba361, 0xc27cbfd1, 0x33eba837, 0x69fce797,
                    0x4a2e3528, 0x5dfb3eb4, 0x0ad4da5a, 0x7ebac089, 0xe216d028, 0x4e328010,
                    0x5799917d, 0x8ffcf6b7, 0xcf2a1774, 0x557f6d80, 0xad5a13c0, 0x490a725f,
                    0xcc258aa4, 0x5c114fe0, 0x41e4ed53, 0x36227bad, 0xc4c6a02a, 0x61b9de42,
                    0x1722ed18, 0x11f107ea, 0x81b39b07, 0x631ea379, 0x36b6018d, 0x03a04ff4,
                    0x4f65cc4f, 0x1f536b76, 0x5bcdb421, 0xf587f885, 0x40e1bcb5, 0x425ccd88,
                    0x9b085f7c, 0xfe06eef0,
                ];

                let buffer = generate_kernel(
                    &[
                        0xe1, 0x30, 0x45, 0xe0, 0x5b, 0x22, 0x93, 0x91, 0x72, 0x61, 0x23, 0x19,
                        0x79, 0x29, 0xac, 0x92, 0xd1, 0x52, 0x2a, 0x3c, 0x18, 0xe7, 0x85, 0x27,
                        0x73, 0xbf, 0xe5, 0xab, 0xb3, 0xb5, 0xa1, 0xef,
                    ],
                    1,
                );

                let compute_shader = compile_kernel(buffer).unwrap();
                let res = execute(&compute_shader, mix_lane).unwrap();

                for (idx, r) in res.iter().enumerate() {
                    assert_eq!(*r, expected[idx]);
                }
            }
        }
    }
}
