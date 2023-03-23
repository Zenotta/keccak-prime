//! Pseudorandom function based on the ProgPoW algorithm.
//! https://eips.ethereum.org/EIPS/eip-1057#specification

use crate::{constants::*, prng::PrngState};
use std::convert::TryInto;

use crate::kiss99::{fnv1a, kiss99, Kiss99State, FNV_OFFSET_BASIS};

const MERGE_MAX_R: u16 = 4;
const MATH_MAX_R: u16 = 10;

fn debug_mix_data(mix_data: &[u32; PROGPOW_REGS * PROGPOW_LANES]) -> String {
    mix_data
        .iter()
        .map(|num| hex::encode(&num.to_be_bytes()))
        .fold(String::new(), |a, b| a + &b + " ")
}

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
) -> [u8; 136] {
    let mut mix_state = [0u32; PROGPOW_REGS * PROGPOW_LANES];

    // initialize PRNG with data from witness
    for lane_id in 0..PROGPOW_LANES {
        let seed = u64::from_be_bytes(
            witness[(lane_id * 8)..((lane_id + 1) * 8)]
                .try_into()
                .unwrap(), // FIXME
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

        vulkan::execute(&prog, mix_state)
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
    result.extend_from_slice(&witness[128..200]); // fixme: don't use magical numbers

    result.try_into().unwrap() // fixme: unwrap
}

/// Returns a digest of mix_state.
fn reduce_mix(mix_state: &[u32; PROGPOW_REGS]) -> u32 {
    let mut digest_lane = FNV_OFFSET_BASIS;

    for lane_id in 0..PROGPOW_REGS {
        digest_lane = fnv1a(digest_lane, mix_state[lane_id]);
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
        buffer.push_str(&generate_progpow_loop_iter(&block_k_root_hash));
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
        /*
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

            let res = prf(&[0; INPUT_HASH_SIZE], &[0; WITNESS_SIZE], 0);

            assert_eq!(res, expected);
        }
         */
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

            let res = prf(&k, &witness, 2);

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
        buffer::{Buffer, BufferAllocateInfo, BufferContents, BufferUsage},
        command_buffer::{
            allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
        },
        device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions,
            QueueCreateInfo, QueueFlags,
        },
        instance::{Instance, InstanceCreateInfo},
        memory::allocator::StandardMemoryAllocator,
        pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
        shader::ShaderModule,
        sync::{self, GpuFuture},
        VulkanLibrary,
    };

    #[cfg(feature = "prf_vulkan_build_clspv")]
    pub fn compile_kernel(kern: String) -> io::Result<Vec<u32>> {
        let output = clspv_sys::compile_from_source(&kern, Default::default());
        dbg!(output.ret_code, output.log);
        Ok(output.output)
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
        dbg!(&output.stderr);

        Ok(output
            .stdout
            .chunks_exact(4)
            .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
            .collect())
    }

    /// ## Inputs
    /// - `compute_shader` - SPIR-V binary of a compiled program.
    pub fn execute<T: BufferContents + Sized + Clone>(compute_shader: &[u32], input_data: T) -> T {
        let library = VulkanLibrary::new().unwrap();

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enumerate_portability: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Choose which physical device to use
        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            khr_variable_pointers: true,
            ..DeviceExtensions::empty()
        };

        // Get a device and a compute queue.
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
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
            .unwrap();

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
        )
        .unwrap();

        // Retrieve a single compute queue.
        let queue = queues.next().unwrap();

        let pipeline = {
            let shader =
                unsafe { ShaderModule::from_words(device.clone(), compute_shader).unwrap() };

            ComputePipeline::new(
                device.clone(),
                shader.entry_point("ProgPoW").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
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
            )
            .unwrap()
        };
        // dbg!(input_buffer.size());

        // Create a descriptor set for the buffer.
        // FIXME: unwrap panics if loop_count == 0
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, input_buffer.clone())],
        )
        .unwrap();

        // Build a command buffer.
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .dispatch([1, 1, 1]) // FIXME: correct number of groups
            .unwrap();

        let command_buffer = builder.build().unwrap();

        // Execute the command buffer.
        let future = sync::now(device)
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // Block execution until the GPU has finished the operation.
        let timeout = None;
        future.wait(timeout).unwrap();

        // Retrieve the buffer contents - it should now contain the mixed data.
        let output_buffer_content = input_buffer.read().unwrap();
        (*output_buffer_content).clone()
    }

    // Vulkan-specific tests
    #[cfg(test)]
    mod tests {
        use crate::{
            constants::{INPUT_HASH_SIZE, PROGPOW_LANES, PROGPOW_REGS},
            prf::{
                generate_kernel, generate_merge_func, generate_progpow_loop_iter,
                generate_random_math_func,
            },
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
            let res = execute(&compute_shader, inputs);

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
            let res = execute(&compute_shader, inputs);

            for r in res {
                println!("{}", hex::encode(r.to_be_bytes()));
            }

            for (idx, x) in res.iter().skip(1).step_by(2).enumerate() {
                assert_eq!(expected[idx], *x);
            }
        }

        #[test]
        fn test_progpow() {
            // Generate a kernel for a single ProgPoW loop
            let mix_lane = [0u32; PROGPOW_REGS * PROGPOW_LANES];
            let expected = [
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
                0x00000000,
            ];

            let buffer = generate_kernel(&[0; INPUT_HASH_SIZE], 1);
            println!("{}", buffer);

            let compute_shader = compile_kernel(buffer).unwrap();
            let res = execute(&compute_shader, mix_lane);

            for (idx, r) in res.iter().enumerate() {
                assert_eq!(*r, expected[idx]);
            }
        }
    }
}
