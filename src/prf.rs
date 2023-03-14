//! Pseudorandom function based on the ProgPoW algorithm.
//! https://eips.ethereum.org/EIPS/eip-1057#specification

use crate::constants::*;
use std::convert::TryInto;

use crate::kiss99::{fnv1a, kiss99, Kiss99State, FNV_OFFSET_BASIS};

/// Pseudorandom function.
pub fn prf(
    block_k_root_hash: &[u8; INPUT_HASH_SIZE],
    witness: &[u8; WITNESS_SIZE],
    loop_count: u16,
) -> [u8; 136] {
    let mut mix_state = [0u32; PROGPOW_REGS * PROGPOW_LANES];

    // initialize PRNG with data from witness
    for lane_id in 0..PROGPOW_LANES {
        fill_mix(
            u64::from_le_bytes(
                witness[(lane_id * 8)..((lane_id + 1) * 8)]
                    .try_into()
                    .unwrap(), // FIXME
            ),
            lane_id as u32,
            &mut mix_state,
        );
    }

    // Generate and compile a program for execution on GPU.
    let prog_source = generate_kernel(0); // TODO: use actual block header as a seed

    let prog = vulkan::compile_kernel(prog_source).unwrap();

    let updated_mix_data = vulkan::execute(&prog, mix_state);

    // Reduce mix data to a per-lane 32-bit digest
    let mut digest_lane = [0u32; PROGPOW_LANES];
    let mut result = Vec::<u8>::with_capacity(136);

    for lane_id in 0..PROGPOW_LANES {
        digest_lane[lane_id] = FNV_OFFSET_BASIS;
        digest_lane[lane_id] = fnv1a(digest_lane[lane_id], updated_mix_data[lane_id]);

        // FIXME: excessive memory copying
        result.extend_from_slice(&digest_lane[lane_id].to_le_bytes());
    }

    // Concat lane digests with the remainder of the witness number
    result.extend_from_slice(&witness[128..200]); // fixme: don't use magical numbers

    result.try_into().unwrap() // fixme: unwrap
}

/// Populates an array of u32 values used by each lane in the hash calculations.
// todo: use const for `mix_state` size
fn fill_mix(seed: u64, lane_id: u32, mix_state: &mut [u32; PROGPOW_REGS * PROGPOW_LANES]) {
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

/// Generates an OpenCL C kernel for PRF.
/// This can be used either in the Vulkan runtime (with the clspv compiler) or in the OpenCL one.
fn generate_kernel(block_num: u64) -> String {
    let mut buffer = String::with_capacity(1024);

    buffer.push_str(&format!(
        "#define PROGPOW_REGS {PROGPOW_REGS}\n\
        typedef unsigned int uint32_t;\n\
        typedef unsigned long uint64_t;\n\
        \
        __kernel void progPowLoop(__global uint32_t mix[PROGPOW_REGS]) {{\n\
            uint32_t data;\n",
    ));

    // Initialize the program seed and sequences.
    let mut mix_seq_dst = [0u32; PROGPOW_REGS];
    let mut mix_seq_src = [0u32; PROGPOW_REGS];

    let mut mix_seq_dst_cnt = 0;

    // initialise RNG seed - progPowInit
    let prog_seed = block_num;

    let mut prog_rnd: Kiss99State = Default::default();
    prog_rnd.z = fnv1a(FNV_OFFSET_BASIS, prog_seed as u32);
    prog_rnd.w = fnv1a(prog_rnd.z, (prog_seed >> 32) as u32);
    prog_rnd.jsr = fnv1a(prog_rnd.w, prog_seed as u32);
    prog_rnd.jcong = fnv1a(prog_rnd.jsr, (prog_seed >> 32) as u32);

    // Create a random sequence of mix destinations for merge() and mix sources for cache reads
    // guarantees every destination merged once
    // guarantees no duplicate cache reads, which could be optimized away
    // Uses Fisher-Yates shuffle
    for i in 0..PROGPOW_REGS {
        mix_seq_dst[i] = i as u32;
        mix_seq_src[i] = i as u32;
    }
    for i in (0..PROGPOW_REGS - 1).rev() {
        {
            let j = kiss99(&mut prog_rnd) % (i as u32 + 1);
            mix_seq_dst.swap(i, j as usize);
        }
        {
            let j = kiss99(&mut prog_rnd) % (i as u32 + 1);
            mix_seq_src.swap(i, j as usize);
        }
    }

    for _i in 0..PROGPOW_CNT_MATH {
        // TODO: add cached memory access instructions which require DAG.
        // DAG can be generated using a pseudorandom algorithm in this case.
        // For now, we can just assume that PROGPOW_CNT_CACHE is set to 0.

        // Random math - generate 2 unique sources
        let src_rnd = kiss99(&mut prog_rnd) as usize % (PROGPOW_REGS * (PROGPOW_REGS - 1));
        let src1 = src_rnd % PROGPOW_REGS; // 0 <= src1 < PROGPOW_REGS
        let mut src2 = src_rnd / PROGPOW_REGS; // 0 <= src2 < PROGPOW_REGS - 1

        if src2 >= src1 {
            // src2 is now any reg other than src1
            src2 += 1;
        }

        let sel1 = kiss99(&mut prog_rnd);

        let dst = mix_seq_dst[mix_seq_dst_cnt % PROGPOW_REGS];
        let sel2 = kiss99(&mut prog_rnd);

        mix_seq_dst_cnt += 1;

        // rand math function
        buffer.push_str("data = ");
        buffer.push_str(&generate_random_math_func(
            &format!("mix[{src1}]"),
            &format!("mix[{src2}]"),
            sel1,
        ));
        buffer.push_str(";\n");

        // merge
        buffer.push_str(&format!("mix[{dst}] = "));
        buffer.push_str(&generate_merge_func(&format!("mix[{dst}]"), "data", sel2));
        buffer.push_str(";\n");
    }

    // TODO: utilize PROGPOW_DAG_LOADS?

    buffer.push_str("}\n");

    buffer
}

/// Merge new data from b into the value in a.
/// Assuming A has high entropy only do ops that retain entropy even if B is low entropy.
/// (i.e. don't do A & B)
fn generate_merge_func(a: &str, b: &str, r: u32) -> String {
    match r % 4 {
        0 => format!("({a} * 33) + {b}"),
        1 => format!("({a} ^ {b}) * 33"),
        // prevent rotate by 0 which is a NOP
        2 => format!(
            "rotate({a}, (uint32_t)({bits})) ^ {b}",
            bits = ((r >> 16) % 31) + 1
        ),
        3 | _ => format!(
            "rotate({a}, (uint32_t)(32 - {bits})) ^ {b}",
            bits = ((r >> 16) % 31) + 1
        ),
    }
}

fn generate_random_math_func(a: &str, b: &str, r: u32) -> String {
    // mix[l][src1], mix[l][src2], sel1)
    match r % 11 {
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
        0 | _ => format!("{a} + {b}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{fill_mix, generate_kernel, prf};
    use crate::constants::*;

    #[test]
    fn generate_kernel_test() {
        let kernel = generate_kernel(123);
        println!("\n===\n{}", kernel);
    }

    #[test]
    fn test_exec() {
        prf(&[0; INPUT_HASH_SIZE], &[0; WITNESS_SIZE], 1);
    }

    #[test]
    fn fill_mix_test_vectors() {
        let hash_seed = 0xEE304846DDD0A47B;

        {
            let lane_id = 0;
            let mut mix_state = [0u32; PROGPOW_REGS * PROGPOW_LANES];
            fill_mix(hash_seed, lane_id, &mut mix_state);

            assert_eq!(mix_state[0], 0x10c02f0d);
            assert_eq!(mix_state[3], 0x43f0394d);
            assert_eq!(mix_state[5], 0xc4e89d4c);
        }

        {
            let lane_id = 13;
            let mut mix_state = [0u32; PROGPOW_REGS * PROGPOW_LANES];
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
    use crate::constants::*;
    use std::io;
    use vulkano::{
        buffer::{Buffer, BufferAllocateInfo, BufferUsage},
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
    pub fn execute(compute_shader: &[u32], mix: [u32; PROGPOW_REGS * PROGPOW_LANES]) -> Vec<u32> {
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
                shader.entry_point("progPowLoop").unwrap(),
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
                mix,
            )
            .unwrap()
        };

        // Create a descriptor set for the buffer.
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
            .dispatch([1024, 1, 1])
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
        let data_buffer_content = input_buffer.read().unwrap();

        data_buffer_content.to_vec()
    }
}
