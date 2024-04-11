use ag_cuda_proxy::{ActiveWorkspace, DeviceParam, KernelConfig};
use ag_cuda_workspace_macro::auto_workspace;
use ag_types::GpuName;
use ark_bn254::{Fr as Scalar, G1Affine as Affine, G1Projective as Curve};
use ark_ff::Field;
use ark_std::Zero;
use rustacuda::error::CudaResult;
use std::time::Instant;

use crate::{GLOBAL, LOCAL};

#[auto_workspace]
pub fn radix_ec_fft(
    workspace: &ActiveWorkspace, input: &mut Vec<Curve>, omegas: &[Scalar],
) -> CudaResult<()> {
    const MAX_LOG2_RADIX: u32 = 8;

    let n = input.len();
    let log_n = n.ilog2();
    assert_eq!(n, 1 << log_n);

    let mut output = vec![Curve::zero(); n];

    let max_deg = std::cmp::min(MAX_LOG2_RADIX, log_n);

    let twiddle = omegas[0].pow([(n >> max_deg) as u64]);

    let mut input_gpu = DeviceParam::new(input)?;
    let mut output_gpu = DeviceParam::new(&mut output)?;

    let stream = workspace.stream()?;
    input_gpu.to_device(&stream)?;

    let mut kernel = workspace.create_kernel()?;

    // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    let mut log_p = 0u32;
    // Each iteration performs a FFT round
    while log_p < log_n {
        // 1=>radix2, 2=>radix4, 3=>radix8, ...
        let deg = std::cmp::min(max_deg, log_n - log_p);

        let n = 1u32 << log_n;

        let virtual_local_work_size = 1 << (deg - 1);

        // The algorithm may require a small local_network_size. However, too
        // small local_network_size will undermine the performance. So we
        // allocate a larger local_network_size, but translate the global
        // parameter before execution.
        let physical_local_work_size = if virtual_local_work_size >= 32 {
            virtual_local_work_size
        } else if n <= 64 {
            virtual_local_work_size
        } else {
            32
        };
        let global_work_size = n / 2 / physical_local_work_size;

        let config = KernelConfig {
            global_work_size: global_work_size as usize,
            local_work_size: physical_local_work_size as usize,
            shared_mem: std::mem::size_of::<Curve>()
                * 2
                * physical_local_work_size as usize,
        };

        let kernel_name = format!("{}_radix_fft", Affine::name());

        let now = Instant::now();

        kernel = kernel
            .func(&kernel_name)?
            .dev_arg(&input_gpu)?
            .dev_arg(&output_gpu)?
            .in_ref(&twiddle)?
            .in_ref_slice(&omegas[..])?
            .empty()?
            .val(n)?
            .val(log_p)?
            .val(deg)?
            .val(virtual_local_work_size)?
            .val(max_deg)?
            .launch(config)?
            .complete()?;

        let dur = now.elapsed().as_secs() * 1000
            + now.elapsed().subsec_millis() as u64;
        println!("GPU (inner) took {}ms.", dur);

        log_p += deg;
        DeviceParam::swap_device_pointer(&mut input_gpu, &mut output_gpu);
    }

    input_gpu.to_host(&stream)?;

    Ok(())
}
