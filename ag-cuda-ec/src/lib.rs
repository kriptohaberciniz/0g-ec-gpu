use ag_cuda_proxy::{DeviceParam, KernelConfig, PreparedModule};
use ag_types::GpuName;
use ark_bn254::{Fr as Scalar, G1Affine as Affine, G1Projective as Curve};
use ark_ff::Field;
use ark_std::Zero;
use rustacuda::error::CudaResult;
use std::time::Instant;

use once_cell::sync::Lazy;
use std::sync::Arc;

pub const PROGRAM: &'static [u8] =
    include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN"));

pub static MODULE: Lazy<PreparedModule> = Lazy::new(|| {
    use rustacuda::{
        context::{Context, ContextFlags, ContextStack},
        device::Device,
        init, CudaFlags,
    };

    // Initialize the CUDA API
    init(CudaFlags::empty()).unwrap();

    // Get the first device
    let device = Device::get_device(0).unwrap();

    // Create a context associated to this device
    let ctx = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
        device,
    )
    .unwrap();
    std::mem::forget(ctx);

    let context = Arc::new(ContextStack::pop().unwrap());

    PreparedModule::from_bytes(&context, PROGRAM).unwrap()
});

pub fn radix_ec_fft(
    input: &mut Vec<Curve>, omegas: &[Scalar],
) -> CudaResult<()> {
    const MAX_LOG2_RADIX: u32 = 8;

    let n = input.len();
    let log_n = n.ilog2();
    assert_eq!(n, 1 << log_n);


    let mut output = vec![Curve::zero(); n];

    let max_deg = std::cmp::min(MAX_LOG2_RADIX, log_n);

    let twiddle = omegas[0].pow([(n >> max_deg) as u64]);
    
    let kernel_name = format!("{}_radix_fft", Affine::name());
    let mut kernel = MODULE.create_kernel(&kernel_name)?;

    let mut input_gpu = DeviceParam::new(input)?;
    let mut output_gpu = DeviceParam::new(&mut output)?;

    kernel.sync_to_device(&mut input_gpu)?;

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

        let now = Instant::now();

        kernel = kernel
            .with_args()
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
            .run(config)?;

        let dur = now.elapsed().as_secs() * 1000
            + now.elapsed().subsec_millis() as u64;
        println!("GPU (inner) took {}ms.", dur);

        log_p += deg;
        DeviceParam::swap_device_pointer(&mut input_gpu, &mut output_gpu);
    }

    kernel.sync_to_host(&mut input_gpu)?;
   
    Ok(())
}
