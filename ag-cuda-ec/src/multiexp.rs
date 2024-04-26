use crate::pairing_suite::{Affine, Curve, Scalar};
use ag_cuda_proxy::{ActiveWorkspace, DeviceData, KernelConfig};
use ag_cuda_workspace_macro::auto_workspace;
use ag_types::{GpuName, GpuRepr, PrimeFieldRepr};
use ark_std::Zero;
use rustacuda::error::CudaResult;
use std::time::Instant;

use crate::{GLOBAL, LOCAL};

#[auto_workspace]
pub fn multiple_multiexp(
    workspace: &ActiveWorkspace, bases: &[<Affine as GpuRepr>::Repr],
    exponents: &[<Scalar as PrimeFieldRepr>::Repr], num_chunks: usize,
    window_size: usize, neg_is_cheap: bool,
) -> CudaResult<Vec<Curve>> {
    let num_windows = (256 + window_size - 1) / window_size;
    let num_lines = bases.len() / exponents.len();
    let work_units = num_windows * num_chunks * num_lines;
    let input_len = exponents.len();

    let bucket_len = if neg_is_cheap {
        1 << (window_size - 1)
    } else {
        (1 << window_size) - 1
    };

    let mut output = vec![Curve::zero(); num_chunks * num_lines];

    let stream = workspace.stream()?;
    let base_gpu = DeviceData::upload(bases, &stream)?;

    let buckets = DeviceData::uninitialized(
        work_units * bucket_len * std::mem::size_of::<Curve>(),
    )?;

    let kernel = workspace.create_kernel()?;

    let local_work_size = num_windows; // most efficient: 32 - 128
    let global_work_size = work_units / local_work_size;

    let config = KernelConfig {
        global_work_size,
        local_work_size,
        shared_mem: 0,
    };

    let kernel_name = format!("{}_multiexp", Affine::name());

    let now = Instant::now();

    kernel
        .func(&kernel_name)?
        .dev_data(&base_gpu)?
        .out_slice(&mut output)?
        .in_ref_slice(&exponents)?
        .dev_data(&buckets)?
        .val(input_len as u32)?
        .val(num_lines as u32)?
        .val(num_chunks as u32)?
        .val(num_windows as u32)?
        .val(window_size as u32)?
        .val(neg_is_cheap)?
        .launch(config)?
        .complete()?;

    let dur =
        now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
    println!("GPU (inner) took {}ms.", dur);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::pairing_suite::{Curve, Scalar};
    use ag_types::{GpuRepr, PrimeFieldRepr};
    use ark_ec::VariableBaseMSM;
    use ark_std::rand::thread_rng;

    use super::*;
    use crate::test_tools::random_input;

    #[test]
    fn test_multiexp_batch() {
        let mut rng = thread_rng();

        const CHUNK_SIZE: usize = 64;
        const CHUNK_NUM: usize = 32;
        const LINES: usize = 2;
        const INPUT_LEN: usize = CHUNK_SIZE * CHUNK_NUM;

        let bases = random_input(INPUT_LEN * LINES, &mut rng);
        let exponents = random_input::<Scalar, _>(INPUT_LEN, &mut rng);

        let bases_gpu: Vec<_> =
            bases.iter().map(GpuRepr::to_gpu_repr).collect();
        let exponents_repr: Vec<_> =
            exponents.iter().map(|x| x.to_repr()).collect();

        let cpu_output: Vec<_> = bases
            .chunks(CHUNK_SIZE)
            .zip(exponents_repr.chunks(CHUNK_SIZE).cycle())
            .map(|(bs, er)| Curve::msm_bigint(bs, er))
            .collect();

        for window_size in 1..=9 {
            let gpu_output: Vec<_> = multiple_multiexp_mt(
                &bases_gpu,
                &exponents_repr,
                CHUNK_NUM,
                window_size,
                true,
            )
            .unwrap();

            assert_eq!(gpu_output.len(), cpu_output.len());

            if gpu_output != cpu_output {
                panic!("Result inconsistent");
            }

            let gpu_output: Vec<_> = multiple_multiexp_mt(
                &bases_gpu,
                &exponents_repr,
                CHUNK_NUM,
                window_size,
                false,
            )
            .unwrap();

            if gpu_output != cpu_output {
                panic!("Result inconsistent");
            }
        }
    }
}

#[cfg(feature = "never")]
#[auto_workspace]
pub fn multiexp_gpu(
    workspace: &ActiveWorkspace, bases: &[<Affine as GpuRepr>::Repr],
    exponents: &[<Scalar as PrimeFieldRepr>::Repr],
) -> CudaResult<Affine> {
    const MAX_WINDOW_SIZE: usize = 10;
    let work_units = 128 * 256; // TODO device.work_units
    let num_terms = bases.len();
    let window_size = std::cmp::min(
        ((((num_terms + work_units - 1) / work_units) as f64).log2() as usize)
            + 2,
        MAX_WINDOW_SIZE,
    );
    // windows_size * num_windows needs to be >= 256 in order for the kernel to
    // work correctly.
    let num_windows = (256 + window_size - 1) / window_size;
    let num_groups = work_units / num_windows;
    let bucket_len = 1 << window_size;

    let mut bucket = vec![Curve::zero(); work_units * bucket_len];
    let mut output = vec![Curve::zero(); work_units];

    let mut bucket_gpu = DeviceParam::new(&mut bucket)?;
    let mut output_gpu = DeviceParam::new(&mut output)?;

    let stream = workspace.stream()?;
    bucket_gpu.to_device(&stream)?;
    output_gpu.to_device(&stream)?;

    let mut kernel = workspace.create_kernel()?;

    let physical_local_work_size = 32usize;
    let global_work_size = work_units / physical_local_work_size;

    let config = KernelConfig {
        global_work_size,
        local_work_size: physical_local_work_size,
        shared_mem: 0usize,
    };

    let kernel_name = format!("{}_multiexp", Affine::name());

    let now = Instant::now();

    //let mut output_gpu = DeviceParam::new(&mut output)?;

    //dbg!(bases.len(), bucket.len(), output.len(), exponents.len(), num_terms,
    // num_groups, num_windows, window_size);
    kernel = kernel
        .func(&kernel_name)?
        .in_ref_slice(bases)?
        //.in_mut_slice(&mut bucket[..])?
        .dev_arg(&bucket_gpu)?
        .dev_arg(&output_gpu)?
        .in_ref_slice(exponents)?
        .val(num_terms as u32)?
        .val(num_groups as u32)?
        .val(num_windows as u32)?
        .val(window_size as u32)?
        .launch(config)?
        .complete()?;

    let dur =
        now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
    println!("GPU (inner) took {}ms.", dur);

    output_gpu.to_host(&stream)?;

    let mut acc = Curve::zero();
    let mut bits = 0;
    let exp_bits = std::mem::size_of::<<Scalar as PrimeFieldRepr>::Repr>() * 8;
    for i in 0..num_windows {
        let w = std::cmp::min(window_size, exp_bits - bits);
        for _ in 0..w {
            acc = acc.double();
        }
        for g in 0..num_groups {
            acc.add_assign(&output[g * num_windows + i]);
        }
        bits += w; // Process the next window
    }

    Ok(acc.into_affine())
}
