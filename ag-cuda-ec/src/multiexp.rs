use ag_cuda_proxy::{ActiveWorkspace, DeviceParam, KernelConfig, DeviceData};
use ag_cuda_workspace_macro::auto_workspace;
use ag_types::{GpuName, GpuRepr, PrimeFieldRepr};
use ark_bls12_381::{Fr as Scalar, G1Affine as Affine, G1Projective as Curve};
use ark_ec::{CurveGroup, Group};
use ark_ff::{Field, PrimeField};
use ark_std::{log2, Zero};
use rustacuda::error::CudaResult;
use std::time::Instant;
use std::ops::AddAssign;

use crate::{GLOBAL, LOCAL};

#[auto_workspace]
pub fn multiexp_gpu(
    workspace: &ActiveWorkspace, bases: &[<Affine as GpuRepr>::Repr], exponents: &[<Scalar as PrimeFieldRepr>::Repr],
) -> CudaResult<Affine> {
    const MAX_WINDOW_SIZE: usize = 10;
    let work_units = 128 * 256; // TODO device.work_units
    let num_terms = bases.len();
    let window_size = std::cmp::min(((((num_terms + work_units - 1) / work_units) as f64).log2() as usize) + 2, MAX_WINDOW_SIZE);
    // windows_size * num_windows needs to be >= 256 in order for the kernel to work correctly.
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
        global_work_size: global_work_size,
        local_work_size: physical_local_work_size,
        shared_mem: 0usize,
    };

    let kernel_name = format!("{}_multiexp", Affine::name());

    let now = Instant::now();

    //let mut output_gpu = DeviceParam::new(&mut output)?;

    //dbg!(bases.len(), bucket.len(), output.len(), exponents.len(), num_terms, num_groups, num_windows, window_size);
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
    
    let dur = now.elapsed().as_secs() * 1000
        + now.elapsed().subsec_millis() as u64;
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

#[auto_workspace]
pub fn multiple_multiexp(
    workspace: &ActiveWorkspace, bases: &[<Affine as GpuRepr>::Repr], exponents: &[<Scalar as PrimeFieldRepr>::Repr], num_groups: usize
) -> CudaResult<Vec<Curve>> {
    let window_size = 7usize;
    let num_windows = (256 + window_size - 1) / window_size;
    let work_units = num_windows * num_groups; // TODO device.work_units
    let num_terms = bases.len();
    let bucket_len = (1 << window_size) - 1;
    // let num_windows_log2 = log2(num_windows);
    // assert_eq!(1 << num_windows_log2, num_windows);
    

    let mut output = vec![Curve::zero(); num_groups];
    
    // dbg!(bases.len(), bucket.len(), output.len(), exponents.len(), num_terms, num_groups, num_windows, window_size);

    

    let stream = workspace.stream()?;
    let base_gpu = DeviceData::upload(bases, &stream)?;
    println!("Size: {}", work_units * bucket_len * std::mem::size_of::<Curve>());
    let buckets = DeviceData::uninitialized(work_units * bucket_len * std::mem::size_of::<Curve>())?;
    // output_gpu.to_device(&stream)?;
    
    let kernel = workspace.create_kernel()?;

    let local_work_size = num_windows; // most efficient: 32 - 128
    let global_work_size = work_units / local_work_size;

    let config = KernelConfig {
        global_work_size,
        local_work_size,
        // shared_mem: std::mem::size_of::<Curve>() * physical_local_work_size * bucket_len as usize,
        shared_mem: 0,
    };

    let kernel_name = format!("{}_multiexp", Affine::name());

    let now = Instant::now();

    kernel
        .func(&kernel_name)?
        .dev_data(&base_gpu)?
        .out_slice(&mut output)?
        .in_ref_slice(exponents)?
        .dev_data(&buckets)?
        .val(dbg!(num_terms) as u32)?
        .val(dbg!(num_groups) as u32)?
        .val(dbg!(num_windows) as u32)?
        .val(dbg!(window_size) as u32)?
        // .val(dbg!(num_windows_log2) as u32)?
        .launch(config)?
        .complete()?;
    
    let dur = now.elapsed().as_secs() * 1000
        + now.elapsed().subsec_millis() as u64;
    println!("GPU (inner) took {}ms.", dur);

    for i in output.iter() {
        assert!(!i.is_zero());
    }

    // output_gpu.to_host(&stream)?;

    Ok(output)
}
