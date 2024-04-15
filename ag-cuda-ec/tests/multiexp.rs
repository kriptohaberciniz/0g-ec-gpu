use std::{sync::Arc, time::Instant};

use ag_types::{GpuRepr, PrimeFieldRepr};
use ark_bls12_381::{Fr as Scalar, G1Affine as Affine};
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field, UniformRand};
use ark_std::{rand::thread_rng, Zero};

use ag_cuda_ec::*;
use ec_gpu_proxy::{multiexp_cpu::{multiexp_cpu, FullDensity}, threadpool::Worker};
use std::ops::AddAssign;

#[test]
fn test_multiexp_sequential() {
    let mut rng = thread_rng();
    init_global_workspace();

    const MAX_DEGREE: usize = 11;
    let bases = (0..(1 << MAX_DEGREE))
        .map(|_| Affine::rand(&mut rng))
        .collect::<Vec<_>>();
    let exponents = (0..(1 << MAX_DEGREE))
        .map(|_| Scalar::rand(&mut rng).to_repr())
        .collect::<Vec<_>>();
    let bases_gpu: Vec<_> =
        bases.iter().map(GpuRepr::to_gpu_repr).collect();
    //for batch_degree in 0..12usize {
        for degree in 10..=MAX_DEGREE {
            //let batch_size = 1 << batch_degree;
            let n = 1 << degree;

            println!("Testing multiexp for {} elements...", n);

            // Evaluate with CPU
            let pool = Worker::new();
            let v = Arc::new(exponents.clone()[..n].to_vec());
            let g = Arc::new(bases.clone()[..n].to_vec());
            let now = Instant::now();
            let acc_cpu = multiexp_cpu(&pool, (g, 0), FullDensity, v).wait().unwrap();
            let cpu_dur = now.elapsed().as_millis();
            println!("CPU took {}ms.", cpu_dur);

            // Evaluate with GPU
            let now = Instant::now();
            let acc_gpu = multiexp_gpu_st(&bases_gpu[..n], &exponents[..n]).unwrap();
            let gpu_dur = now.elapsed().as_millis();
            println!("GPU took {}ms.", gpu_dur);

            // Evaluate with CPU
            println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

            assert_eq!(acc_gpu, acc_cpu);

            println!("============================");
        }
    //}
}
