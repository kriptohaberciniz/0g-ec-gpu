#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::time::Instant;

use ag_build::generate;
use ark_bls12_381::{Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::UniformRand;
use ec_gpu_proxy::{
    ec_fft::EcFftKernel,
    ec_fft_cpu::{parallel_ec_fft, serial_ec_fft},
    threadpool::Worker,
};
use rust_gpu_tools::Device;

fn omega<F: FftField>(num_coeffs: usize) -> F {
    // Compute omega, the 2^exp primitive root of unity
    let exp = (num_coeffs as f32).log2().floor() as u32;
    let mut omega = F::TWO_ADIC_ROOT_OF_UNITY;
    for _ in exp..F::TWO_ADICITY {
        omega = omega.square();
    }
    omega
}

fn build_ec_fft() {
    generate(&ag_build::SourceBuilder::new().add_ec_fft::<G1Affine>())
}

#[test]
pub fn gpu_ec_fft_consistency() {
    fil_logger::maybe_init();
    let mut rng = rand::thread_rng();

    let worker = Worker::new();
    let log_threads = worker.log_num_threads();
    build_ec_fft();
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_program::load_program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = EcFftKernel::<G1Affine>::create(programs)
        .expect("Cannot initialize kernel!");

    for log_d in 1..=16 {
        let d = 1 << log_d;

        let mut v1_coeffs = (0..d)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect::<Vec<_>>();
        let v1_omega = Fr::get_root_of_unity(v1_coeffs.len() as u64).unwrap();
        let mut v2_coeffs = v1_coeffs.clone();

        println!("Testing FFTg for {} elements...", d);

        let mut now = Instant::now();
        kern.radix_ec_fft_many(&mut [&mut v1_coeffs], &[v1_omega], &[log_d])
            .expect("GPU FFTg failed!");
        let gpu_dur = now.elapsed().as_secs() * 1000
            + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        let fft_domain =
            Radix2EvaluationDomain::<Fr>::new(v2_coeffs.len()).unwrap();
        now = Instant::now();
        fft_domain.fft_in_place(&mut v2_coeffs);

        let cpu_dur = now.elapsed().as_secs() * 1000
            + now.elapsed().subsec_millis() as u64;
        println!("CPU ({} cores) took {}ms.", 1 << log_threads, cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(v1_coeffs, v2_coeffs);
        println!("============================");
    }
}

#[test]
pub fn gpu_ec_fft_many_consistency() {
    fil_logger::maybe_init();
    let mut rng = rand::thread_rng();

    let worker = Worker::new();
    let log_threads = worker.log_num_threads();
    let devices = Device::all();
    build_ec_fft();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_program::load_program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = EcFftKernel::<G1Affine>::create(programs)
        .expect("Cannot initialize kernel!");

    for log_d in 1..=16 {
        let d = 1 << log_d;

        let mut v11_coeffs = (0..d)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect::<Vec<_>>();
        let mut v12_coeffs = (0..d)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect::<Vec<_>>();
        let mut v13_coeffs = (0..d)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect::<Vec<_>>();
        let v11_omega = omega::<Fr>(v11_coeffs.len());
        let v12_omega = omega::<Fr>(v12_coeffs.len());
        let v13_omega = omega::<Fr>(v13_coeffs.len());

        let mut v21_coeffs = v11_coeffs.clone();
        let mut v22_coeffs = v12_coeffs.clone();
        let mut v23_coeffs = v13_coeffs.clone();
        let v21_omega = v11_omega;
        let v22_omega = v12_omega;
        let v23_omega = v13_omega;

        println!("Testing FFTg3 for {} elements...", d);

        let mut now = Instant::now();
        kern.radix_ec_fft_many(
            &mut [&mut v11_coeffs, &mut v12_coeffs, &mut v13_coeffs],
            &[v11_omega, v12_omega, v13_omega],
            &[log_d, log_d, log_d],
        )
        .expect("GPU FFTg3 failed!");
        let gpu_dur = now.elapsed().as_secs() * 1000
            + now.elapsed().subsec_millis() as u64;
        println!("GPU FFTg3 took {}ms.", gpu_dur);

        now = Instant::now();
        if log_d <= log_threads {
            serial_ec_fft::<G1Affine>(&mut v21_coeffs, &v21_omega, log_d);
            serial_ec_fft::<G1Affine>(&mut v22_coeffs, &v22_omega, log_d);
            serial_ec_fft::<G1Affine>(&mut v23_coeffs, &v23_omega, log_d);
        } else {
            parallel_ec_fft::<G1Affine>(
                &mut v21_coeffs,
                &worker,
                &v21_omega,
                log_d,
                log_threads,
            );
            parallel_ec_fft::<G1Affine>(
                &mut v22_coeffs,
                &worker,
                &v22_omega,
                log_d,
                log_threads,
            );
            parallel_ec_fft::<G1Affine>(
                &mut v23_coeffs,
                &worker,
                &v23_omega,
                log_d,
                log_threads,
            );
        }
        let cpu_dur = now.elapsed().as_secs() * 1000
            + now.elapsed().subsec_millis() as u64;
        println!("CPU FFTg3 ({} cores) took {}ms.", 1 << log_threads, cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert!(v11_coeffs == v21_coeffs);
        assert!(v12_coeffs == v22_coeffs);
        assert!(v13_coeffs == v23_coeffs);

        println!("============================");
    }
}
