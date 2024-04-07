use std::time::Instant;

use ark_bn254::{Fr as Scalar, G1Affine as Affine};
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field, UniformRand};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{rand::thread_rng, Zero};

use ag_cuda_ec::radix_ec_fft;

#[test]
fn test_ec_fft() {
    let mut rng = thread_rng();

    for degree in 1..20usize {
        let n = 1 << degree;

        println!("Testing FFTg for {} elements...", n);

        let mut omegas = vec![Scalar::zero(); 32];
        omegas[0] = Scalar::get_root_of_unity(n).unwrap();
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }

        let mut v1_coeffs = (0..n)
            .map(|_| Affine::rand(&mut rng).into_group())
            .collect::<Vec<_>>();
        let mut v2_coeffs = v1_coeffs.clone();

        // Evaluate with GPU
        let now = Instant::now();
        radix_ec_fft(&mut v1_coeffs, &omegas[..]).unwrap();
        let gpu_dur = now.elapsed().as_millis();
        println!("GPU took {}ms.", gpu_dur);

        // Evaluate with CPU
        let fft_domain =
            Radix2EvaluationDomain::<Scalar>::new(v2_coeffs.len()).unwrap();
        let now = Instant::now();
        fft_domain.fft_in_place(&mut v2_coeffs);
        let cpu_dur = now.elapsed().as_millis();
        println!("CPU took {}ms.", cpu_dur);

        // Evaluate with CPU
        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(v1_coeffs, v2_coeffs);

        println!("============================");
    }
}
