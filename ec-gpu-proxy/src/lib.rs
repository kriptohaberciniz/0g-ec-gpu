#[cfg(test)]
extern crate ark_bls12_381 as chosen_ark_suite;
//extern crate ark_bls12_381 as chosen_ark_suite;

/// Fast Fourier Transform on the GPU.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub mod fft;
/// Fast Fourier Transform on the CPU.
pub mod fft_cpu;

/// Fast Fourier Transform for G1 on the GPU.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub mod ec_fft;
/// Fast Fourier Transform for G1 on the CPU.
pub mod ec_fft_cpu;

/// Multiexponentiation on the GPU.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub mod multiexp;
/// Multiexponentiation on the CPU.
pub mod multiexp_cpu;

/// Helpers for multithreaded code.
pub mod threadpool;

fn pow_vartime<F: ark_ff::Field, S: AsRef<[u64]>>(base: &F, exp: S) -> F {
    let mut res = F::ONE;
    for e in exp.as_ref().iter().rev() {
        for i in (0..64).rev() {
            res = res.square();

            if ((*e >> i) & 1) == 1 {
                res.mul_assign(base);
            }
        }
    }

    res
}
