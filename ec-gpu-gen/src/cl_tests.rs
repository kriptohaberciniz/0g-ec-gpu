use crate::pow_vartime;
#[cfg(feature = "cuda")]
use crate::source::CUDA_PROGRAM;
use crate::source::{call_kernel, GpuScalar};

use ark_ff::UniformRand;
use ark_ff::{Field, PrimeField};
use chosen_ark_suite::Fr as Scalar;
use chosen_ark_suite::G1Projective as Curve;
use ec_gpu::PrimeFieldRepr;
use rand::{thread_rng, Rng};

macro_rules! impl_kernel_wrapper {
    ($name: ident) => {
        #[cfg(feature = "cuda")]
        impl rust_gpu_tools::cuda::KernelArgument for $name {
            fn as_c_void(&self) -> *mut std::ffi::c_void {
                &self.0 as *const _ as _
            }
        }

        #[cfg(feature = "opencl")]
        impl rust_gpu_tools::opencl::KernelArgument for $name {
            fn push(&self, kernel: &mut rust_gpu_tools::opencl::Kernel) {
                unsafe { kernel.builder.set_arg(&self.0) };
            }
        }
    };
}

#[repr(transparent)]
struct GpuCurve(Curve);
impl_kernel_wrapper!(GpuCurve);

#[repr(transparent)]
struct GpuBigInt(<Scalar as PrimeField>::BigInt);
impl_kernel_wrapper!(GpuBigInt);

// #[repr(transparent)]
// struct GpuScalar(Scalar);
// impl_kernel_wrapper!(GpuScalar);

#[test]
fn test_ec() {
    use rust_gpu_tools::{program_closures, GPUError};
    let mut rng = thread_rng();
    for _ in 0..100 {
        let a = Curve::rand(&mut rng);
        let b = Scalar::rand(&mut rng);
        let target = a * b;
        let closures = program_closures!(|program, _args| -> Result<Curve, GPUError> {
            let mut cpu_buffer = vec![Curve::default()];

            let buffer = program.create_buffer_from_slice(&cpu_buffer).unwrap();

            let kernel = program.create_kernel("test_ec", 1, 512).unwrap();
            kernel
                .arg(&GpuCurve(a))
                .arg(&GpuScalar(b))
                .arg(&buffer)
                .run()
                .unwrap();

            program.read_into_buffer(&buffer, &mut cpu_buffer).unwrap();
            Ok(cpu_buffer[0])
        });

        let answer = CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();
        assert_eq!(answer, target);
    }
}

#[test]
fn test_add() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = Scalar::rand(&mut rng);
        let c = a + b;

        assert_eq!(
            call_kernel("test_add", &[GpuScalar(a), GpuScalar(b)], &[]),
            c
        );
    }
}

#[test]
fn test_sub() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = Scalar::rand(&mut rng);
        let c = a - b;
        assert_eq!(
            call_kernel("test_sub", &[GpuScalar(a), GpuScalar(b)], &[]),
            c
        );
    }
}

#[test]
fn test_mul() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = Scalar::rand(&mut rng);
        let c = a * b;

        assert_eq!(
            call_kernel("test_mul", &[GpuScalar(a), GpuScalar(b)], &[]),
            c
        );
    }
}

#[test]
fn test_pow() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = rng.gen::<u32>();
        let c = pow_vartime(&a, [b as u64]);
        assert_eq!(call_kernel("test_pow", &[GpuScalar(a)], &[b]), c);
    }
}

#[test]
fn test_sqr() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = a.square();

        assert_eq!(call_kernel("test_sqr", &[GpuScalar(a)], &[]), b);
    }
}

#[test]
fn test_double() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = a.double();

        assert_eq!(call_kernel("test_double", &[GpuScalar(a)], &[]), b);
    }
}

#[test]
fn test_unmont() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b: Scalar = unsafe { std::mem::transmute(a.to_repr()) };
        assert_eq!(call_kernel("test_unmont", &[GpuScalar(a)], &[]), b);
    }
}

#[test]
fn test_mont() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a_repr = Scalar::rand(&mut rng).to_repr();
        let a: Scalar = unsafe { std::mem::transmute(a_repr) };
        let b = Scalar::from_repr(a_repr).unwrap();
        assert_eq!(call_kernel("test_mont", &[GpuScalar(a)], &[]), b);
    }
}
