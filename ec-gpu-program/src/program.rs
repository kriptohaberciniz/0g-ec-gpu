use super::*;

#[macro_export]
/// Helper macro to create a program for a device.
///
/// It will embed the CUDA fatbin/OpenCL source code within your binary. The source needs to be
/// generated via [`crate::source::generate`] in your `build.rs`.
///
/// It returns a `[crate::rust_gpu_tools::Program`] instance.
macro_rules! program {
    ($device:ident) => {{
        use ec_gpu_program::*;

        match check_framework($device) {
            #[cfg(feature = "cuda")]
            Ok(Framework::Cuda) => {
                build_cuda_program($device, include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN")))
            }
            #[cfg(feature = "opencl")]
            Ok(Framework::Opencl) => {
                build_opencl_program($device, include_str!(env!("_EC_GPU_OPENCL_KERNEL_SOURCE")))
            }
            Err(e) => Err(e),
        }
    }};
}

#[cfg(feature = "test-tools")]
#[macro_export]
macro_rules! load_program {
    ($device:ident) => {{
        use ec_gpu_program::*;
        use std::io::Read;

        match check_framework($device) {
            #[cfg(feature = "cuda")]
            Ok(Framework::Cuda) => {
                let mut file =
                    std::fs::File::open(std::env::var("_EC_GPU_CUDA_KERNEL_FATBIN").unwrap())
                        .unwrap();
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer).unwrap();
                build_cuda_program($device, &buffer)
            }
            #[cfg(feature = "opencl")]
            Ok(Framework::Opencl) => {
                let mut buffer =
                    std::fs::read_to_string(std::env::var("_EC_GPU_OPENCL_KERNEL_SOURCE").unwrap())
                        .unwrap();
                build_opencl_program($device, &buffer)
            }
            Err(e) => Err(e),
        }
    }};
}

pub use rust_gpu_tools::{Device, Framework};

pub fn check_framework(device: &Device) -> EcResult<Framework> {
    // Selects a CUDA or OpenCL on the `EC_GPU_FRAMEWORK` environment variable and the
    // compile-time features.
    //
    // You cannot select CUDA if the library was compiled without support for it.
    let framework = match ::std::env::var("EC_GPU_FRAMEWORK") {
        Ok(env) => match env.as_ref() {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Framework::Cuda
                }

                #[cfg(not(feature = "cuda"))]
                return Err(EcError::Simple("CUDA framework is not supported, please compile with the `cuda` feature enabled."));
            }
            "opencl" => {
                #[cfg(feature = "opencl")]
                {
                    Framework::Opencl
                }

                #[cfg(not(feature = "opencl"))]
                return Err(EcError::Simple("OpenCL framework is not supported, please compile with the `opencl` feature enabled."));
            }
            _ => device.framework(),
        },
        Err(_) => device.framework(),
    };
    Ok(framework)
}

#[cfg(feature = "cuda")]
pub fn build_cuda_program(device: &Device, kernel: &[u8]) -> EcResult<rust_gpu_tools::Program> {
    use rust_gpu_tools::{cuda::Program, GPUError};

    let cuda_device = device.cuda_device().ok_or(GPUError::DeviceNotFound)?;
    let program = Program::from_bytes(cuda_device, kernel)?;
    Ok(rust_gpu_tools::Program::Cuda(program))
}

#[cfg(feature = "opencl")]
pub fn build_opencl_program(device: &Device, source: &str) -> EcResult<rust_gpu_tools::Program> {
    use rust_gpu_tools::{opencl::Program, GPUError};

    let opencl_device = device.opencl_device().ok_or(GPUError::DeviceNotFound)?;
    let program = Program::from_opencl(opencl_device, source)?;
    Ok(rust_gpu_tools::Program::Opencl(program))
}