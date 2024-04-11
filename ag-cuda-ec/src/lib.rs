mod ec_fft;

use ag_cuda_proxy::CudaWorkspace;
use ag_cuda_workspace_macro::construct_workspace;

const FATBIN: &'static [u8] =
    include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN"));

construct_workspace!(|| CudaWorkspace::from_bytes(FATBIN).unwrap());

pub use ec_fft::*;
