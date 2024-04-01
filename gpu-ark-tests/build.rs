#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn main() {}

#[cfg(any(feature = "cuda", feature = "opencl"))]
fn main() {
    use ark_bn254 as chosen_ark_suite;
    use ec_gpu_gen::SourceBuilder;

    let source_builder = SourceBuilder::new()
        .add_fft::<chosen_ark_suite::Fr>()
        .add_multiexp::<chosen_ark_suite::G1Affine, chosen_ark_suite::Fq>()
        .add_multiexp::<chosen_ark_suite::G2Affine, chosen_ark_suite::Fq2>();
    ec_gpu_gen::generate(&source_builder);
}
