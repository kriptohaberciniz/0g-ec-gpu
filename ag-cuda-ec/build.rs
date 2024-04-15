fn main() {
    use ag_build::{generate, SourceBuilder};

    let source = SourceBuilder::new()
        .add_ec_fft::<ark_bls12_381::G1Affine>()
        .add_multiexp::<ark_bls12_381::G1Affine>();
    
    generate(&source);
}
