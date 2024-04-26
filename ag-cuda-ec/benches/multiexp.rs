use ag_cuda_ec::{
    init_global_workspace,
    multiexp::*,
    pairing_suite::{Curve, Scalar},
    test_tools::random_input_by_cycle,
};
use ag_types::{GpuRepr, PrimeFieldRepr};
use ark_ec::VariableBaseMSM;
use ark_std::rand::thread_rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::time::Instant;

fn main() { bench_multiexp(); }

fn bench_multiexp() {
    let mut rng = thread_rng();
    init_global_workspace();

    const MAX_DEGREE: usize = 10;
    const BATCH_NUM: usize = 12;

    const INPUT_LEN: usize = 1 << (MAX_DEGREE + BATCH_NUM);

    let bases = random_input_by_cycle(INPUT_LEN, 99, &mut rng);
    let exponents_scalar =
        random_input_by_cycle::<Scalar, _>(INPUT_LEN, 73, &mut rng);

    let now = Instant::now();
    let exponents: Vec<_> =
        exponents_scalar.par_iter().map(Scalar::to_repr).collect();
    let unmont_dur = now.elapsed().as_millis();
    println!(
        "unmont_dur CPU took {}ms for {} scalars.",
        unmont_dur,
        exponents.len()
    ); // 10ms for 1M, 0ms for 2048

    let bases_gpu: Vec<_> = bases.iter().map(GpuRepr::to_gpu_repr).collect();
    let n = 1 << MAX_DEGREE;

    println!("Testing multiexp for {} elements...", n);

    // Evaluate with CPU arkworks
    let now = Instant::now();
    let acc_arkworks: Vec<_> = bases
        .chunks(1 << MAX_DEGREE)
        .zip(exponents.chunks(1 << MAX_DEGREE))
        .map(|(bs, es)| Curve::msm_bigint(bs, es))
        .collect();
    let cpu_dur = now.elapsed().as_millis();
    println!("arkworks CPU took {}ms.", cpu_dur);

    // Evaluate with GPU
    let now = Instant::now();
    let acc_gpu: Vec<_> =
        multiple_multiexp_st(&bases_gpu, &exponents, 1024, 8, false).unwrap();
    let gpu_dur = now.elapsed().as_millis();
    println!("GPU took {}ms.", gpu_dur);

    // Evaluate with CPU
    println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

    let gpu_output: Curve = acc_gpu.iter().cloned().sum();
    let cpu_output: Curve = acc_arkworks.iter().cloned().sum();
    if gpu_output != cpu_output {
        panic!("Result inconsistent");
    }

    println!("============================");
}

#[cfg(feature = "never")]
fn bench_multiexp_old() {
    use ec_gpu_proxy::{
        multiexp_cpu::{multiexp_cpu, FullDensity},
        threadpool::Worker,
    };

    let mut rng = thread_rng();
    init_global_workspace();

    const MAX_DEGREE: usize = 11;
    let bases = (0..(1 << MAX_DEGREE))
        .map(|_| Affine::rand(&mut rng))
        .collect::<Vec<_>>();
    let exponents_scalar = (0..(1 << MAX_DEGREE))
        .map(|_| Scalar::rand(&mut rng))
        .collect::<Vec<_>>();
    let now = Instant::now();
    let exponents: Vec<_> =
        exponents_scalar.par_iter().map(|x| x.to_repr()).collect();
    let unmont_dur = now.elapsed().as_millis();
    println!(
        "unmont_dur CPU took {}ms for {} scalars.",
        unmont_dur,
        exponents.len()
    ); // 10ms for 1M, 0ms for 2048
    let bases_gpu: Vec<_> = bases.iter().map(GpuRepr::to_gpu_repr).collect();
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
        let acc_cpu =
            multiexp_cpu(&pool, (g, 0), FullDensity, v).wait().unwrap();
        let cpu_dur = now.elapsed().as_millis();
        println!("ec-gpu CPU took {}ms.", cpu_dur);

        // Evaluate with CPU arkworks
        let now = Instant::now();
        let acc_arkworks = Curve::msm_bigint(&bases[..n], &exponents[..n]);
        let cpu_dur = now.elapsed().as_millis();
        println!("arkworks CPU took {}ms.", cpu_dur);

        // Evaluate with GPU
        let now = Instant::now();
        let acc_gpu =
            multiexp_gpu_st(&bases_gpu[..n], &exponents[..n]).unwrap();
        let gpu_dur = now.elapsed().as_millis();
        println!("GPU took {}ms.", gpu_dur);

        // Evaluate with CPU
        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(acc_gpu, acc_arkworks.into_affine());
        assert_eq!(acc_gpu, acc_cpu);

        println!("============================");
    }
    //}
}
