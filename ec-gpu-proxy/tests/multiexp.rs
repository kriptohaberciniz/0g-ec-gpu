#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::{sync::Arc, time::Instant};

use ag_build::{self, generate};
use ag_types::{GpuCurveAffine, PrimeFieldRepr};
use ark_bn254::{Fr, G1Affine};
use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use ec_gpu_program::EcError;
use ec_gpu_proxy::{
    multiexp::MultiexpKernel,
    multiexp_cpu::{multiexp_cpu, FullDensity, QueryDensity, SourceBuilder},
    threadpool::Worker,
};
use rust_gpu_tools::Device;

fn multiexp_gpu<Q, D, G, S>(
    pool: &Worker, bases: S, density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeFieldRepr>::Repr>>,
    kern: &mut MultiexpKernel<G>,
) -> Result<G::Curve, EcError>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: GpuCurveAffine,
    S: SourceBuilder<G>,
{
    let exps = density_map.as_ref().generate_exps::<G::Scalar>(exponents);
    let (bss, skip) = bases.get();
    kern.multiexp(pool, bss, exps, skip).map_err(Into::into)
}

fn build_multiexp() {
    generate(&ag_build::SourceBuilder::new().add_multiexp::<G1Affine>())
}

#[test]
fn gpu_multiexp_consistency() {
    fil_logger::maybe_init();
    const MAX_LOG_D: usize = 20;
    const START_LOG_D: usize = 10;
    let devices = Device::all();
    build_multiexp();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_program::load_program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = MultiexpKernel::<G1Affine>::create(programs, &devices)
        .expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();

    let mut bases = (0..(1 << START_LOG_D))
        .map(|_| G1Affine::rand(&mut rng))
        .collect::<Vec<_>>();
    //println!("bases[0] = {:?}", bases[0]);

    for log_d in START_LOG_D..=MAX_LOG_D {
        let g = Arc::new(bases.clone());

        let samples = 1 << log_d;
        println!("Testing Multiexp for {} elements...", samples);

        let v = Arc::new(
            (0..samples)
                .map(|_| Fr::rand(&mut rng).to_repr())
                .collect::<Vec<_>>(),
        );
        //println!("v[0] = {:?}", v[0]);

        let mut now = Instant::now();
        let gpu = multiexp_gpu(
            &pool,
            (g.clone(), 0),
            FullDensity,
            v.clone(),
            &mut kern,
        )
        .unwrap();

        // 如果没有错误，继续使用gpu值进行后续操作
        let gpu_dur = now.elapsed().as_secs() * 1000
            + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        let cpu = multiexp_cpu(&pool, (g.clone(), 0), FullDensity, v.clone())
            .wait()
            .unwrap();
        let cpu_dur = now.elapsed().as_secs() * 1000
            + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(cpu.into_affine(), gpu.into_affine());

        println!("============================");

        bases = [bases.clone(), bases.clone()].concat();
    }
}
