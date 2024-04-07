use crate::params::{DeviceParam, NullPointer};

use super::params::{Param, ParamIO};

use std::ffi::c_void;

use rustacuda::{
    context::ContextStack, error::CudaResult, function::Function,
    stream::Stream,
};

#[cfg(feature = "timer")]
use std::time::Instant;

#[derive(Debug)]
pub struct KernelConfig {
    pub global_work_size: usize,
    pub local_work_size: usize,
    pub shared_mem: usize,
}

pub struct Kernel<'a> {
    function: Function<'a>,
    stream: Stream,
}

impl<'a> Kernel<'a> {
    pub fn new(function: Function<'a>, stream: Stream) -> Self {
        Self { function, stream }
    }

    pub fn with_args<'b>(self) -> KernelWithArgs<'a, 'b> {
        KernelWithArgs {
            k: self,
            args: Vec::new(),
            #[cfg(feature = "timer")]
            instant: Instant::now(),
        }
    }

    pub fn sync_to_device<'b, T>(&self, param: &mut DeviceParam<'_, T>) -> CudaResult<()> {
        param.to_device(&self.stream)
    }

    pub fn sync_to_host<'b, T>(&self, param: &mut DeviceParam<'_, T>) -> CudaResult<()> {
        param.to_host(&self.stream)
    }
}

pub struct KernelWithArgs<'a, 'b> {
    k: Kernel<'a>,
    args: Vec<Box<dyn ParamIO + 'b>>,
    #[cfg(feature = "timer")]
    instant: Instant,
}

impl<'a, 'b> KernelWithArgs<'a, 'b> {
    #[inline]
    #[allow(unused_variables)]
    pub fn elapsed(&self, note: &str) {
        #[cfg(feature = "timer")]
        {
            println!(
                "Elapsed {} us ({})",
                self.instant.elapsed().as_micros(),
                note
            );
        }
    }

    pub fn val<T: 'static>(mut self, input: T) -> CudaResult<Self> {
        self.receive_param(Param::InVal(input))?;
        Ok(self)
    }

    pub fn in_ref<T>(mut self, input: &'b T) -> CudaResult<Self> {
        self.receive_param(Param::InRef(input))?;
        Ok(self)
    }

    pub fn in_mut<T>(mut self, input: &'b mut T) -> CudaResult<Self> {
        self.receive_param(Param::InMut(input))?;
        Ok(self)
    }

    pub fn out<T>(mut self, output: &'b mut T) -> CudaResult<Self> {
        self.receive_param(Param::Out(output))?;
        Ok(self)
    }

    pub fn in_ref_slice<T>(mut self, input: &'b [T]) -> CudaResult<Self> {
        self.receive_param(Param::InRefSlice(input))?;
        Ok(self)
    }

    pub fn in_mut_slice<T>(mut self, input: &'b mut [T]) -> CudaResult<Self> {
        self.receive_param(Param::InMutSlice(input))?;
        Ok(self)
    }

    pub fn out_slice<T>(mut self, output: &'b mut [T]) -> CudaResult<Self> {
        self.receive_param(Param::OutSlice(output))?;
        Ok(self)
    }

    pub fn dev_arg<T>(mut self, output: &'b DeviceParam<'_, T>) -> CudaResult<Self> {
        self.elapsed("device param");
        self.args.push(Box::new(output));
        Ok(self)
    }

    pub fn empty(mut self) -> CudaResult<Self> {
        self.elapsed("empty param");
        self.args.push(Box::new(NullPointer));
        Ok(self)
    }

    fn receive_param<T>(&mut self, arg: Param<'b, T>) -> CudaResult<()> {
        let b = arg.before_call(&self.k.stream)?;
        self.elapsed("param");
        self.args.push(Box::new((arg, b)));
        Ok(())
    }

    fn run_inner(&mut self, config: KernelConfig) -> CudaResult<()> {
        let args: Vec<*mut c_void> =
            self.args.iter().map(|x| x.param_pointer()).collect();
        self.elapsed("before launch");
        unsafe {
            self.k
                .stream
                .launch(
                    &self.k.function,
                    config.global_work_size as u32,
                    config.local_work_size as u32,
                    config.shared_mem as u32,
                    &args,
                )
                ?
        }
        self.elapsed("after launch");
        self.k.stream.synchronize()?;
        self.elapsed("exec done");
        Ok(())
    }

    pub fn run(mut self, config: KernelConfig) -> CudaResult<Kernel<'a>> {
        self.run_inner(config)?;
        for mut param in self.args.drain(..) {
            param.after_call(&self.k.stream)?;
            #[cfg(feature = "timer")]
            {
                println!(
                    "Elapsed {} us (back)",
                    self.instant.elapsed().as_micros()
                );
            }
        }
        Ok(self.k)
    }
}

impl<'a> Drop for Kernel<'a> {
    fn drop(&mut self) { let _ = ContextStack::pop(); }
}
