use super::kernel::Kernel;

use rustacuda::{
    context::{ContextStack, CurrentContext, UnownedContext},
    error::CudaResult,
    module::Module,
    stream::{Stream, StreamFlags},
};
use std::{ffi::CString, sync::Arc};

pub struct PreparedModule {
    context: Arc<UnownedContext>,
    module: Module,
}

// TODO: re-check thread safety
unsafe impl Send for PreparedModule {}
unsafe impl Sync for PreparedModule {}

impl PreparedModule {
    pub fn from_bytes(
        context: &Arc<UnownedContext>, bytes: &[u8],
    ) -> CudaResult<Self> {
        CurrentContext::set_current(&**context)?;
        let maybe_module = Module::load_from_bytes(bytes);
        ContextStack::pop().expect("Cannot remove context.");

        Ok(Self {
            context: context.clone(),
            module: maybe_module?,
        })
    }

    pub fn create_kernel(&self, name: &str) -> CudaResult<Kernel> {
        let function_name =
            CString::new(name).expect("Kernel name must not contain nul bytes");
        let function = self.module.get_function(&function_name)?;

        // TODO: check maintenance of context in cuda.
        CurrentContext::set_current(&*self.context)?;
        let maybe_stream = Stream::new(StreamFlags::NON_BLOCKING, None);
        let stream = maybe_stream?;
        Ok(Kernel::new(function, stream))
    }
}
