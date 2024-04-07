#[macro_export]
macro_rules! program {
    ($device:ident) => {
        compile_error!(
            "At least one of the features `cuda` or `opencl` must be enabled."
        );
    };
}

#[cfg(feature = "test-tools")]
#[macro_export]
macro_rules! load_program {
    ($device:ident) => {
        compile_error!(
            "At least one of the features `cuda` or `opencl` must be enabled."
        );
    };
}
