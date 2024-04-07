// use rustacuda::memory::DeviceBuffer;

// #[repr(transparent)]
// struct NullDeviceBuffer(DeviceBuffer<u8>);

// unsafe impl Send for NullDeviceBuffer {}
// unsafe impl Sync for NullDeviceBuffer {}

// pub(crate) static NULL: NullDeviceBuffer = unsafe {
//     if let Ok(res) = DeviceBuffer::uninitialized(0) {
//         NullDeviceBuffer(res)
//     } else {
//         unreachable!()
//     }
// };