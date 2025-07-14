fn main() {
    // cuda.md with details about CUDA
    #[cfg(feature = "cuda")]
    unsafe {
        println!(
            "CUDA   available: {}",
            cuda_driver_sys::cuInit(0) == cuda_driver_sys::CUresult::CUDA_SUCCESS
        );
        println!(
            "Device available: {:?}",
            candle_core::utils::cuda_is_available()
        );
    }
}
