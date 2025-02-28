fn main() {
	#[cfg(feature = "cuda")]
    unsafe {
        println!("CUDA available: {}", cuda_driver_sys::cuInit(0) == cuda_driver_sys::CUresult::CUDA_SUCCESS);
    }
}