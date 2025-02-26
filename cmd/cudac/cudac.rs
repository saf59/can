#[cfg(target_os = "windows")]
fn main() {
    unsafe {
        println!("CUDA available: {}", cuda_driver_sys::cuInit(0) == cuda_driver_sys::CUresult::CUDA_SUCCESS);
    }
}