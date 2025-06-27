use clap::Parser;
use medius_utils::{detect_by, show_is_cuda};
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Command line arguments for the detecting program
#[derive(Parser)]
struct Args {
    /// Verbose mode
    #[arg(short, default_value_t = false)]
    verbose: bool,
    /// Path to stereo *.wav file with sample rate 192_000
    wav: String,
    /// Laser frequency
    #[arg(short, default_value_t = 37523.4522)]
    frequency: f32,
}
pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    // Parse wav file
    let wav_path: &Path = args.wav.as_ref();
    let all = fs::read(wav_path)?;
    // Parse command line arguments
    let freq = args.frequency;
    let verbose = args.verbose;
    let meta_ba = include_bytes!("./../../models/model.meta");
    let safetensors_ba = include_bytes!("./../../models/model.safetensors");
    // Detect wp
    //let wp = detect( &all, freq, verbose, false)?;
    let wp = detect_by( &all, freq, verbose, false,meta_ba,safetensors_ba)?;
    // Show result
    if args.verbose {
        show_is_cuda();
        println!("result: {wp:5.3?}");
        println!("{:5.3?}", Instant::now().duration_since(start));
    } else {
        println!("{wp:5.3?}");
    }
    Ok(())
}
