use clap::Parser;
use medius_utils::{detect, show_is_cuda};
use std::fs;
use std::path::Path;
use std::time::Instant;
use medius_meta::static_meta;

/// Command line arguments for the detecting program
#[derive(Parser)]
struct Args {
    /// Verbose mode
    #[arg(short, default_value_t = false)]
    verbose: bool,
    /// Path to stereo *.wav file with sample rate 192_000
    wav: String,
    /// Joined mode, default is false
    #[arg(short, default_value_t = false)]
    joined: bool,
}
pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    // Parse wav file
    let wav_path:&Path = args.wav.as_ref();
    let joined = args.joined;
    let all = fs::read(wav_path)?;
    // Parse command line arguments
    let verbose = args.verbose;
    // Detect wp
    let meta = static_meta();
    let class =  detect(meta, &all, 0.0, verbose, joined)?;
    // Show result
    if args.verbose {
        show_is_cuda();
        println!("result: {:?}", class as i32);
        println!("{:5.3?}", Instant::now().duration_since(start));
    } else {
        println!("{:?}", class as i32);
    }
    Ok(())
}
