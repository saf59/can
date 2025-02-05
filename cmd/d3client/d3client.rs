use clap::Parser;
use medius_utils::detect;
use std::fs;
use std::path::Path;

/// Command line arguments for the detecting program
#[derive(Parser)]
struct Args {
    /// Path to stereo *.wav file with sample rate 192_000
    wav: String,
    /// Laser frequency
    #[arg(short, default_value_t = 37523.4522)]
    frequency: f32,
    #[arg(short, default_value = "http://app.ispredict.com:9447")]
    server: String,
}
pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Parse wav file
    let wav_path: &Path = args.wav.as_ref();
    let all = fs::read(wav_path)?;
    // Parse command line arguments
    let freq = args.frequency;
    let server = args.server;
    let verbose = false;
    // Detect wp
    let wp = detect(&all, freq, verbose)?;
    // Show result
    println!("{:5.2?}", wp);
    Ok(())
}
