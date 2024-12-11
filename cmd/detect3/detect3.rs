use std::path::Path;
use clap::Parser;
use std::time::Instant;
use candle_core::{Device, Tensor, D};
use candle_nn::ops;
use medius_model::{get_model, Model};
use medius_parser::{parse_wav, BufSize};

#[derive(Parser)]
struct Args {
    /// Path to stereo *.wav file with sample rate 192_000
    wav:String,
    /// FFT buffer size
    #[clap(short, value_enum, default_value_t = BufSize::Small)]
    buf_size: BufSize,
    /// Number of beans
    #[arg(short, default_value_t = 260)]
    n: usize,
    /// Laser frequency
    #[arg(short, default_value_t = 37523.4522)]
    frequency: f32,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    let buf_size: usize = args.buf_size as usize;
    let inputs = args.n;
    let data = parse_wav(args.wav.as_ref() as &Path, args.n, args.frequency, buf_size).unwrap();
    let data = Tensor::from_vec(data, (1,inputs), &Device::Cpu)?;
    let dev = candle_core::Device::cuda_if_available(0)?;
    let (varmap, model) = get_model(&dev, inputs, 5, 40, 10, "./models/stat_n260Tlist_40_10.safetensors".as_ref())?;
    let logits = model.forward(&data)?;
    let max =logits.argmax(D::Minus1);
    let log_sm = ops::log_softmax(&logits, D::Minus1)?;
    println!("{:?} {:?}",log_sm,max?);

    println!("{:5.2?}", Instant::now().duration_since(start));
    Ok(())
}
