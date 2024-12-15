use candle_core::{Tensor, D};
use clap::Parser;
use medius_meta::Meta;
use medius_model::{get_model, Model};
use medius_parser::parse_wav;
use std::path::Path;
use std::time::Instant;
use candle_nn::VarMap;

#[derive(Parser)]
struct Args {
    /// Verbose mode
    #[arg(short, default_value_t = false)]
    verbose:bool,
    /// Path to stereo *.wav file with sample rate 192_000
    wav:String,
    /// Laser frequency
    #[arg(short, default_value_t = 37523.4522)]
    frequency: f32,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    let meta = &Meta::load_default();
    let buff_size: usize = meta.buff_size.clone() as usize;
    let inputs = meta.n;
    let dev = candle_core::Device::cuda_if_available(0)?;

    let data = parse_wav(args.wav.as_ref() as &Path, inputs, args.frequency, buff_size).unwrap();
    let data = Tensor::from_vec(data, (1,inputs), &dev)?;
    let mut varmap = VarMap::new();
    let model = get_model(&dev, &mut varmap,  meta, args.verbose)?;
    let logits = model.forward(&data)?;
    let max =logits.argmax(D::Minus1).unwrap().to_vec1::<u32>().unwrap();
    let max = max.first();
    let wp:f32 = (*max.unwrap() as f32)  * -0.1;
    if args.verbose {
        println!("result: {:?}", wp);
        println!("{:5.2?}", Instant::now().duration_since(start));
    } else {println!("{:?}", wp);}
    Ok(())
}
