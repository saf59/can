use candle_core::{Tensor, D};
use clap::Parser;
use medius_meta::{Meta, ModelType, DEFAULT, MODELS_DIR};
use medius_model::{get_model, Model};
use medius_parser::parse_wav;
use std::fs;
use std::fs::create_dir_all;
use std::path::Path;
use std::time::Instant;

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
    let meta = init();
    let buff_size: usize = meta.buff_size.clone() as usize;
    let inputs = meta.n;
    let dev = candle_core::Device::cuda_if_available(0)?;
    let data = parse_wav(args.wav.as_ref() as &Path, inputs, args.frequency, buff_size).unwrap();
    let data = Tensor::from_vec(data, (1,inputs), &dev)?;
    let (_vm,model) = get_model(&dev, &meta, args.verbose)?;
    let result = model.forward(&data)?;
    let wp = match meta.model_type {
        ModelType::Classification => by_class(&result),
        ModelType::Regression => by_regr(&result)
    }?;
    if args.verbose {
        println!("result: {:5.2?}", wp);
        println!("{:5.2?}", Instant::now().duration_since(start));
    } else {println!("{:5.2?}", wp);}
    Ok(())
}

fn by_class(logits: &Tensor) -> anyhow::Result<f32> {
    let max = logits.argmax(D::Minus1).unwrap().to_vec1::<u32>().unwrap();
    let max = max.first();
    let wp: f32 = (*max.unwrap() as f32) * -0.1;
    Ok(wp)
}
fn by_regr(logits: &Tensor) -> anyhow::Result<f32> {
    let wp: f32 = logits.flatten_all()?.get(0).unwrap().to_scalar::<f32>().unwrap();
    Ok(wp)
}

pub fn init() -> Meta {
    if !(DEFAULT.as_ref() as &Path).exists() {
        let bytes = include_bytes!("./../../models/model.meta");
        let _ = create_dir_all(MODELS_DIR);
        fs::write(DEFAULT, bytes).expect("Unable to write default meta file");
    }
    let meta = Meta::load_default();
    let binding = &meta.model_file();
    let model_path: &Path = binding.as_ref();
    if !model_path.exists() {
        let bytes = include_bytes!("./../../models/model.safetensors");
        let _ = create_dir_all(model_path.parent().unwrap());
        fs::write(model_path, bytes).expect("Unable to write default meta file");
    }
    meta
}