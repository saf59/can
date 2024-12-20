use candle_core::safetensors::Load;
use candle_core::{Tensor, D};
use candle_nn::VarMap;
use clap::Parser;
use medius_meta::{Meta, ModelType};
use medius_model::{get_model, Model};
use medius_parser::parse_wav;
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
    let meta = static_meta();
    let buff_size: usize = meta.buff_size.clone() as usize;
    let inputs = meta.n;
    let dev = candle_core::Device::cuda_if_available(0)?;
    // Parse wav file
    let data = parse_wav(args.wav.as_ref() as &Path, inputs, args.frequency, buff_size)
        .unwrap();
    // Convert data(extracted wav properties) to Tensor
    let data = Tensor::from_vec(data, (1, inputs), &dev)?;
    // Build model and fill it VarMap
    let (_vm, model) = get_model(&dev, &meta, args.verbose, &fill_from_static)?;
    let result = model.forward(&data)?;
    // Extract wp from result by model type
    let wp = match meta.model_type {
        ModelType::Classification => by_class(&result),
        ModelType::Regression => by_regr(&result),
    }?;
    if args.verbose {
        println!("result: {:5.2?}", wp);
        println!("{:5.2?}", Instant::now().duration_since(start));
    } else {
        println!("{:5.2?}", wp);
    }
    Ok(())
}
/// Extract classification result
fn by_class(logits: &Tensor) -> anyhow::Result<f32> {
    let max = logits.argmax(D::Minus1).unwrap().to_vec1::<u32>().unwrap();
    let max = max.first();
    let wp: f32 = (*max.unwrap() as f32) * -0.1;
    Ok(wp)
}
/// Extract regression result
fn by_regr(logits: &Tensor) -> anyhow::Result<f32> {
    let wp: f32 = logits
        .flatten_all()?
        .get(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    Ok(wp)
}
/// Get Meta embed resource
fn static_meta() -> Meta {
    let buf = include_bytes!("./../../models/model.meta");
    serde_yaml::from_slice(buf).unwrap()
}
/// Fill VarMap from embed
fn fill_from_static(_meta: &Meta, _verbose: bool, varmap: &mut VarMap) -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    let buf = include_bytes!("./../../models/model.safetensors");
    let map = safetensors::SafeTensors::deserialize(buf).unwrap();
    for (k, v) in map.tensors() {
        let _ = varmap.set_one(k,v.load(&dev)?);
    }
    Ok(())
}
