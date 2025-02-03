use candle_core::{Device, Tensor, D};
use candle_core::safetensors::Load;
use candle_nn::VarMap;
use medius_meta::{Meta, ModelType};
use medius_model::{get_model, Model};
use medius_parser::parse_all;

pub fn detect(all: &Vec<u8>, freq: f32, verbose: bool) -> anyhow::Result<f32> {
    let meta = static_meta();
    let buff_size: usize = meta.buff_size.clone() as usize;
    let inputs = meta.n;
    let dev = candle_core::Device::cuda_if_available(0)?;

    let data = parse_all(all, inputs, freq, buff_size)?;
    // Convert data(extracted wav properties) to Tensor
    let data = Tensor::from_vec(data, (1, inputs), &dev)?;
    // Build model and fill it VarMap
    let (_vm, model) = get_model(&dev, &meta, verbose, &fill_from_static)?;
    let result = model.forward(&data)?;
    // Extract wp from result by model type
    let wp = match meta.model_type {
        ModelType::Classification => by_class(&result),
        ModelType::Regression => by_regr(&result),
    }?;
    Ok(wp)
}

/// Extract classification result
fn by_class(logits: &Tensor) -> anyhow::Result<f32> {
    let max = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let max = max.first();
    let wp: f32 = (*max.unwrap() as f32) * -0.1;
    Ok(wp)
}
/// Extract regression result
fn by_regr(logits: &Tensor) -> anyhow::Result<f32> {
    let wp: f32 = logits
        .flatten_all()?
        .get(0)?
        .to_scalar::<f32>()?;
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
    let map = safetensors::SafeTensors::deserialize(buf)?;
    for (k, v) in map.tensors() {
        let _ = varmap.set_one(k,v.load(&dev)?);
        // v.load(&dev)? ->  v.convert(&dev)?  in 0.5 version, but candle 0.8.2 use load
    }
    Ok(())
}

pub fn show_is_cuda() {
    let device = Device::cuda_if_available(0).unwrap();
    println!("Device:{:?}",device);
}

#[cfg(test)]
mod tests {
    use crate::show_is_cuda;
    #[test]
    fn is_cuda() {
        show_is_cuda()
    }
}