use anyhow::Error;
use candle_core::{Device, Tensor, D};
use candle_nn::VarMap;
use medius_meta::{fill_safetensors, meta_from_ba, safetensors_from_ba, AlgType, Meta, ModelType};
use medius_model::{get_model, Model};
use medius_parser::{parse_all, parse_hom, read_wav};
use utils::{column_averages, default_mm, normalize_row_columns};
pub fn detect_by(
    all: &[u8],
    freq: f32,
    verbose: bool,
    joined: bool,
    meta_ba: &[u8],
    safetensor_ba: &[u8],
) -> anyhow::Result<f32> {
    let meta = meta_from_ba(meta_ba)?;
    let buff_size: usize = meta.buff_size.clone() as usize;
    let inputs = meta.n;
    let dev = Device::cuda_if_available(0)?;
    let alg_type = meta.alg_type.clone();
    let fill = move |_meta: &Meta, _flag: bool, varmap: &mut VarMap| {
        let safetensors = safetensors_from_ba(safetensor_ba)?;
        fill_safetensors(varmap, safetensors)
    };

    if alg_type != medius_meta::AlgType::HOM {
        let data = parse_all(all, inputs, freq, buff_size, alg_type)?;
        detect_by_single_vec(verbose, &meta, inputs, &dev, &data, &fill)
    } else {
        let raw = read_wav(all)?;
        detect_by_many_vectors(verbose, joined, &meta, inputs, &dev, &raw, &fill)
    }
}

fn detect_by_many_vectors(
    verbose: bool,
    joined: bool,
    meta: &Meta,
    inputs: usize,
    dev: &Device,
    raw: &[f32],
    fill: &dyn Fn(&Meta, bool, &mut VarMap) -> anyhow::Result<()>,
) -> anyhow::Result<f32> {
    let hom_data = parse_hom(raw)?;
    let (medians, multiplier) = default_mm();
    if joined {
        // If joined, we use the same function as for all
        let mut avg = column_averages(&hom_data);
        //println!("avg1: {:?}", avg);
        if meta.flag {
            avg = normalize_row_columns(&avg, &medians, &multiplier);
        }
        //println!("avg2: {:?}", avg);
        detect_by_single_vec(verbose, meta, inputs, dev, &avg, fill)
    } else {
        // If not joined, we use the special function for hom
        let results = hom_data
            .iter()
            .map(|h| {
                let hn = if meta.flag {
                    // If flag is true, we normalize the data
                    normalize_row_columns(h, &medians, &multiplier)
                } else {
                    h.to_vec()
                };
                detect_by_single_vec(verbose, meta, inputs, dev, &hn, fill)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(most_frequent_value(&results).unwrap_or(0.0))
    }
}
fn most_frequent_value(values: &[f32]) -> anyhow::Result<f32> {
    let mut counts = std::collections::HashMap::new();
    for &v in values {
        let key = (v * 100.0).round() as i32;
        *counts.entry(key).or_insert(0) += 1;
    }
    let most = counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(value, _)| value as f32 / 100.0);
    most.ok_or_else(|| anyhow::anyhow!("No values found"))
}
fn detect_by_single_vec(
    verbose: bool,
    meta: &Meta,
    inputs: usize,
    dev: &Device,
    data: &[f32],
    fill: &dyn Fn(&Meta, bool, &mut VarMap) -> anyhow::Result<()>,
) -> anyhow::Result<f32> {
    // Convert data(extracted wav properties) to Tensor
    let data = Tensor::from_vec(data.to_vec(), (1, inputs), dev)?;
    // Build model and fill it VarMap
    let (_vm, model) = get_model(dev, meta, verbose, fill)?;
    let result = model.forward(&data)?;
    // Extract wp from result by model type
    let wp = match meta.model_type {
        ModelType::Classification => by_class(&result, &meta.alg_type),
        ModelType::Regression => by_regr(&result),
    }?;
    Ok(wp)
}

/// Extract classification result
fn by_class(logits: &Tensor, alg_type: &AlgType) -> anyhow::Result<f32> {
    let max = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let max = max.first();
    if alg_type == &AlgType::HOM {
        // For HOM, we return the first value as the result
        return Ok(*max.unwrap_or(&0) as f32);
    }
    let wp: f32 = (*max.unwrap() as f32) * -0.1;
    Ok(wp)
}

/// Extract regression result
fn by_regr(logits: &Tensor) -> anyhow::Result<f32> {
    let wp: f32 = logits.flatten_all()?.get(0)?.to_scalar::<f32>()?;
    Ok(wp)
}

pub fn show_is_cuda() {
    let device = Device::cuda_if_available(0).unwrap();
    println!("Device:{device:?}");
}

#[cfg(test)]
mod tests {
    use crate::show_is_cuda;
    #[test]
    fn is_cuda() {
        show_is_cuda()
    }
}
