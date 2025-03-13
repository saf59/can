use candle_core::cpu::erf::erf;
pub mod statistics;
pub mod fft;
pub mod mfcc;
pub mod umap;
pub mod t_sne;

use std::env;
use std::fmt::Debug;

pub fn set_root() {
    let root = project_root::get_project_root().unwrap();
    let _ = env::set_current_dir(&root);
}

pub fn first_char<T: Debug>(e: &T) -> char {
    let name = format!("{:?}", e);
    name.chars().next().unwrap()
}
pub fn enum_name<T: Debug>(e: &T) -> String {
    let name = format!("_{:?}", e);
    match name.as_str() {
        "_Relu" => "".to_string(),
        _ => name.to_lowercase()
    }
}

pub fn median_and_multiplier(values: &[f32]) -> (f32, f32) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = values.len();
    let mid = len / 2;

    let median = if len % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    let centered_data: Vec<f32> = values.iter().map(|v| v - median).collect();
    let mean: f32 = centered_data.iter().sum::<f32>() / centered_data.len() as f32;
    let variance: f32 = centered_data
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>()
        / centered_data.len() as f32;
    let std_dev = variance.sqrt();

    let scale_factor = 1.0 / (2.0 * std_dev);
    let scaled_data: Vec<f32> = centered_data.iter().map(|v| v * scale_factor).collect();

    let normal_distribution_share: f32 = (1.0 - erf((1.0 / 2.0_f32.sqrt()).into())) as f32;
    let emissions_share = scaled_data
        .into_iter()
        .filter(|v| *v < 0.0 || *v > 1.0)
        .count() as f32
        / centered_data.len() as f32;

    let multiplier = if emissions_share <= normal_distribution_share {
        scale_factor
    } else {
        scale_factor * (normal_distribution_share / emissions_share)
    };

    (median, multiplier)
}
pub fn normalize_array(data: &[f32], median: f32, multiplier: f32) -> Vec<f32> {
    data.iter()
        .map(|v| normalize(*v, median, multiplier))
        .collect()
}
fn normalize(value: f32, median: f32, multiplier: f32) -> f32 {
    (value - median) * multiplier
}
