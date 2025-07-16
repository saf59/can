use anyhow::anyhow;
use anyhow::{Context, Result};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
pub mod fft;
pub mod mfcc;
pub mod statistics;
pub mod t_sne;
pub mod umap;
use std::env;
use std::fmt::Debug;

pub fn set_root() {
    let root = project_root::get_project_root().unwrap();
    let _ = env::set_current_dir(&root);
}

pub fn first_char<T: Debug>(e: &T) -> char {
    let name = format!("{e:?}");
    name.chars().next().unwrap()
}
pub fn enum_name<T: Debug>(e: &T) -> String {
    let name = format!("_{e:?}");
    match name.as_str() {
        "_Relu" => "".to_string(),
        _ => name.to_lowercase(),
    }
}

pub fn median_and_multiplier_columns(values: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<f32>)> {
    if values.is_empty() {
        return Err(anyhow!("Input is empty"));
    }
    let num_cols = values[0].len();
    if !values.iter().all(|row| row.len() == num_cols) {
        return Err(anyhow!("All rows must have the same length"));
    }

    let mut medians = Vec::with_capacity(num_cols);
    let mut multipliers = Vec::with_capacity(num_cols);

    for col in 0..num_cols {
        let col_values: Vec<f32> = values.iter().map(|row| row[col]).collect();
        let mut sorted = col_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = sorted.len();
        let mid = len / 2;
        let median = if len % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
        let centered_data: Vec<f32> = col_values.iter().map(|v| v - median).collect();
        let mean: f32 = centered_data.iter().sum::<f32>() / centered_data.len() as f32;
        let variance: f32 = centered_data
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            / centered_data.len() as f32;
        let std_dev = variance.sqrt();

        let scale_factor = 1.0 / (2.0 * std_dev);
        let scaled_data: Vec<f32> = centered_data.iter().map(|v| v * scale_factor).collect();

        let normal_distribution_share: f32 = 0.3173105;
        let emissions_share = scaled_data
            .iter()
            .filter(|v| **v < 0.0 || **v > 1.0)
            .count() as f32
            / centered_data.len() as f32;

        let multiplier = if emissions_share <= normal_distribution_share {
            scale_factor
        } else {
            scale_factor * (normal_distribution_share / emissions_share)
        };

        medians.push(median);
        multipliers.push(multiplier);
    }

    Ok((medians, multipliers))
}

pub fn normalize_data_columns(
    data: &[Vec<f32>],
    median: &[f32],
    multiplier: &[f32],
) -> Vec<Vec<f32>> {
    data.iter()
        .map(|row| normalize_row_columns(row, median, multiplier))
        .collect()
}
pub fn normalize_row_columns(data: &[f32], median: &[f32], multiplier: &[f32]) -> Vec<f32> {
    data.iter()
        .zip(median.iter().zip(multiplier.iter()))
        .map(|(v, (m, mult))| (v - m) * mult)
        .collect()
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

    let normal_distribution_share: f32 = 0.3173105;
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

pub fn ensure_dir_exists<P: AsRef<Path>>(path: P) -> Result<()> {
    let _ = fs::create_dir_all(path);
    Ok(())
}
pub fn write_csv_strings<P: AsRef<Path>>(data: &Vec<String>, path: &P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for line in data {
        writeln!(writer, "{line}")?;
    }
    Ok(())
}

pub fn vecvecf32_to_vecstring(data: &[Vec<f32>]) -> Vec<String> {
    data.iter()
        .map(|row| {
            row.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

pub fn vecstring_to_vecvecf32(data: &[String]) -> Result<Vec<Vec<f32>>> {
    data.iter()
        .enumerate()
        .map(|(i, line)| {
            line.split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f32>()
                        .with_context(|| format!("Parse error '{}' on line {}", s, i + 1))
                })
                .collect()
        })
        .collect()
}
pub fn read_csv_strings<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let file =
        File::open(&path).with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    for line in reader.lines() {
        data.push(line?);
    }
    Ok(data)
}

pub fn vecusize_to_vecstring(data: &[usize]) -> Vec<String> {
    data.iter().map(|v| v.to_string()).collect()
}

pub fn vecstring_to_vecusize(data: &[String]) -> Result<Vec<usize>> {
    data.iter()
        .enumerate()
        .map(|(i, s)| {
            s.trim()
                .parse::<usize>()
                .with_context(|| format!("Parse error '{}' on line {}", s, i + 1))
        })
        .collect()
}
pub fn column_averages(data: &Vec<Vec<f32>>) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    let cols = data[0].len();
    let mut sums = vec![0.0; cols];
    let mut counts = vec![0; cols];

    for row in data {
        for (i, &val) in row.iter().enumerate() {
            sums[i] += val;
            counts[i] += 1;
        }
    }

    sums.iter()
        .zip(counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    //#[ignore]
    fn test_normalize_row_columns_with_default_mm() {
        let (medians, multipliers) = default_mm();
        // The first row from data/H34_SF/x.csv was used
        let data = vec![
            38464.332,297664600.0,0.68671995,0.05161904,0.016855761,3.2270377,1.2013962,1.6664546,2.5818412,4.345497,38464.332,297664260.0,3431049200000.0,702762900000000.0,0.68671995,3.0516186,17229.715,85029.086,772957.44,-0.20920213,-1.1790084,0.552654,3.7304523,1.0001067,1.0003206,1.0006407,1.0010673,85029.086,772957.06,-117948910.0,-3198579400000.0,0.2865481,1.8209916,876.315
        ];
        // The first row from data/H34_ST/x.csv was used
        let expected: Vec<f32> = vec![
            0.2570546,-0.055317216,-0.40166542,-0.21743114,0.013794116,-0.0144968815,-0.25794613,-0.28998002,-0.29007003,-0.27395043,0.2570546,-0.055322323,-0.5084484,-0.0,-0.40168872,-0.2174313,-0.061788652,0.009051006,0.1450848,-0.12806472,-0.04461706,-0.097088784,-0.09955013,0.14152722,0.14285941,0.14243728,0.14134958,0.009051006,0.14551343,-0.09854499,-0.019760655,0.1760994,-0.04461706,0.12963651
            ];

        let result = normalize_row_columns(&data, &medians, &multipliers);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
        }
    }
    #[rustfmt::skip]
    fn default_mm() -> (Vec<f32>, Vec<f32>){
        let medians = vec![35932.02,309936100.0,1.0617762,0.7261131,0.016100075,3.2556899,1.2491057,1.8738801,3.213705,6.054162,35932.02,309936830.0,5849854500000.0,64886580000000000.0,1.0617762,3.726113,17605.002,85024.52,712766.94,-0.09914484,-1.1225301,0.5775845,3.7607677,1.0000986,1.0002959,1.0005915,1.0009857,85024.52,712704.0,0.0,0.0,0.17612782,1.8774699,844.25525];
        let multipliers = vec![0.00010151042,0.000000004507747,1.0709519,0.32236034,18.253918,0.5059593,5.406613,1.3980033,0.4590763,0.1603308,0.00010151042,0.0000000045077693,0.00000000000021020621,0.0,1.0710099,0.32236034,0.00016464255,0.0019837855,0.0000024104188,1.1636093,0.7899756,3.894378,3.2838109,17458.996,5789.309,2893.1082,1733.5161,0.0019837855,0.000002415025,0.00000000083550206,0.0000000000000061779496,1.5947978,0.7899756,0.004043599];
        (medians, multipliers)
    }
}
