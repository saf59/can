use anyhow::{Context, Result};
use ho::HigherOrderMomentsAnalyzer;
use indicatif::ProgressBar;
use medius_parser::hob;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;
use utils::{
    column_averages, ensure_dir_exists, median_and_multiplier_columns, normalize_data_columns,
    vecusize_to_vecstring, vecvecf32_to_vecstring, write_csv_strings,
};

fn main() {
    let path = r"W:\data\medius\part3\impulse_meta.csv";
    let rows = read_meta_rows_from_csv(path).unwrap();
    println!("Read {} rows", rows.len());
    let grouped = group_and_sort_meta_rows_by_path(&rows);
    println!("Groups: {:?}", grouped.keys().len());

    let bands = 1;
    let band_width = 3000.0;
    let centers = vec![85_000.0];
    let window = ""; //"hanning";
    let mut analyzer = HigherOrderMomentsAnalyzer::default();

    let num_iters = rows.len() as u64;
    let now = Instant::now();
    let bar = ProgressBar::new(num_iters);
    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<usize> = Vec::new();
    let mut x_raw: Vec<Vec<f32>> = Vec::new();
    let mut y_raw: Vec<usize> = Vec::new();
    for group in grouped.values() {
        let mut x_group: Vec<Vec<f32>> = Vec::new();
        let mut y_group: usize = 0;
        for row in group {
            let impulse_path = format!(
                "W:/data/medius/part3/impulse/{}/{}_{}.bin",
                row.r#type, row.typeIndex, row.pulse
            );
            let impulse_path: &Path = impulse_path.as_ref();
            if !impulse_path.exists() {
                println!("Impulse: {impulse_path:?}");
            } else {
                // Your processing logic here
                let data_f64 = read_f64_vec_from_file(impulse_path)
                    .with_context(|| format!("Failed to read f64 data from: {impulse_path:?}"))
                    .unwrap();
                let data_f32 = f64_slice_to_f32_vec(&data_f64);
                let result = hob(
                    &data_f32,
                    bands,
                    band_width,
                    window,
                    &centers,
                    &mut analyzer,
                );
                // тут надо добавить данные в 4 вектора
                x.push(result.clone());
                x_group.push(result);
                y.push(row.r#type);
                y_group = row.r#type;
            }
            bar.inc(1);
        }
        x_raw.push(column_averages(&x_group));
        y_raw.push(y_group);
    }
    bar.finish();
    let elapsed = now.elapsed();
    let calc_per_sec: f64 = (num_iters as f64) / (elapsed.as_secs() as f64);
    println!("Total runtime: {elapsed:.2?}");
    println!("Calculations per second: {calc_per_sec:.2?}.");
    save_to_csv(&mut x, &mut y, "data/H34_BF/".as_ref());
    save_to_csv(&mut x_raw, &mut y_raw, "data/H34_SF/".as_ref());
    //build median and multyplier
    let (median, multyplier) = median_and_multiplier_columns(&x).unwrap();
    // Normalize data
    let mut x_norm = normalize_data_columns(&x, &median, &multyplier);
    let mut x_raw_norm = normalize_data_columns(&x_raw, &median, &multyplier);
    // Save normalized data
    save_to_csv(&mut x_norm, &mut y, "data/H34_BT/".as_ref());
    save_to_csv(&mut x_raw_norm, &mut y_raw, "data/H34_ST/".as_ref());
    // Save median and multyplier to 2 CSV
    let norm: Vec<Vec<f32>> = vec![median.clone(), multyplier.clone()];
    let norm_strings = vecvecf32_to_vecstring(&norm);
    write_csv_strings(&norm_strings, &"data/H34_BT/norm.csv").unwrap();
    write_csv_strings(&norm_strings, &"data/H34_ST/norm.csv").unwrap();
}
fn save_to_csv(x: &mut [Vec<f32>], y: &mut [usize], dir: &Path) {
    let x_strings = vecvecf32_to_vecstring(x);
    let y_strings = vecusize_to_vecstring(y);
    ensure_dir_exists(dir).unwrap();
    write_csv_strings(x_strings.as_ref(), &dir.join("x.csv")).unwrap();
    write_csv_strings(y_strings.as_ref(), &dir.join("y.csv")).unwrap();
}
#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct MetaRow {
    pub path: String,
    pub r#type: usize,
    pub typeIndex: usize,
    pub pulse: usize,
}
fn group_and_sort_meta_rows_by_path(rows: &[MetaRow]) -> HashMap<String, Vec<&MetaRow>> {
    let mut map: HashMap<String, Vec<&MetaRow>> = HashMap::new();
    for row in rows {
        map.entry(row.path.clone()).or_default().push(row);
    }
    for group in map.values_mut() {
        group.sort_by_key(|r| r.pulse);
    }
    map
}
fn read_meta_rows_from_csv<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<MetaRow>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_path(&path)
        .with_context(|| format!("Failed to open CSV file: {:?}", path.as_ref()))?;
    rdr.records().next(); // Skip the header row
    let mut rows = Vec::new();
    for result in rdr.deserialize() {
        let row: MetaRow = result.with_context(|| "Failed to deserialize MetaRow")?;
        if !row.path.contains("_MS150_5_P30_2_20_") {
            rows.push(row);
        }
    }
    Ok(rows)
}
fn read_f64_vec_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    let bytes =
        fs::read(&path).with_context(|| format!("Failed to read file: {:?}", path.as_ref()))?;
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let val = f64::from_be_bytes(chunk.try_into().context("Failed to convert bytes to f64")?);
        out.push(val);
    }
    Ok(out)
}
pub fn f64_slice_to_f32_vec(input: &[f64]) -> Vec<f32> {
    input.iter().map(|&x| x as f32).collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore] // Ignore this test by default, as it requires a specific file path
    fn test_read_f64_vec_from_file() {
        let path: &Path = "W:/data/medius/part3/impulse/0/0_0.bin".as_ref();

        // Read back using the function
        let result = read_f64_vec_from_file(path).unwrap();
        let first = result.first().unwrap();
        let last = result.last().unwrap();
        // Read 2586, first: 0.05157470703125, last:0.214935302734375
        println!("{:?},{:?},{:?},", result.len(), first, last);
        assert_eq!(result.len(), 2586);
        assert!((first - 0.05157470703125).abs() < 1e-6);
        assert!((last - 0.214935302734375).abs() < 1e-6);
    }
    #[test]
    #[ignore] // Requires the actual file at the given path
    fn test_read_meta_rows_from_csv() {
        let path = r"W:\data\medius\part3\impulse_meta.csv";
        let rows = read_meta_rows_from_csv(path).unwrap();
        assert!(!rows.is_empty(), "No rows read from CSV");
        for row in &rows {
            assert!(
                !row.path.contains("_MS150_5_P30_2_20_"),
                "Filtered row found"
            );
        }
        println!("Read {} rows", rows.len());
        let grouped = group_and_sort_meta_rows_by_path(&rows);
        println!("Groups: {:?}", grouped.keys().len());
    }
}
