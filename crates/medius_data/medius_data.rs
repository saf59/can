extern crate core;

use candle_core::{Device, Tensor};
use csv::Reader;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct Dataset {
    pub train_data: Tensor,
    pub train_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
}
pub fn load_dir<T: AsRef<Path>>(
    dir: T,
    train_part: f32,
) -> candle_core::Result<Dataset> {
    //let p: &Path = "./".as_ref();
    //println!("in:{:?}",p.canonicalize());
    let y = read_medius_y(&dir.as_ref())?;
    let x = read_medius_x(&dir.as_ref())?;
    let size = y.len();
    let width = x.len() / size;
    let mut indexes: Vec<usize> = (0..size).collect();
    indexes.shuffle(&mut thread_rng());
    let border: usize = ((size as f32) * train_part) as usize;
    let (train_x, train_y) = fill_x_y(0..border, &x, &y, width);
    let (test_x, test_y) = fill_x_y(border..indexes.len(), &x, &y, width);
    let mut nodes: HashMap<u8, usize> = HashMap::new();
    for n in y.iter() {
        nodes.entry(*n).and_modify(|count| *count += 1).or_insert(1);
    }
    Ok(Dataset {
        train_data: Tensor::from_vec(train_x, (border, width), &Device::Cpu)?,
        train_labels: Tensor::from_vec(train_y, border, &Device::Cpu)?,
        test_data: Tensor::from_vec(test_x, (size - border, width), &Device::Cpu)?,
        test_labels: Tensor::from_vec(test_y, size - border, &Device::Cpu)?,
        labels: nodes.len(),
    })
}
fn fill_x_y(
    ind: core::ops::Range<usize>,
    x: &Vec<f32>,
    y: &Vec<u8>,
    width: usize,
) -> (Vec<f32>, Vec<u8>) {
    let mut out_x: Vec<f32> = Vec::new();
    let mut out_y: Vec<u8> = Vec::new();
    for i in ind {
        out_y.push(y[i]);
        for ix in &x[(i * width)..((i+1) * width)] {
            out_x.push(*ix);
        }
    }
    (out_x, out_y)
}
fn read_medius_x(dir: &std::path::Path) -> candle_core::Result<Vec<f32>> {
    let reader = get_reader(dir, "x.csv")?;
    let data = reader
        .into_records()
        .flatten()
        .filter(|row| !row.is_empty())
        .flat_map(|l| l.into_iter().map(|v| v.parse::<f32>()).collect::<Vec<_>>())
        .flatten()
        .collect();
    Ok(data)
}

fn read_medius_y(dir: &std::path::Path) -> candle_core::Result<Vec<u8>> {
    let reader = get_reader(dir, "y.csv")?;
    let data = reader
        .into_records()
        .flatten()
        .filter(|row| !row.is_empty())
        .flat_map(|l| l.get(0).expect("expect u8").parse::<u8>())
        .collect();
    Ok(data)
}
fn get_reader(dir: &Path, csv: &str) -> candle_core::Result<Reader<File>> {
    let x_entry = std::fs::File::open(dir.join(csv))?;
    let reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(x_entry);
    Ok(reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_x_y() {
        //let y = read_medius_y("W:/data/medius/src/V3/stat_n260Tlist".as_ref()).unwrap();
        let base: &Path = "../../data".as_ref();
        let y = read_medius_x(&base.join("stat_n260Tlist")).unwrap();
        println!("{:?},{:?},{:?}", y.len(), y[1], y.last())
    }
    #[test]
    fn test_load_dir() {
        let base: &Path = "../../data".as_ref();
        let dataset = load_dir(&base.join("stat_n260Tlist"),0.9);
        println!("{:?}",dataset.unwrap().test_data.shape())
    }
}