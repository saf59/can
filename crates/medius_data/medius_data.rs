extern crate core;

use candle_core::{Device, Tensor};
use csv::Reader;
use rand::rng;
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

pub struct Dataset {
    pub train_data: Tensor,
    pub train_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
}
impl Dataset {
    pub fn classes(&self, is_regression:bool) -> usize {
        if is_regression {return 1;}
        let train_classes: HashSet<_> = self.train_labels.flatten_all().unwrap().to_vec1().unwrap().into_iter().collect();
        let test_classes: HashSet<_> = self.test_labels.flatten_all().unwrap().to_vec1::<u8>().unwrap().into_iter().collect();
        train_classes.union(&test_classes).count()
    }
}

pub fn load_dir<T: AsRef<Path>>(dir: T, train_part: f32,dev: &Device) -> candle_core::Result<Dataset> {
    let y = read_medius_y(dir.as_ref())?;
    let x = read_medius_x(dir.as_ref())?;
    let size = y.len();
    let width = x.len() / size;
    let mut indexes: Vec<usize> = (0..size).collect();
    indexes.shuffle(&mut rng()); // randomize indexes
    let border: usize = ((size as f32) * train_part) as usize;
    let (train, test) = if train_part < 1.0 {
        (0..border, border..indexes.len())
    } else {
        (0..indexes.len(), 0..indexes.len())
    };
    let (train_size, test_size) = (&train.len(),&test.len());
    let (train_x, train_y) = fill_x_y(train, &x, &y, width);
    let (test_x, test_y) = fill_x_y(test, &x, &y, width);
    Ok(Dataset {
        train_data: Tensor::from_vec(train_x, (*train_size, width), dev)?,
        train_labels: Tensor::from_vec(train_y, *train_size, dev)?,
        test_data: Tensor::from_vec(test_x, (*test_size, width), dev)?,
        test_labels: Tensor::from_vec(test_y, *test_size, dev)?,
    })
}
fn fill_x_y(
    ind: core::ops::Range<usize>,
    x: &[f32],
    y: &[u8],
    width: usize,
) -> (Vec<f32>, Vec<u8>) {
    let mut out_x: Vec<f32> = Vec::new();
    let mut out_y: Vec<u8> = Vec::new();
    for i in ind {
        out_y.push(y[i]);
        for ix in &x[(i * width)..((i + 1) * width)] {
            out_x.push(*ix);
        }
    }
    (out_x, out_y)
}
fn read_medius_x(dir: &Path) -> candle_core::Result<Vec<f32>> {
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

fn read_medius_y(dir: &Path) -> candle_core::Result<Vec<u8>> {
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
    let x_entry = File::open(dir.join(csv))?;
    let reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(x_entry);
    Ok(reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use utils::set_root;

    const BASE: &str = "./data/B260_ST";

    #[test]
    fn test_x_y() {
        set_root();
        let y = read_medius_x(&Path::new(BASE)).unwrap();
        println!("{:?},{:?},{:?}", y.len(), y[1], y.last())
    }
    #[test]
    fn test_load_dir() {
        set_root();
        let device = Device::cuda_if_available(0).unwrap();
        let dataset = load_dir(BASE, 0.9,&device).unwrap();
        println!("{:?}-> {:?}", &dataset.test_data.shape(),&dataset.classes(false));
    }
}

/// Print information about the dataset
pub fn print_dataset_info(m: &Dataset) {
    print!("train-data: {:?}", m.train_data.shape());
    print!(", train-labels: {:?}", m.train_labels.shape());
    print!(", test-data: {:?}", m.test_data.shape());
    println!(", test-labels: {:?}", m.test_labels.shape());
}
