use candle_core::{Device, Tensor};
use csv::Reader;
use std::fs::File;
use std::path::Path;

pub struct Dataset {
    pub train_data: Tensor,
    pub train_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
    pub data_width: usize,
}
/*pub fn load_dir<T: AsRef<std::path::Path>>(dir: T) -> candle_core::Result<Dataset> {
    let dir = dir.as_ref();
    //let (x,y) = read_x_y(&dir)?;

}
*/
fn read_medius_x(dir: &std::path::Path) -> candle_core::Result<Vec<f32>> {
    let reader = get_reader(dir, "x.csv")?;
    let data= reader.into_records()
        .flatten()
        .filter(|row| !row.is_empty())
        .flat_map(|l| l.into_iter().map(|v| v.parse::<f32>()).collect::<Vec<_>>())
        .flatten()
        .collect();
    Ok(data)
}

fn read_medius_y(dir: &std::path::Path) -> candle_core::Result<Vec<u8>> {
    let reader = get_reader(dir, "y.csv")?;
    let data= reader.into_records()
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

pub fn data_set(device: &Device) -> (Tensor, Tensor) {
    let x = Tensor::from_slice(
        &[
            1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.,
            1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0.,
            0., 0., 1., 1.,
        ],
        (1, 1, 6, 8),
        device,
    )
    .unwrap();
    let y = Tensor::from_slice(
        &[
            0., 1., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., -1., 0.,
            0., 1., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., -1., 0.,
        ],
        (6, 7),
        device,
    )
    .unwrap();
    (x, y)
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_x_y() {
        //let y = read_medius_y("W:/data/medius/src/V3/stat_n260Tlist".as_ref()).unwrap();
        let base:&Path = "../../data".as_ref();
        let y = read_medius_x(&base.join("stat_n260Tlist")).unwrap();
        println!("{:?},{:?},{:?}", y.len(), y[1], y.last())
    }
}
