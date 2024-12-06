use candle_core::{Device, Tensor};

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
fn read_medius_x(dir: &std::path::Path) -> candle_core::Result<(Vec<f32>)> {
    let x_entry = std::fs::File::open(&dir.join("x.csv"))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(x_entry);
    let mut data = vec![0f32; 100];
    for row in reader.records() {
        let row = match row {
            Err(_) => continue,
            Ok(row) => {
                if row.len() > 0 {
                    row.iter() //.map(f32::from)
                        .for_each(|v| data.push(v.parse().unwrap()))
                }
            }
        };
    }
    Ok(data)
}
fn read_medius_y(dir: &std::path::Path) -> candle_core::Result<(Vec<u8>)> {
    println!("Path: {:?}",&dir.join("y.csv"));
    let x_entry = std::fs::File::open(&dir.join("y.csv"))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(x_entry);
    let mut data = vec![0u8];
    for row in reader.records() {
        let row = match row {
            Err(_) => continue,
            Ok(row) => {
                if row.len()>0 { data.push(row.get(0).unwrap().parse().unwrap())}
            }
        };
    }
    Ok(data)
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
        let y = read_medius_y("../../data/stat_n260Tlist".as_ref()).unwrap();
        println!("{:?},{:?},{:?}",y.len(),y[1],y.last())
    }
}
