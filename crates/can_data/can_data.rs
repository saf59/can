extern crate core;

use candle_core::{Device, Tensor};

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

}
