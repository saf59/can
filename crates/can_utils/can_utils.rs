use candle_core::Tensor;

pub fn l2_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let f = y_pred.flatten_all().unwrap();
    let y = y_true.flatten_all().unwrap();
    (f - y).unwrap().powf(2.).unwrap().mean(0).unwrap()
}
