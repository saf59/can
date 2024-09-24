use candle_core::{Device, Tensor};
use std::error::Error;
use std::time::Instant;

// Как работают трансформеры: разбираем математику / Хабр
// https://habr.com/ru/articles/785474/
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;
    let start = Instant::now();
    let _ = action(&device);
    println!("{:?}, Cuda:{:?}", Instant::now().duration_since(start), &device.is_cuda());
    Ok(())
}

fn action(device: &Device) -> Result<(), Box<dyn Error>> {
    let wk1 = Tensor::from_slice(&[1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.], (4, 3), device)?;
    let wv1 = Tensor::from_slice(&[0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0.], (4, 3), device)?;
    let wq1 = Tensor::from_slice(&[0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0.], (4, 3), device)?;

    let wk2 = Tensor::from_slice(&[0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 1., 0.], (4, 3), device)?;
    let wv2 = Tensor::from_slice(&[1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.], (4, 3), device)?;
    let wq2 = Tensor::from_slice(&[1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1.], (4, 3), device)?;

    let embedding = Tensor::from_slice(&[1., 3., 3., 5., 2.84, 3.99, 4., 6.], (2, 4), device)?;

    prepare(&embedding, &wk1, &wv1, &wq1, &device)?;
    let attention1 = attention(&embedding, &wk1, &wv1, &wq1, device)?;
    let attention2 = attention(&embedding, &wk2, &wv2, &wq2, device)?;
    pv("attention1", &attention1);
    pv("attention2", &attention2);
    // 4.3.5  attentions = np.concatenate([attention1, attention2], axis=1)
    let attentions = Tensor::cat(&[attention1, attention2], 1)?;
    pv("attentions", &attentions);
    Ok(())
}
fn attention(x: &Tensor, wk: &Tensor, wv: &Tensor, wq: &Tensor, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let k = x.matmul(wk)?;
    let v = x.matmul(wv)?;
    let q = x.matmul(wq)?;

    //let scores = q.matmul(&k.transpose(0, 1).unwrap())?;
    let scores = q.matmul(&k.t().unwrap())?;
    //let sq3 = Tensor::from_slice(&[3.0_f64.sqrt()], 1, device)?; // ver 1
    let sq3 = Tensor::from_slice(&[30.0_f64], 1, device)?;
    let scores = scores.broadcast_div(&sq3)?;
    let scores = softmax(&scores);
    Ok(scores.matmul(&v)?)
}
fn pv(prefix:&str, tensor:&Tensor) {
    match tensor.rank() {
        1 => println!("{}: {:.2?}",prefix, tensor.to_vec1::<f64>().unwrap()),
        2 => println!("{}: {:.2?}",prefix, tensor.to_vec2::<f64>().unwrap()),
        3 => println!("{}: {:.2?}",prefix, tensor.to_vec3::<f64>().unwrap()),
        x => println!("Too big Tensor size:{:?}",x)
    }

}
fn softmax(x: &Tensor) -> Tensor {
    candle_nn::ops::softmax(x, 1).unwrap()
}
#[allow(dead_code)]
fn prepare(embedding: &Tensor, wk1: &Tensor, wv1: &Tensor, wq1: &Tensor, device: &Device) -> Result<(), Box<dyn Error>> {
    let k1 = &embedding.matmul(wk1)?;
    let v1 = &embedding.matmul(wv1)?;
    let q1 = &embedding.matmul(wq1)?;

    println!("k1:{:?}", k1.to_vec2::<f64>());
    println!("v1:{:?}", v1.to_vec2::<f64>());
    println!("q1:{:?}", q1.to_vec2::<f64>());

    //let scores1 = &q1.matmul(&k1.transpose(0, 1).unwrap())?;
    let scores1 = &q1.matmul(&k1.t().unwrap())?;
    println!("scores1:{:.2?}", scores1.to_vec2::<f64>()?);

    let v3 = 3.0_f64.sqrt();
    let sq3 = Tensor::from_slice(&[v3], 1, device)?;
    let sqrt3 = scores1.broadcast_div(&sq3)?;
    let sm = softmax(&sqrt3);
    println!("sm:{:.2?}", sm.to_vec2::<f64>()?);
    let a1 = sm.matmul(v1)?;
    println!("a1:{:.2?}", &a1.to_vec2::<f64>()?);
    Ok(())
}
