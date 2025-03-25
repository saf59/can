use candle_core::{Device, Tensor};
use std::error::Error;
use std::time::Instant;

// Как работают трансформеры: разбираем математику / Хабр
// https://habr.com/ru/articles/785474/
fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::cuda_if_available(0)?;
    let start = Instant::now();
    let _ = action(&device);
    println!(
        "{:?}, Cuda:{:?}",
        Instant::now().duration_since(start),
        &device.is_cuda()
    );
    Ok(())
}

fn action(device: &Device) -> Result<(), Box<dyn Error>> {
    // input:
    // Hello -> [1,2,3,4]
    // World -> [2,3,4,5]
    // positioning PE(pos,2i) = sin(pos/10000^(2i/d_model)), PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    // «Hello» -> [0, 1, 0, 1]
    // «World» -> [0.84, 0.99, 0, 1]
    // Для первого слова и позиции 1 = 1:
    // i = 1 (нечётный): PE(0,1) = cos(0 / 10000^(2*1 / 4)) = cos(0) = 1
    // Для второго слова и позиции 1 = 0.99:
    // i = 1 (нечётный): PE(1,1) = cos(1 / 10000^(2*1 / 4)) = cos(1 / 10000^0.5) ≈ cos(0.01) ≈ 0.99
    // 2. 2 input vectors of size 4 + position embedding result:
    // «Hello» = [1,2,3,4] + [0, 1, 0, 1] = [1, 3, 3, 5]
    // «World» = [2,3,4,5] + [0.84, 0.99, 0, 1] = [2.84, 3.99, 4, 6]
    let embedding = Tensor::from_slice(&[1., 3., 3., 5., 2.84, 3.99, 4., 6.], (2, 4), device)?;

    // Каждая матрица Wk,Wv,Wq будет иметь размер 4x3.
    // 3 выбрано произвольно, но использование слишком малого размера внимания уменьшит точность модели
    // Каждая матрица будет преобразовывать четырёхмерные эмбеддинги
    // в трёхмерные ключи (K), значения (V) и запросы (Q).
    let (wk1, wv1, wq1) = kvq1(device)?; // статичечкий пример из статьи
    let (wk2, wv2, wq2) = kvq2(device)?;

    // Only for debug && print
    prepare(&embedding, &wk1, &wv1, &wq1, device)?;

    let attention1 = attention(&embedding, &wk1, &wv1, &wq1, device)?;
    let attention2 = attention(&embedding, &wk2, &wv2, &wq2, device)?;
    pv("attention1", &attention1);
    pv("attention2", &attention2);
    // 4.3.5  attentions = np.concatenate([attention1, attention2], axis=1)
    // Multi-head(X) = Concat(head₁, ..., headₙ)⋅W⁰
    let attentions = Tensor::cat(&[attention1, attention2], 1)?;
    pv("attentions", &attentions);
    let attention = attentions.matmul(&w0(device))?;
    // размерность attention теперь такая же, как у входных данных
    pv("attention", &attention);
    // дальше 5. Слой с прямой связью FFN
    // Первый линейный слой с прямой связью принимает входные данные и расширяет их, например до 8
    // затем Relu
    // Второй линейный слой с прямой связью принимает данные из предыдущего слоя и сжимает их до исходной размерности
    // relu(attention.dot(W1) + b1).dot(W2) + b2
    // где W1(4,8) и W2(8,4) - матрицы весов, а b1(8) и b2(4) - векторы смещения.
    // Далее LayerNorm
    // Z(x) = LayerNorm(x + Attention(x))
    // FFN(x) = ReLU(x*W1 + b1)*W2 + b2
    // Encoder(x) = LayerNorm(Z(x) + FFN(Z(x) + x))
    // И так N раз.
    // ВСЕ
    Ok(())
}
// https://telegra.ph/Transformer-09-13-15
// Attention(Q,K,V) = softmax(QKᵀ/√dₖ)⋅V
// Задача - по Q и K понять насколько насколько документ релевантен (какой  V выбирать).
fn attention(
    x: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wq: &Tensor,
    device: &Device,
) -> Result<Tensor, Box<dyn Error>> {
    // Q - запрос, K - краткая сводка по документу и V - сам документ.
    // Получим матрицы Q, K и V путем перемножения X (входных данных) на соответствующие
    // матрицы весов  Wq, Wk и Wv соответственно:
    let k = x.matmul(wk)?;
    let v = x.matmul(wv)?;
    let q = x.matmul(wq)?;

    //let scores = q.matmul(&k.transpose(0, 1).unwrap())?;
    let scores = q.matmul(&k.t().unwrap())?;
    // 4.3.2  scores = np.dot(q, k.T) / np.sqrt(d_k) // но только для больших размерностей
    // let sq3 = Tensor::from_slice(&[3.0_f64.sqrt()], 1, device)?;
    // если размерность маленькая, то пример покажет кривой результат, поэтому
    // в статье заменено на scores / 30
    let sq3 = Tensor::from_slice(&[30.0_f64], 1, device)?;
    let scores = scores.broadcast_div(&sq3)?;
    let scores = softmax(&scores);
    Ok(scores.matmul(&v)?)
}
fn pv(prefix: &str, tensor: &Tensor) {
    match tensor.rank() {
        1 => println!("{}: {:.2?}", prefix, tensor.to_vec1::<f64>().unwrap()),
        2 => println!("{}: {:.2?}", prefix, tensor.to_vec2::<f64>().unwrap()),
        3 => println!("{}: {:.2?}", prefix, tensor.to_vec3::<f64>().unwrap()),
        x => println!("Too big Tensor size:{:?}", x),
    }
}
fn softmax(x: &Tensor) -> Tensor {
    candle_nn::ops::softmax(x, 1).unwrap()
}
#[allow(dead_code)]
fn prepare(
    embedding: &Tensor,
    wk1: &Tensor,
    wv1: &Tensor,
    wq1: &Tensor,
    device: &Device,
) -> Result<(), Box<dyn Error>> {
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

#[rustfmt::skip]
fn kvq1(device: &Device) -> Result<(Tensor, Tensor, Tensor), Box<dyn Error>> {
    let wk1 = Tensor::from_slice(&[1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
        (4, 3),device)?;
    let wv1 = Tensor::from_slice(&[0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0.],
        (4, 3),device)?;
    let wq1 = Tensor::from_slice(&[0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0.],
        (4, 3),device)?;
    Ok((wk1, wv1, wq1))
}
#[rustfmt::skip]
fn kvq2(device: &Device) -> Result<(Tensor, Tensor, Tensor), Box<dyn Error>> {
    let wk2 = Tensor::from_slice(&[0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 1., 0.],
        (4, 3),device)?;
    let wv2 = Tensor::from_slice(&[1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],
        (4, 3),device)?;
    let wq2 = Tensor::from_slice(&[1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1.],
        (4, 3),device)?;
    Ok((wk2, wv2, wq2))
}
#[rustfmt::skip]
fn w0(device: &Device) -> Tensor {
    Tensor::from_slice(&[
        0.79445237,0.1081456,0.27411536,0.78394531,
        0.29081936,-0.36187258,-0.32312791,-0.48530339,
        -0.36702934,-0.76471963,-0.88058366,-1.73713022,
        -0.02305587,-0.64315981,-0.68306653,-1.25393866,
        0.29077448,-0.04121674,0.01509932,0.13149906,
        0.57451867,-0.08895355,0.02190485,0.24535932,],(6, 4),device).unwrap()
}
