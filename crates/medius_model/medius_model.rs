use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use medius_data::{load_dir, print_dataset_info, Dataset};
use medius_meta::Accuracy;
pub(crate) use medius_meta::{
    Activation, Meta,
    ModelType::{Classification, Regression},
    DEFAULT_VM,
};
use std::fs::create_dir_all;
use std::ops::Mul;
use std::path::{Path, PathBuf};

pub trait Model: Sized {
    fn new(
        vs: VarBuilder,
        inputs: usize,
        outputs: usize,
        hidden: &[usize],
        activation: Activation,
    ) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

pub struct Mlp {
    layers: Vec<Linear>,
    activation: Activation,
}

impl Model for Mlp {
    fn new(
        vs: VarBuilder,
        inputs: usize,
        outputs: usize,
        hidden: &[usize],
        activation: Activation,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut prev_size = inputs;

        // Add hidden layers
        for (i, &hidden_size) in hidden.iter().enumerate() {
            // each layer must have unique name like ln1, ln2, ln3...
            layers.push(candle_nn::linear(
                prev_size,
                hidden_size,
                vs.pp(format!("ln{}", i + 1)),
            )?);
            prev_size = hidden_size;
        }

        // Add output layer
        layers.push(candle_nn::linear(
            prev_size,
            outputs,
            vs.pp(format!("ln{}", hidden.len() + 1)),
        )?);

        Ok(Self { layers, activation })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        // Apply all layers except the last one with activation
        for layer in &self.layers[..self.layers.len() - 1] {
            x = layer.forward(&x)?;
            x = self.activation.forward(&x)?;
        }
        // Apply the last layer without activation
        self.layers.last().unwrap().forward(&x)
    }
}

pub fn training_loop(datapath: PathBuf, meta: &mut Meta) -> anyhow::Result<()> {
    let dev = &Device::cuda_if_available(0)?;
    println!("Device:{:?}, {:?}, {:?}", dev, &meta.small() , &datapath);
    let dataset = load_dir(datapath, meta.train_part, dev)?;
    print_dataset_info(&dataset);
    let binding = meta.model_file();
    let model_path: &Path = binding.as_ref();
    let _ = check_hidden(meta.hidden.as_ref());
    meta.outputs = dataset.classes(meta.model_type == Regression);
    let (varmap, model) = get_model(dev, meta, false, &fill_from_file)?;
    match meta.model_type {
        Classification => train_classification(dataset, meta, dev, &varmap, &model),
        Regression => train_regression(dataset, meta, dev, &varmap, &model),
    }?;
    println!(
        "\nsaving trained weights to {:}",
        model_path.to_string_lossy()
    );
    let _ = varmap.save(model_path);
    let _ = varmap.save(DEFAULT_VM);
    Ok(())
}

pub fn test_all(datapath: PathBuf, meta: &mut Meta, accuracy: Accuracy) -> anyhow::Result<f32> {
    let dev = &Device::cuda_if_available(0)?;
    let dataset = load_dir(datapath, meta.train_part, dev)?;
    meta.outputs = dataset.classes(meta.model_type == Regression);
    let (_varmap, model) = get_model(dev, meta, false, &fill_from_file)?;
    let test_data = dataset.test_data.to_device(dev)?;
    let test_accuracy = match meta.model_type {
        Classification => {
            let test_labels = dataset.test_labels.to_dtype(DType::U32)?;
            //println!("{:?}", test_labels.eq(0.0)?.to_dtype(DType::F32).unwrap().elem_count());
            test_classification(&model, &test_data, &test_labels, accuracy)
        }
        Regression => {
            let test_labels = labels_to_wp(&dataset.test_labels, dev);
            test_regression(&model, &test_data, &test_labels, accuracy)
        }
    }?;
    Ok(test_accuracy)
}
pub fn test_less01(datapath: PathBuf, meta: &mut Meta) -> anyhow::Result<f32> {
    let dev = &Device::cuda_if_available(0)?;
    let dataset = load_dir(datapath, meta.train_part, dev)?;
    meta.outputs = dataset.classes(meta.model_type == Regression);
    let (_varmap, model) = get_model(dev, meta, false, &fill_from_file)?;
    let test_data = dataset.test_data.to_device(dev)?;
    let test_accuracy = match meta.model_type {
        Classification => {
            let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(dev)?;
            test_classification(&model, &test_data, &test_labels, Accuracy::Percent01)
        }
        Regression => {
            let test_labels = labels_to_wp(&dataset.test_labels, dev);
            test_regression(&model, &test_data, &test_labels, Accuracy::Percent01)
        }
    }?;
    Ok(test_accuracy)
}

// Rest of the training and testing functions remain largely unchanged, just updating types
fn train_classification(
    m: Dataset,
    meta: &mut Meta,
    dev: &Device,
    varmap: &VarMap,
    model: &Mlp,
) -> anyhow::Result<()> {
    //let mut opt = SGD::new(varmap.all_vars(), meta.learning_rate)?;
    let mut opt = adamw(meta, varmap)?;
    let train_data = m.train_data.to_device(dev)?;
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(dev)?;
    let test_data = m.test_data.to_device(dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(dev)?;
    for epoch in 1..=meta.epochs {
        let loss = match meta.batch_size {
            1 => train_classification_epoch(&model, &mut opt, &train_data, &train_labels),
            _ => train_classification_batches(
                &model,
                &mut opt,
                &train_data,
                &train_labels,
                meta.batch_size,
            ),
        }?;
        let test_accuracy = test_classification(model, &test_data, &test_labels, Accuracy::Percent01)?;
        print!("{epoch:4} train loss: {loss:8.7} test acc: {test_accuracy:5.2}%       \r");
    }
    Ok(())
}
fn train_classification_epoch(
    model: &&Mlp,
    opt: &mut impl Optimizer,
    train_data: &Tensor,
    train_labels: &Tensor,
) -> anyhow::Result<f32> {
    let logits = &model.forward(train_data)?;
    let log_sm = ops::log_softmax(logits, D::Minus1)?;
    let loss = loss::nll(&log_sm, train_labels)?;
    opt.backward_step(&loss)?;
    Ok(loss.to_scalar::<f32>()?)
}
fn train_classification_batches(
    model: &&Mlp,
    opt: &mut impl Optimizer,
    train_data: &Tensor,
    train_labels: &Tensor,
    batch_size: usize,
) -> anyhow::Result<f32> {
    let chunks: usize = train_data.dims()[0] / batch_size;
    let train_data_chunks = train_data.chunk(chunks, 0)?;
    let train_labels_chunks = train_labels.chunk(chunks, 0)?;
    let len = train_data_chunks.len() as f32;
    let mut total_loss = 0f32;
    // Iterate over chunks of data and labels
    for (data, labels) in train_data_chunks.into_iter().zip(train_labels_chunks) {
        let logits = &model.forward(&data)?;
        let log_sm = ops::log_softmax(logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &labels)?;
        opt.backward_step(&loss)?;
        total_loss += loss.to_scalar::<f32>()?;
    }
    Ok(total_loss / len)
}
fn train_regression(
    m: Dataset,
    meta: &mut Meta,
    dev: &Device,
    varmap: &VarMap,
    model: &Mlp,
) -> anyhow::Result<()> {
    let mut opt = adamw(meta, varmap)?;
    let train_data = m.train_data.to_device(dev)?;
    let train_labels = labels_to_wp(&m.train_labels, dev);
    let test_data = m.test_data.to_device(dev)?;
    let test_labels = labels_to_wp(&m.test_labels, dev);
    for epoch in 1..=meta.epochs {
        let loss = match meta.batch_size {
            1 => train_regression_epoch(&model, &mut opt, &train_data, &train_labels),
            _ => train_regression_batches(
                &model,
                &mut opt,
                &train_data,
                &train_labels,
                meta.batch_size,
            ),
        }?;
        let test_accuracy = test_regression(model, &test_data, &test_labels, Accuracy::Percent01)?;
        print!("{epoch:6} train loss: {loss:8.7} test acc: {test_accuracy:5.2}%       \r");
    }
    Ok(())
}

fn adamw(meta: &mut Meta, varmap: &VarMap) -> Result<AdamW> {
    AdamW::new(
        varmap.all_vars(),
        ParamsAdamW { // ювелирной настройки: β₁=0.9, β₂=0.95, eps=1e-15
            lr: meta.learning_rate,
            ..Default::default()
        },
    )
}

fn labels_to_wp(labels: &Tensor, dev: &Device) -> Tensor {
    labels
        .to_dtype(DType::F32)
        .unwrap()
        .mul(-0.1)
        .unwrap()
        .to_device(dev)
        .unwrap()
}
fn train_regression_epoch(
    model: &&Mlp,
    opt: &mut AdamW,
    train_data: &Tensor,
    train_labels: &Tensor,
) -> anyhow::Result<f32> {
    let logits = &model.forward(train_data).unwrap().flatten_to(1)?;
    let loss = logits.sub(train_labels)?.sqr()?.mul(0.5).unwrap().mean(0)?;
    opt.backward_step(&loss)?;
    Ok(loss.to_scalar::<f32>()?)
}

fn train_regression_batches(
    model: &&Mlp,
    opt: &mut AdamW,
    train_data: &Tensor,
    train_labels: &Tensor,
    batch_size: usize,
) -> anyhow::Result<f32> {
    let chunks: usize = train_data.dims()[0] / batch_size;
    let train_data_chunks = train_data.chunk(chunks, 0)?;
    let train_labels_chunks = train_labels.chunk(chunks, 0)?;
    let len = train_data_chunks.len() as f32;
    let mut total_loss = 0f32;
    for (data, labels) in train_data_chunks.into_iter().zip(train_labels_chunks) {
        let logits = &model.forward(&data).unwrap().flatten_to(1)?;
        let loss = logits.sub(&labels)?.sqr()?.mul(0.5).unwrap().mean(0)?;
        opt.backward_step(&loss)?;
        total_loss += loss.to_scalar::<f32>()?;
    }
    Ok(total_loss / len)
}

fn test_classification(
    model: &Mlp,
    data: &Tensor,
    labels: &Tensor,
    accuracy: Accuracy,
) -> anyhow::Result<f32> {
    let logits = model.forward(data)?
        .argmax(D::Minus1)?;
    if accuracy == Accuracy::Loss {
        let logits01 = logits.to_dtype(DType::F32)
            .unwrap()
            .mul(-0.1)?; // 0,1,2.... * 0.1
        let labels01 = labels.to_dtype(DType::F32).unwrap().mul(-0.1)?;
        to_loss(&labels01, logits01)
    } else {
        to_percent(labels, &logits)
    }
}
fn test_regression(
    model: &Mlp,
    data: &Tensor,
    labels: &Tensor,
    accuracy: Accuracy,
) -> anyhow::Result<f32> {
    let logits = model.forward(data).unwrap().flatten_to(1)?;
    if accuracy == Accuracy::Loss {
        to_loss(labels, logits)
    } else {
        to_01(labels, logits)
    }
}

fn to_loss(labels: &Tensor, logits: Tensor) -> anyhow::Result<f32> {
    let loss = logits.sub(labels)?.sqr()?.mul(0.5).unwrap().mean(0)?;
    let accuracy = loss.to_scalar::<f32>()?;
    Ok(accuracy)
}
fn to_percent(labels: &Tensor, logits: &Tensor) -> anyhow::Result<f32> {
    //println!("labels:{:?}, logits:{:?}", labels.dtype(), logits.dtype());
    //let logits = logits.to_dtype(DType::U8)?;
    let sum_ok = logits
        .eq(labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let percentage = (sum_ok / labels.dims1()? as f32) * 100.0;
    Ok(percentage)
}

// candle-examples/examples/mnist-training/main.rs
fn to_01(labels: &Tensor, logits: Tensor) -> anyhow::Result<f32> {
    let diff = logits.sub(labels)?.abs()?;
    let within_threshold = diff.le(0.1)?.to_dtype(DType::F32)?;

    let count_within_threshold = within_threshold.sum_all()?.to_scalar::<f32>()?;
    let total_count = labels.dims1()? as f32;
    let percentage = (count_within_threshold / total_count) * 100.0;
    Ok(percentage)
}

pub fn get_model(
    dev: &Device,
    meta: &Meta,
    verbose: bool,
    fill: &dyn Fn(&Meta, bool, &mut VarMap) -> anyhow::Result<()>,
) -> anyhow::Result<(VarMap, Mlp)> {
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let inputs = meta.n;
    let outputs = meta.outputs;
    // Extract meta.hidden to slice &[usize]
    let hidden: Vec<usize> = meta
        .hidden
        .as_ref()
        .unwrap()
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    if verbose {
        println!("inputs:{inputs:?},outputs:{outputs:?},hidden:{hidden:?}", );
    }
    let model = Mlp::new(
        vs.clone(),
        inputs,
        outputs,
        &hidden,
        meta.activation.clone(),
    )?;
    fill(meta, verbose, &mut varmap)?;
    Ok((varmap, model))
}

fn check_hidden(hidden: Option<&String>) -> anyhow::Result<()> {
    let re = regex::Regex::new(r"^\d+(,\d+)*$").unwrap();
    if hidden.is_none() || hidden.unwrap().is_empty() || !re.is_match(hidden.unwrap()) {
        return Err(anyhow::anyhow!("hidden layers are not defined"));
    }
    Ok(())
}

pub fn fill_from_file(meta: &Meta, verbose: bool, varmap: &mut VarMap) -> anyhow::Result<()> {
    let binding = meta.model_file();
    let model_path: &Path = binding.as_ref();
    if model_path.exists() {
        if verbose {
            println!("loading weights from {:}", model_path.to_string_lossy());
        }
        let _ = create_dir_all(model_path.parent().unwrap());
        varmap.load(model_path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_to_01_basic() {
        let dev = Device::Cpu;
        // Labels and logits are close, so all should be within threshold
        let labels = Tensor::from_vec(vec![0.0, 0.1, 0.2], (3,), &dev).unwrap();
        let logits = Tensor::from_vec(vec![0.05, 0.15, 0.18], (3,), &dev).unwrap();
        let result = to_01(&labels, logits).unwrap();
        // All diffs are <= 0.1, so expect 100%
        assert!((result - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_to_01_partial() {
        let dev = Device::Cpu;
        // Only one value is within threshold
        let labels = Tensor::from_vec(vec![0.0, 0.1, 0.2], (3,), &dev).unwrap();
        let logits = Tensor::from_vec(vec![0.52, 0.13, 0.54], (3,), &dev).unwrap();
        let result = to_01(&labels, logits).unwrap();
        // Only the second value matches, so expect 33.33%
        assert!((result - 33.3333).abs() < 1e-2);
    }
}
