use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, SGD};
use medius_data::{load_dir, print_dataset_info, Dataset};
use medius_meta::{Activation, Meta, ModelType, DEFAULT_VM};
use std::fs::create_dir_all;
use std::ops::Mul;
use std::path::{Path, PathBuf};

pub trait Model: Sized {
    fn new(
        vs: VarBuilder,
        inputs: usize,
        outputs: usize,
        hidden0: usize,
        hidden1: usize,
        activation: Activation
    ) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

pub struct Mlp {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
    activation: Activation
}
impl Model for Mlp {
    fn new(
        vs: VarBuilder,
        inputs: usize,
        outputs: usize,
        hidden0: usize,
        hidden1: usize,
        activation: Activation
    ) -> Result<Self> {
        let ln1 = candle_nn::linear(inputs, hidden0, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(hidden0, hidden1, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(hidden1, outputs, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3, activation })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs =  self.activation.forward(&xs)?;// xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        //let xs = xs.relu()?;
        let xs =  self.activation.forward(&xs)?;
        self.ln3.forward(&xs)
    }
}
pub fn training_loop(datapath:PathBuf, meta: &mut Meta) -> anyhow::Result<()> {
    let dev = &Device::cuda_if_available(0)?;
    println!("Device:{:?}",dev);
    let dataset = load_dir(datapath, meta.train_part,dev)?;
    print_dataset_info(&dataset);
    let binding = meta.model_file();
    let model_path: &Path = binding.as_ref();
    meta.outputs = if meta.model_type == ModelType::Regression { 1 } else { dataset.labels };
    let (varmap, model) = get_model(dev, meta, false, &fill_from_file)?;
    match meta.model_type {
        ModelType::Classification => train_classification(dataset, meta, dev, &varmap, &model),
        ModelType::Regression => train_regression(dataset, meta, dev, &varmap, &model),
    }?;
    println!(
        "\nsaving trained weights to {:}",
        model_path.to_string_lossy()
    );
    let _ = varmap.save(model_path);
    let _ = varmap.save(DEFAULT_VM); // only for include_bytes!("../.././models/model.safetensors");
    Ok(())
}

pub fn test_all(datapath:PathBuf, meta: &mut Meta) -> anyhow::Result<f32> {
    let dev = &Device::cuda_if_available(0)?;
    let dataset = load_dir(datapath, meta.train_part,dev)?;
    meta.outputs = if meta.model_type == ModelType::Regression { 1 } else { dataset.labels };
    let (_varmap, model) = get_model(dev, meta, false, &fill_from_file)?;
    let test_data = dataset.test_data.to_device(dev)?;
    let test_accuracy = match meta.model_type {
        ModelType::Classification => {
            let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(dev)?;
            test_classification(&model, &test_data, &test_labels)
        },
        ModelType::Regression => {
            let test_labels = dataset
                .train_labels
                .to_dtype(DType::F32)
                .unwrap()
                .mul(-0.1)?
                .to_device(dev)?;
            test_regression(&model, &test_data, &test_labels)
        },
    }?;
    Ok(test_accuracy)
}

fn train_classification(
    m: Dataset,
    meta: &mut Meta,
    dev: &Device,
    varmap: &VarMap,
    model: &Mlp,
) -> anyhow::Result<()> {
    let mut opt = SGD::new(varmap.all_vars(), meta.learning_rate)?;
    let train_data = m.train_data.to_device(dev)?;
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(dev)?;
    let test_data = m.test_data.to_device(dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(dev)?;
    let n_batches = train_data.dim(0)? / meta.batch_size;
    let batch_idxs = (0..n_batches).collect::<Vec<usize>>();
    for epoch in 1..=meta.epochs {
        let loss = match meta.batch_size {
            1 => train_classification_epoch(&model, &mut opt, &train_data, &train_labels),
            _ => train_classification_batches(
                &model,
                &mut opt,
                &train_data,
                &train_labels,
                meta.batch_size,
                n_batches,
                &batch_idxs,
            ),
        }?;
        let test_accuracy = test_classification(model, &test_data, &test_labels)?;
        print!(
            "{epoch:4} train loss: {loss:8.5} test acc: {:5.2}% \r",
            100. * test_accuracy
        );
    }
    Ok(())
}
fn train_classification_epoch(
    model: &&Mlp,
    opt: &mut SGD,
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
    opt: &mut SGD,
    train_data: &Tensor,
    train_labels: &Tensor,
    batch_size: usize,
    n_batches: usize,
    batch_idxs: &[usize],
) -> anyhow::Result<f32> {
    let mut sum_loss = 0f32;
    for batch_idx in batch_idxs.iter() {
        let data = train_data.narrow(0, batch_idx * batch_size, batch_size)?;
        let labels = train_labels.narrow(0, batch_idx * batch_size, batch_size)?;
        let logits = &model.forward(&data)?;
        let log_sm = ops::log_softmax(logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &labels)?;
        opt.backward_step(&loss)?;
        sum_loss += loss.to_scalar::<f32>()?;
    }
    Ok(sum_loss / n_batches as f32)
}
fn train_regression(
    m: Dataset,
    meta: &mut Meta,
    dev: &Device,
    varmap: &VarMap,
    model: &Mlp,
) -> anyhow::Result<()> {
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: meta.learning_rate,
            ..Default::default()
        },
    )?;
    let train_data = m.train_data.to_device(dev)?;
    let train_labels = m
        .train_labels
        .to_dtype(DType::F32)
        .unwrap()
        .mul(-0.1)?
        .to_device(dev)?;
    let test_data = m.test_data.to_device(dev)?;
    let test_labels = m
        .test_labels
        .to_dtype(DType::F32)
        .unwrap()
        .mul(-0.1)?
        .to_device(dev)?;
    let n_batches = train_data.dim(0)? / meta.batch_size;
    let batch_idxs = (0..n_batches).collect::<Vec<usize>>();
    for epoch in 1..=meta.epochs {
        let loss = match meta.batch_size {
            1 => train_regression_epoch(&model, &mut opt, &train_data, &train_labels),
            _ => train_regression_batches(
                &model,
                &mut opt,
                &train_data,
                &train_labels,
                meta.batch_size,
                n_batches,
                &batch_idxs,
            ),
        }?;
        let test_accuracy = test_regression(model, &test_data, &test_labels)?;
        print!("{epoch:5} train loss: {loss:8.6} test acc: {:5.6}% \r", 100. * test_accuracy);
    }
    Ok(())
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
    n_batches: usize,
    batch_idxs: &[usize],
) -> anyhow::Result<f32> {
    let mut sum_loss = 0f32;
    for batch_idx in batch_idxs.iter() {
        let data = train_data.narrow(0, batch_idx * batch_size, batch_size)?;
        let labels = train_labels.narrow(0, batch_idx * batch_size, batch_size)?;
        let logits = &model.forward(&data).unwrap().flatten_to(1)?;
        let loss = logits.sub(&labels)?.sqr()?.mul(0.5).unwrap().mean(0)?;
        opt.backward_step(&loss)?;
        sum_loss += loss.to_scalar::<f32>()?;
    }
    Ok(sum_loss / n_batches as f32)
}
fn test_classification(
    model: &Mlp,
    data: &Tensor,
    labels: &Tensor,
) -> anyhow::Result<f32> {
    let logits = model.forward(data)?;
    let sum_ok = logits.argmax(D::Minus1)?.eq(labels)?.to_dtype(DType::F32)?
        .sum_all()?.to_scalar::<f32>()?;
    let accuracy = sum_ok / labels.dims1()? as f32;
    Ok(accuracy)
}
fn test_regression(model: &Mlp, data: &Tensor, labels: &Tensor) -> anyhow::Result<f32> {
    let logits = model.forward(data).unwrap().flatten_to(1)?;
    let loss = logits.sub(labels)?.sqr()?.mul(0.5).unwrap().mean(0)?;
    let accuracy = loss.to_scalar::<f32>()?;
    Ok(accuracy)
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
    let hidden0 = meta.hidden0;
    let hidden1 = meta.hidden1;
    let outputs = meta.outputs;
    if verbose {
        println!(
            "inputs:{:?},outputs:{:?},hidden:[{:?},{:?}]",
            inputs, outputs, meta.hidden0, meta.hidden1
        );
    }
    let model = Mlp::new(vs.clone(), inputs, outputs, hidden0, hidden1, meta.activation.clone())?;
    fill(meta, verbose, &mut varmap)?;
    Ok((varmap, model))
}
/// Fill VarMap from file
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
    use crate::show_is_cuda;
    #[test]
    fn is_cuda() {
        show_is_cuda()
    }
}
pub fn show_is_cuda() {
    let device = Device::cuda_if_available(0).unwrap();
    println!("Device:{:?}",device);
}
