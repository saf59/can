use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{
    loss, ops, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, SGD,
};
use medius_data::{load_dir, print_dataset_info, Dataset};
pub(crate) use medius_meta::{Activation, Meta, ModelType::{Classification, Regression}, DEFAULT_VM};
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
            layers.push(candle_nn::linear(prev_size, hidden_size, vs.pp(format!("ln{}", i +1)))?);
            prev_size = hidden_size;
        }

        // Add output layer
        layers.push(candle_nn::linear(prev_size, outputs, vs.pp(format!("ln{}", hidden.len() +1)))?);

        Ok(Self {
            layers,
            activation,
        })
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
    println!("Device:{:?}, {:?}", dev, &meta.small());
    let dataset = load_dir(datapath, meta.train_part, dev)?;
    print_dataset_info(&dataset);
    let binding = meta.model_file();
    let model_path: &Path = binding.as_ref();
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

pub fn test_all(datapath: PathBuf, meta: &mut Meta) -> anyhow::Result<f32> {
    let dev = &Device::cuda_if_available(0)?;
    let dataset = load_dir(datapath, meta.train_part, dev)?;
    meta.outputs = dataset.classes(meta.model_type == Regression);
    let (_varmap, model) = get_model(dev, meta, false, &fill_from_file)?;
    let test_data = dataset.test_data.to_device(dev)?;
    let test_accuracy = match meta.model_type {
        Classification => {
            let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(dev)?;
            test_classification(&model, &test_data, &test_labels)
        }
        Regression => {
            let test_labels = dataset
                .train_labels
                .to_dtype(DType::F32)
                .unwrap()
                .mul(-0.1)?
                .to_device(dev)?;
            test_regression(&model, &test_data, &test_labels)
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
    let mut opt = SGD::new(varmap.all_vars(), meta.learning_rate)?;
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
        let test_accuracy = test_classification(model, &test_data, &test_labels)?;
        print!(
            "{epoch:4} train loss: {loss:8.6} test acc: {:5.3}% \r",
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
) -> anyhow::Result<f32> {
    let chunk_size:usize = train_data.dims()[0] / batch_size;
    let train_data_chunks = train_data.chunk(chunk_size,0)?;
    let train_labels_chunks = train_labels.chunk(chunk_size,0)?;

    let mut f_loss = 0f32;
    for (data, labels) in train_data_chunks.into_iter().zip(train_labels_chunks) {
        let logits = &model.forward(&data)?;
        let log_sm = ops::log_softmax(logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &labels)?;
        opt.backward_step(&loss)?;
        f_loss = loss.to_scalar::<f32>()?;
    }
    Ok(f_loss)
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
    for epoch in 1..=meta.epochs {
        let loss = match meta.batch_size {
            1 => train_regression_epoch(&model, &mut opt, &train_data, &train_labels),
            _ => train_regression_batches(&model, &mut opt, &train_data, &train_labels,
                                          meta.batch_size),
        }?;
        let test_accuracy = test_regression(model, &test_data, &test_labels)?;
        print!(
            "{epoch:6} train loss: {loss:8.7} test acc: {:8.7}% \r",
            100. * test_accuracy
        );
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
) -> anyhow::Result<f32> {
    let chunk_size:usize = train_data.dims()[0] / batch_size;
    let train_data_chunks = train_data.chunk(chunk_size,0)?;
    let train_labels_chunks = train_labels.chunk(chunk_size,0)?;
    let mut f_loss = 0f32;
    for (data, labels) in train_data_chunks.into_iter().zip(train_labels_chunks) {
        let logits = &model.forward(&data).unwrap().flatten_to(1)?;
        let loss = logits.sub(&labels)?.sqr()?.mul(0.5).unwrap().mean(0)?;
        opt.backward_step(&loss)?;
        f_loss = loss.to_scalar::<f32>()?;
    }
    Ok(f_loss)
}
fn test_classification(model: &Mlp, data: &Tensor, labels: &Tensor) -> anyhow::Result<f32> {
    let logits = model.forward(data)?;
    let sum_ok = logits
        .argmax(D::Minus1)?
        .eq(labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
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
    let outputs = meta.outputs;
    // Extract meta.hidden to slice &[usize]
    let hidden:Vec<usize> = meta.hidden.as_ref().unwrap().split(',')
        .map(|x| x.parse().unwrap()).collect();
    if verbose {
        println!(
            "inputs:{:?},outputs:{:?},hidden:{:?}",
            inputs, outputs, hidden
        );
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