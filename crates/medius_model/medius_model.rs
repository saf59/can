use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use medius_data::Dataset;
use medius_meta::{Meta, ModelType, DEFAULT_VM};
use std::fs::create_dir_all;
use std::ops::Mul;
use std::path::Path;

pub trait Model: Sized {
    fn new(
        vs: VarBuilder,
        inputs: usize,
        outputs: usize,
        hidden0: usize,
        hidden1: usize,
    ) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

pub struct Mlp {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl Model for Mlp {
    fn new(
        vs: VarBuilder,
        inputs: usize,
        outputs: usize,
        hidden0: usize,
        hidden1: usize,
    ) -> Result<Self> {
        let ln1 = candle_nn::linear(inputs, hidden0, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(hidden0, hidden1, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(hidden1, outputs, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}
pub fn training_loop(m: Dataset, meta: &mut Meta) -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    let binding = meta.model_file();
    let model_path: &Path = binding.as_ref();
    meta.outputs = if meta.model_type== ModelType::Regression {1} else {m.labels};
    let (varmap, model) = get_model(&dev, meta, false,&fill_from_file)?;
    match meta.model_type {
        ModelType::Classification => train_classification(m, meta, &dev, &varmap, &model),
        ModelType::Regression => train_regression(m, meta, &dev, &varmap, &model)
    }?;
    println!("\nsaving trained weights to {:}", model_path.to_string_lossy());
    let _ = varmap.save(model_path);
    let _ = varmap.save(DEFAULT_VM); // only for include_bytes!("../.././models/model.safetensors");
    Ok(())
}

fn train_classification(m: Dataset, meta: &mut Meta, dev: &Device, varmap: &VarMap, model: &Mlp) -> anyhow::Result<()> {
    let mut opt = candle_nn::SGD::new(varmap.all_vars(), meta.learning_rate)?;
    let train_data = m.train_data.to_device(dev)?;
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(dev)?;
    let test_data = m.test_data.to_device(dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(dev)?;
    for epoch in 1..meta.epochs {
        let logits = &model.forward(&train_data)?;
        let log_sm = ops::log_softmax(logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        opt.backward_step(&loss)?;
        let test_accuracy = test_classification(model, &test_data, &test_labels)?;
        print!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}% \r",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    Ok(())
}
fn train_regression(m: Dataset, meta: &mut Meta, dev: &Device, varmap: &VarMap, model: &Mlp) -> anyhow::Result<()> {
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), ParamsAdamW {
        lr: meta.learning_rate,
        ..Default::default()
    })?;
    let train_data = m.train_data.to_device(dev)?;
    let train_labels = m.train_labels.to_dtype(DType::F32).unwrap().mul(-0.1)?.to_device(dev)?;
    let test_data = m.test_data.to_device(dev)?;
    let test_labels = m.test_labels.to_dtype(DType::F32).unwrap().mul(-0.1)?.to_device(dev)?;
    for epoch in 1..meta.epochs {
        let logits = &model.forward(&train_data).unwrap().flatten_to(1)?;
        let loss = logits.sub(&train_labels)?.sqr()?.mul(0.5).unwrap();
        let loss =loss.mean(0)?;
        opt.backward_step(&loss)?;
        let test_accuracy = test_regression(model, &test_data, &test_labels)?;
        print!(
            "{epoch:4} train loss: {:8.6} test acc: {:5.6}% \r",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    Ok(())
}


fn test_classification(model: &Mlp, test_data: &Tensor, test_labels: &Tensor) -> anyhow::Result<f32> {
    let test_logits = model.forward(test_data)?;
    let sum_ok = test_logits
        .argmax(D::Minus1)?
        .eq(test_labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let test_accuracy = sum_ok / test_labels.dims1()? as f32;
    Ok(test_accuracy)
}
fn test_regression(model: &Mlp, test_data: &Tensor, test_labels: &Tensor) -> anyhow::Result<f32> {
    let test_logits = model.forward(test_data).unwrap().flatten_to(1)?;
    let loss = test_logits.sub(test_labels)?.sqr()?.mul(0.5).unwrap().mean(0)?;
    let test_accuracy = loss.to_vec0::<f32>()?;
    Ok(test_accuracy)
}

pub fn get_model(
    dev: &Device,
    meta: &Meta,
    verbose: bool,
    f: &dyn Fn(&Meta, bool, &mut VarMap) -> anyhow::Result<()>
) -> anyhow::Result<(VarMap,Mlp)> {
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let inputs = meta.n;
    let hidden0 = meta.hidden0;
    let hidden1 = meta.hidden1;
    let outputs = meta.outputs;
    if verbose { println!("inputs:{:?},outputs:{:?},hidden:[{:?},{:?}]",
                 inputs, outputs, meta.hidden0, meta.hidden1); }
    let model = Mlp::new(vs.clone(), inputs, outputs, hidden0, hidden1)?;
    //fill_from_file(meta, verbose, &mut varmap)?;
    f(meta, verbose, &mut varmap)?;
    Ok((varmap,model))
}
/// Fill VarMap from file
pub fn fill_from_file(meta: &Meta, verbose: bool, varmap: &mut VarMap) -> anyhow::Result<()> {
    let binding = meta.model_file();
    let model_path: &Path = binding.as_ref();
    if model_path.exists() {
        if verbose { println!("loading weights from {:}", model_path.to_string_lossy()); }
        let _ = create_dir_all(model_path.parent().unwrap());
        varmap.load(model_path)?;
    }
    Ok(())
}
