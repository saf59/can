use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use medius_data::Dataset;
use medius_meta::{Meta, DEFAULT_VM};
use std::fs::create_dir_all;
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
pub fn training_loop(m: Dataset, meta: &Meta) -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_data = m.train_data.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    //let (_len, inputs) = m.train_data.shape().dims2()?;
    let binding = meta.model_file();
    let model_path: &Path = binding.as_ref();
    let mut varmap = VarMap::new();
    let model = get_model(&dev, &mut varmap, meta, false)?;

    let mut opt = candle_nn::SGD::new(varmap.all_vars(), meta.learning_rate)?;
    /*    let mut opt = candle_nn::AdamW::new(varmap.all_vars(),ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        })?;
    */
    let test_data = m.test_data.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    for epoch in 1..meta.epochs {
        let logits = model.forward(&train_data)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        opt.backward_step(&loss)?;

        let test_logits = model.forward(&test_data)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        print!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}% \r",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    println!(
        "\nsaving trained weights to {:}",
        model_path.to_string_lossy()
    );
    let _ = varmap.save(model_path);
    let _ = varmap.save(DEFAULT_VM);
    Ok(())
}

pub fn get_model(
    dev: &Device,
    varmap: &mut VarMap,
    meta: &Meta,
    verbose: bool,
) -> anyhow::Result<Mlp> {
    //let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(varmap, DType::F32, dev);
    let labels = 5;
    let inputs = meta.n;
    let hidden0 = meta.hidden0;
    let hidden1 = meta.hidden1;
    let binding = meta.model_file();
    let model_path: &Path = binding.as_ref();

    if verbose {
        println!(
            "inputs:{:?},outputs:{:?},hidden:[{:?},{:?}]",
            inputs, labels, meta.hidden0, meta.hidden1
        );
    }
    let model = Mlp::new(vs.clone(), inputs, labels, hidden0, hidden1)?;
    let _ = create_dir_all(model_path.parent().unwrap());
    if verbose {
        println!("loading weights from {:}", model_path.to_string_lossy());
    }
    if model_path.exists() {
        varmap.load(model_path)?;
    }
    Ok(model)
}
