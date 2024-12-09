use std::fs::create_dir_all;
use std::path::Path;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use medius_data::Dataset;

trait Model: Sized {
    fn new(vs: VarBuilder, inputs: usize, hidden0: usize, hidden1: usize) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

pub struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Model for Mlp {
    fn new(vs: VarBuilder, inputs: usize, hidden0: usize, hidden1: usize) -> Result<Self> {
        let ln1 = candle_nn::linear(inputs, hidden0, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(hidden0, hidden1, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

pub struct TrainingArgs {
    pub model: String,
    pub learning_rate: f64,
    pub epochs: usize,
    pub hidden0: usize,
    pub hidden1: usize,
}

pub fn training_loop(m: Dataset, args: &TrainingArgs) -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_data = m.train_data.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let (_len, inputs) = m.train_data.shape().dims2()?;
    println!("inputs:{:?},outputs:{:?},hidden:[{:?},{:?}]", &inputs, &m.labels, args.hidden0, args.hidden1);
    let model = Mlp::new(vs.clone(), inputs, args.hidden0, args.hidden1)?;
    let model_path: &Path = Path::new::<Path>(args.model.as_ref());
    let _ = create_dir_all(model_path.parent().unwrap());
    println!("loading weights from {:}", model_path.to_string_lossy());
    if model_path.exists() {
        varmap.load(model_path)?;
    }
    let mut opt = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;
    /*    let mut opt = candle_nn::AdamW::new(varmap.all_vars(),ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        })?;
    */
    let test_data = m.test_data.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    for epoch in 1..args.epochs {
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
        "\nsaving trained weights in {:}",
        model_path.to_string_lossy()
    );
    let _ = varmap.save(model_path);
    Ok(())
}
