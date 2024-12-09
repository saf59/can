#[cfg(feature = "accelerate")]
extern crate accelerate_src; // This should reach 91.5% accuracy.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::Path;
use clap::{Parser, ValueEnum};

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use medius_data::{load_dir, Dataset};

const IMAGE_DIM: usize = 260;
const LABELS: usize = 5;

trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Model for Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABELS, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
}

fn training_loop<M: Model>(
    m: Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_data.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = M::new(vs.clone())?;

    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        varmap.load(load)?
    } else {
        println!("loading weights from out");
        varmap.load("out")?
    }

    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;
    let test_images = m.test_data.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    for epoch in 1..args.epochs {
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        varmap.save(save)?
    } else {
        println!("saving trained weights in out");
        varmap.save("out")?
    }
    Ok(())
}

#[derive(ValueEnum, Clone)]
enum WhichModel {
    Linear,
    Mlp,
    Cnn,
}

#[derive(Parser)]
struct Args {

    // #[clap(value_enum, default_value_t = WhichModel::Linear)]
    // model: WhichModel,

    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// The file where to save the trained weights, in safetensors format.
    #[arg(long)]
    save: Option<String>,

    /// The file where to load the trained weights from, in safetensors format.
    #[arg(long)]
    load: Option<String>,

    // The directory where to load the dataset from, in ubyte format.
    // #[arg(long)]
    // local_mnist: Option<String>,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Load the dataset
/*    let m = if let Some(directory) = args.local_mnist {
        candle_datasets::vision::mnist::load_dir(directory)?
    } else {
        candle_datasets::vision::mnist::load()?
    };
*/
    let base: &Path = "./data".as_ref();
    let m =load_dir(&base.join("stat_n260Tlist"),0.9)?;
    println!("train-images: {:?}", m.train_data.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_data.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    let default_learning_rate = args.learning_rate.unwrap_or(0.05);
    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
        load: args.load,
        save: args.save,
    };
    training_loop::<Mlp>(m, &training_args)
}
