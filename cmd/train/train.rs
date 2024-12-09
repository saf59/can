use std::fs::create_dir_all;
use clap::{Parser, ValueEnum};
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_nn::Activation::Relu;
use medius_data::{load_dir, Dataset};

trait Model: Sized {
    fn new(vs: VarBuilder,inputs:usize,outputs:usize,hidden:&[usize]) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Model for Mlp {
    fn new(vs: VarBuilder,inputs:usize,outputs:usize,hidden:&[usize]) -> Result<Self> {
        let ln1 = candle_nn::linear(inputs, hidden[0], vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(hidden[0], hidden[1], vs.pp("ln2"))?;
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
    model:String,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
}

fn training_loop<M: Model>(m: Dataset, args: &TrainingArgs) -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_data = m.train_data.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let (len,inputs) = m.train_data.shape().dims2()?;
    println!("inputs:{:?},outputs:{:?}",&inputs,&m.labels);
    let hidden = &[40,10];
    let model = M::new(vs.clone(),inputs,m.labels,hidden)?;
    let model_path:&Path = Path::new::<Path>(args.model.as_ref());
    let _ = create_dir_all(model_path.parent().unwrap());
    println!("loading weights from {:}",model_path.to_string_lossy());
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
    println!("\nsaving trained weights in {:}",model_path.to_string_lossy());
    let _ = varmap.save(model_path);
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
    #[arg(long, default_value_t = 0.9)]
    train_part: f32,

    #[arg(long, default_value_t = 0.5)]
    learning_rate: f64,

    #[arg(long, default_value_t = 10)]
    epochs: usize,

    #[arg(long, default_value_t = 40)]
    hidden0: usize,

    #[arg(long, default_value_t = 10)]
    hidden1: usize,

    /// The file where to save the trained weights, in safetensors format.
    #[arg(long, default_value_t = String::from("stat_n260Tlist"))]
    data: String,

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
    let m = load_dir(base.join("stat_n260Tlist"), args.train_part)?;
    print!("train-data: {:?}", m.train_data.shape());
    print!(", train-labels: {:?}", m.train_labels.shape());
    print!(", test-data: {:?}", m.test_data.shape());
    println!(", test-labels: {:?}", m.test_labels.shape());
    let model = format!("./models/{:}_{:?}_{:?}.safetensors",args.data,args.hidden0,args.hidden1);
    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        model:model,
        load: args.load,
        save: args.save,
    };
    let start = Instant::now();
    let _ = training_loop::<Mlp>(m, &training_args);
    println!("{:5.2?}", Instant::now().duration_since(start));
    Ok(())
}
