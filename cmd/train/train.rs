use clap::Parser;
use medius_meta::{Activation, AlgType, BufSize, DataType, Meta, ModelType, Scale};
use medius_model::training_loop;
use std::path::Path;
use std::process;
use std::time::Instant;
/// Command line arguments for the training program
#[derive(Parser)]
struct Args {
    /// Show defaults and exit
    #[arg(short, default_value_t = false)]
    defaults: bool,
    /// Inputs
    #[arg(short)]
    n: Option<usize>,
    /// Algorithm
    #[arg(long, value_enum)]
    alg_type: Option<AlgType>,
    /// Data type
    #[arg(long, value_enum)]
    data_type: Option<DataType>,
    /// Buffer size
    #[arg(long, value_enum)]
    buff_size: Option<BufSize>,
    /// Use frequency scaling
    #[arg(long, value_enum)]
    scale: Option<Scale>,
    /// Use normalisation
    #[arg(long)]
    norm: Option<bool>,
    /// Model type
    #[arg(long, value_enum)]
    model_type: Option<ModelType>,
    /// Activation function, default is Relu
    #[arg(short, long, value_enum)]
    activation: Option<Activation>,
    /// Train epochs, default = 0
    #[arg(short, long)]
    epochs: Option<usize>,
    /// Train batch size
    #[arg(short, long)]
    batch_size: Option<usize>,
    /// The part of train data in x.csv and y.csv
    #[arg(long)]
    train_part: Option<f32>,
    #[arg(long)]
    learning_rate: Option<f64>,
    /// List of hidden layers, separated by commas
    #[arg(long)]
    hidden: Option<String>,
}

pub fn main() -> anyhow::Result<()> {
    let mut meta = meta()?;
    let base: &Path = "./data".as_ref();
    let datapath = base.join(meta.data_name());
    println!("{:?},{:?}",&datapath,&meta.model_name());
    let start = Instant::now();
    // Run training loop
    match training_loop(datapath, &mut meta) {
        Ok(_) => {
            println!("{:5.2?}", Instant::now().duration_since(start));
            meta.save();
        }
        Err(e) => println!("{e:?}!!!"),
    }
    Ok(())
}
/// Load metadata and parse command line arguments
#[allow(clippy::manual_map)]
fn meta() -> anyhow::Result<Meta> {
    let args = Args::parse();
    // All defaults are from Meta::default() except epochs
    let mut meta = Meta::load_default();
    // Override metadata with command line arguments if provided
    if let Some(n) = args.n {
        meta.n = n;
    }
    if let Some(alg_type) = args.alg_type {
        meta.alg_type = alg_type;
    }
    if let Some(data_type) = args.data_type {
        meta.data_type = data_type;
    } else {
        meta.data_type = DataType::None; // DEFAULT
    }
    if let Some(buff_size) = args.buff_size {
        meta.buff_size = buff_size;
    } else {
        meta.buff_size = BufSize::None; // DEFAULT
    }
    if let Some(scale) = args.scale {
        meta.scale = match scale {
            Scale::True => Some(true),
            Scale::False => Some(false),
            Scale::None => None
        };
    } else {
        meta.scale = None; // DEFAULT
    }

    meta.norm = if let Some(norm) = args.norm { Some(norm) } else { None };

    if let Some(model_type) = args.model_type {
        meta.model_type = model_type;
    }
    if let Some(activation) = args.activation {
        meta.activation = activation;
    }
    // else {  meta.activation = Activation::Relu  } // DEFAULT -> NOT NEEDED
    if let Some(epochs) = args.epochs {
        meta.epochs = epochs;
    } else {
        meta.epochs = 0
    } // DEFAULT
    if let Some(batch_size) = args.batch_size {
        meta.batch_size = batch_size;
    }
    if let Some(train_part) = args.train_part {
        meta.train_part = train_part;
    }
    if let Some(learning_rate) = args.learning_rate {
        meta.learning_rate = learning_rate;
    }
    if let Some(hidden) = args.hidden {
        meta.hidden = Some(hidden);
    } else if meta.hidden.is_none() {
        meta.hidden = Some("40,10".to_string());
    }
    if args.defaults {
        println!("{meta:#?}");
        process::exit(0);
    }
    if meta.alg_type == AlgType::HOM && meta.model_type != ModelType::Classification {
        return Err(anyhow::Error::msg(format!(
            "HOM is not implemented for {:#?} and model_type:{:#?}!",
            meta.alg_type, meta.model_type
        )));
    }
    Ok(meta)
}
