use clap::Parser;
use std::path::Path;
use std::process;
use std::time::Instant;

use medius_data::load_dir;
use medius_meta::{AlgType, BufSize, Meta, ModelType};
use medius_model::training_loop;

#[derive(Parser)]
struct Args {
    /// Show defaults and exit
    #[arg(short, default_value_t=false)]
    defaults: bool,
    /// Inputs
    #[arg(short)]
    n: Option<usize>,
    /// Algorithm
    #[arg(long, value_enum)]
    alg_type: Option<AlgType>,
    /// Buffer size
    #[arg(long, value_enum)]
    buff_size: Option<BufSize>,
    /// Use frequency scaling
    #[arg(long)]
    scaled_frequency: Option<bool>,
    /// Model type
    #[arg(long, value_enum)]
    model_type: Option<ModelType>,
    #[arg(long)]
    epochs: Option<usize>,
    /// The part of train data in x.csv and y.csv
    #[arg(long)]
    train_part: Option<f32>,
    #[arg(long)]
    learning_rate: Option<f64>,
    #[arg(long)]
    hidden0: Option<usize>,
    #[arg(long)]
    hidden1: Option<usize>,
}

pub fn main() -> anyhow::Result<()> {
    let meta = meta()?;
    let base: &Path = "./data".as_ref();
    let m = load_dir(base.join(&meta.data_name()), meta.train_part)?;
    print!("train-data: {:?}", m.train_data.shape());
    print!(", train-labels: {:?}", m.train_labels.shape());
    print!(", test-data: {:?}", m.test_data.shape());
    println!(", test-labels: {:?}", m.test_labels.shape());
    let start = Instant::now();
    let _ = training_loop(m, &meta);
    println!("{:5.2?}", Instant::now().duration_since(start));
    meta.save();
    Ok(())
}

fn meta() -> anyhow::Result<Meta> {
    let args = Args::parse();
    let mut meta = Meta::load_default();
    if let Some(epochs) = args.epochs {
        meta.epochs = epochs;
    }
    if let Some(train_part) = args.train_part {
        meta.train_part = train_part;
    }
    if let Some(learning_rate) = args.learning_rate {
        meta.learning_rate = learning_rate;
    }
    if let Some(hidden0) = args.hidden0 {
        meta.hidden0 = hidden0;
    }
    if let Some(hidden1) = args.hidden1 {
        meta.hidden1 = hidden1;
    }
    if args.defaults {
        println!("{:#?}",meta);
        process::exit(0);
    }
    Ok(meta)
}
