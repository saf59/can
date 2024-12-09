use clap::Parser;
use std::path::Path;
use std::time::Instant;

use medius_data::load_dir;
use medius_model::{training_loop, TrainingArgs};

#[derive(Parser)]
struct Args {
    /// The part of train data in x.csv and y.csv
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

    /// The ./data/[subDir] (or canonical dir) with two data files x.csv and y.csv
    #[arg(long, default_value_t = String::from("stat_n260Tlist"))]
    data: String,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let base: &Path = "./data".as_ref();
    let m = load_dir(base.join(&args.data), args.train_part)?;
    print!("train-data: {:?}", m.train_data.shape());
    print!(", train-labels: {:?}", m.train_labels.shape());
    print!(", test-data: {:?}", m.test_data.shape());
    println!(", test-labels: {:?}", m.test_labels.shape());
    let model = format!(
        "./models/{:}_{:?}_{:?}.safetensors",
        args.data, args.hidden0, args.hidden1
    );
    let training_args = TrainingArgs {
        model,
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        hidden0: args.hidden0,
        hidden1: args.hidden1,
    };
    let start = Instant::now();
    let _ = training_loop(m, &training_args);
    println!("{:5.2?}", Instant::now().duration_since(start));
    Ok(())
}
