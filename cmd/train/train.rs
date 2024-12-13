use clap::Parser;
use std::path::Path;
use std::time::Instant;

use medius_data::load_dir;
use medius_meta::Meta;
use medius_model::training_loop;

#[derive(Parser)]
struct Args {
    /// The part of train data in x.csv and y.csv
    #[arg(long)]
    train_part: Option<f32>,

    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long)]
    epochs: Option<usize>,

    #[arg(long)]
    hidden0: Option<usize>,

    #[arg(long)]
    hidden1: Option<usize>,

    /// The ./data/[subDir] (or canonical dir) with two data files x.csv and y.csv
    #[arg(long, default_value_t = String::from("stat_n260Tlist"))]
    data: String,
}

pub fn main() -> anyhow::Result<()> {
    let meta = meta()?;
    let base: &Path = "./data".as_ref();
    let m = load_dir(base.join(&meta.data_name()), meta.train_part)?;
    print!("train-data: {:?}", m.train_data.shape());
    print!(", train-labels: {:?}", m.train_labels.shape());
    print!(", test-data: {:?}", m.test_data.shape());
    println!(", test-labels: {:?}", m.test_labels.shape());
    //let model = meta.model_name();
    let start = Instant::now();
    let _ = training_loop(m, &meta);
    println!("{:5.2?}", Instant::now().duration_since(start));
    Ok(())
}

fn meta() -> anyhow::Result<Meta> {
    let args = Args::parse();
    let mut meta = Meta::default();
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
    Ok(meta)
}
