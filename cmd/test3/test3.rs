use clap::Parser;
use medius_meta::{Accuracy, Meta};
use medius_model::test_all;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// Show loss. Default - show accuracy when different is less than 0.1
    #[arg(long, value_enum, default_value_t = Accuracy::Percent)]
    accuracy: Accuracy,
    /// For all models. Default - only for current model
    #[arg(long, default_value_t = false)]
    all: bool,
}
pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut meta = Meta::load_default();
    meta.train_part = 1.0;
    let accuracy = args.accuracy;
    if args.all {
        let path = Path::new("./models");
        let subdirs = list_subdirectories(path)?;
        for subdir in subdirs {
            let mut meta = Meta::load(&path.join(&subdir))?;
            let start = Instant::now();
            let details = meta.name_out();
            match test_one(&mut meta, accuracy.clone()) {
                Ok(test_accuracy) => {
                    let end = Instant::now();
                    let elapsed = end.duration_since(start).as_secs_f32() * 1000.0;
                    if accuracy == Accuracy::Loss {
                        println!("{details},{test_accuracy},{elapsed:5.2?}");
                    } else {
                        println!("{details},{test_accuracy:5.2?}%,{elapsed:5.2?}");
                    }
                }
                Err(e) => println!("{e:?}"),
            }
        }
    } else {
        match test_one(&mut meta, accuracy.clone()) {
            Ok(test_accuracy) => {
                if accuracy == Accuracy::Loss {
                    println!("{test_accuracy:5.7?}");
                } else {
                    println!("{test_accuracy:5.2?}%" );
                }
            }
            Err(e) => println!("{e:?}"),
        }
    }
    Ok(())
}
fn test_one(meta: &mut Meta, accuracy: Accuracy) -> anyhow::Result<f32> {
    let base: &Path = "./data".as_ref();
    let datapath = base.join(meta.data_name());
    test_all(datapath, meta, accuracy)
}
fn list_subdirectories(path: &Path) -> std::io::Result<Vec<String>> {
    let mut subdirs = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = path.file_name() {
                if let Some(name_str) = name.to_str() {
                    subdirs.push(name_str.to_string());
                }
            }
        }
    }
    Ok(subdirs)
}
