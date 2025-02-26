use clap::Parser;
use medius_meta::Meta;
use medius_model::test_all;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// Show loss. Default - show accuracy when different is less than 0.1
    #[arg(long, default_value_t = false)]
    loss: bool,
    /// For all models. Default - only for current model
    #[arg(long, default_value_t = false)]
    all: bool,
}
pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut meta = Meta::load_default();
    meta.train_part = 1.0;
    let loss = args.loss;
    if args.all {
        let path = Path::new("./models");
        let subdirs = list_subdirectories(path)?;
        for subdir in subdirs {
            let meta = Meta::load(&path.join(&subdir));
            let start = Instant::now();
            match test_one(&mut meta.unwrap(),loss) {
                Ok(test_accuracy) => {
                    let end = Instant::now();
                    let elapsed = end.duration_since(start);
                    if args.loss {
                        println!("{},{},{}", subdir, test_accuracy, elapsed.as_secs_f32());
                    } else {
                        println!("{},{:5.2?}%,{}", subdir, test_accuracy,elapsed.as_secs_f32());
                    }
                }
                Err(e) => println!("{:?}", e),
            }
        }
    } else {
        match test_one(&mut meta,loss) {
            Ok(test_accuracy) => {
                if args.loss {
                    println!("{:5.7?}", test_accuracy);
                } else {
                    println!("{:5.2?}%", test_accuracy);
                }
            }
            Err(e) => println!("{:?}", e),
        }
    }
    Ok(())
}
fn test_one(meta: &mut Meta, loss: bool) -> anyhow::Result<f32> {
    let base: &Path = "./data".as_ref();
    let datapath = base.join(meta.data_name());
    test_all(datapath, meta, loss)
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
