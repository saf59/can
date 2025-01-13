use medius_meta::Meta;
use medius_model::test_all;
use std::path::Path;

pub fn main() -> anyhow::Result<()> {
    let mut meta = Meta::load_default();
    meta.train_part = 1.0;
    let base: &Path = "./data".as_ref();
    let datapath = base.join(meta.data_name());
    match test_all(datapath, &mut meta) {
        Ok(test_accuracy) => {
            println!("{:5.6?}", test_accuracy);
        }
        Err(e) => println!("{:?}",e)
    }
    Ok(())
}