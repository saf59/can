use clap::ValueEnum;
use std::fmt::Debug;
use std::fs;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::string::ToString;
use utils::first_char;

pub const MODELS_DIR: &str = "./models";
pub const DEFAULT: &str = "./models/model.meta";
pub const DEFAULT_VM: &str = "./models/model.safetensors";
const META_NAME: &str = "model.meta";
const MODEL_NAME: &str = "model.safetensors";
#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq,Debug)]
pub struct Meta {
    // data && parse
    pub n: usize,
    pub alg_type: AlgType,
    pub buff_size: BufSize,
    pub scaled_frequency: bool,
    // model
    pub model_type: ModelType,
    #[serde(skip)]
    pub epochs: usize,
    pub learning_rate: f64,
    pub train_part: f32,
    pub hidden0: usize,
    pub hidden1: usize,
}
impl Default for Meta {
    fn default() -> Self {
        Self {
            // data && parse
            n: 260,
            alg_type: AlgType::Bin,
            buff_size: BufSize::Small,
            scaled_frequency: true,
            // model
            model_type: ModelType::Classification,
            epochs: 100,
            learning_rate: 0.5,
            train_part: 0.9,
            hidden0: 40,
            hidden1: 10,
        }
    }
}
#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum ModelType {
    Regression,
    Classification,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum AlgType {
    Bin,
    Mfcc,
    Stat,
}
#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum BufSize {
    Big = 65_536 * 2,
    Small = 65_536,
}
impl Meta {
    pub fn save(&self) {
        let file = self.meta_file();
        let parent = file.parent().expect("Parent dir.");
        let _ = create_dir_all(parent);
        let out_string = serde_yaml::to_string(&self).unwrap();
        fs::write(file, &out_string).expect("Unable to write meta file");
        fs::write(DEFAULT, out_string).expect("Unable to write default meta file");
    }

    pub fn load_default() ->Meta {
        if !(DEFAULT.as_ref() as &Path).exists() { return Meta::default();}
        let buf = fs::read(DEFAULT).unwrap();
        serde_yaml::from_slice(&buf).unwrap()
    }

    pub fn init() -> Meta {
        if !(DEFAULT.as_ref() as &Path).exists() {
            let bytes = include_bytes!("../../models/model.meta");
            let _ = create_dir_all(MODELS_DIR);
            fs::write(DEFAULT, bytes).expect("Unable to write default meta file");
        }
        let meta = Meta::load_default();
        let binding = &meta.model_file();
        let model_path: &Path = binding.as_ref();
        if !model_path.exists() {
            let bytes = include_bytes!("../../models/model.safetensors");
            let _ = create_dir_all(model_path.parent().unwrap());
            fs::write(model_path, bytes).expect("Unable to write default meta file");
        }
        meta
    }


    pub fn data_name(&self) -> String {
        let at = first_char(&self.alg_type);
        let sn = &self.n.to_string();
        let bs = first_char(&self.buff_size);
        let sf = if self.scaled_frequency { 'T' } else { 'F' };
        format!("{at}{sn}_{bs}{sf}")
    }
    pub fn model_name(&self) -> String {
        let mt = first_char(&self.model_type);
        let at = first_char(&self.alg_type);
        let sn = &self.n.to_string();
        let bs = first_char(&self.buff_size);
        let h0 = &self.hidden0.to_string();
        let h1 = &self.hidden1.to_string();
        let sf = if self.scaled_frequency { 'T' } else { 'F' };
        format!("{mt}_{h0}_{h1}_{at}{sn}_{bs}{sf}")
    }
    pub fn meta_file(&self) -> PathBuf {
        let name = &self.model_name();
        Self::named(name, META_NAME)
    }
    pub fn model_file(&self) -> PathBuf {
        let name = &self.model_name();
        Self::named(name, MODEL_NAME)
    }

    fn named(type_name: &str, file_name: &str) -> PathBuf {
        let dir: &Path = MODELS_DIR.as_ref();
        let dir = &dir.join(type_name);
        dir.join(file_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_name() {
        let meta: Meta = Default::default();
        let name = meta.model_name();
        assert_eq!(name, "C_40_10_B260_ST");
    }
}
