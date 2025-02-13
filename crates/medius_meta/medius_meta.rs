use clap::ValueEnum;
use std::fmt::Debug;
use std::fs;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::string::ToString;
use candle_core::Tensor;
use candle_nn::ops;
use utils::{enum_name, first_char};

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
    pub activation: Activation,
    #[serde(skip)]
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub train_part: f32,
    pub hidden0: usize,
    pub hidden1: usize,
    pub outputs: usize
}
impl Default for Meta {
    /// Provides default values for the `Meta` struct
    fn default() -> Self {
        Self {
            // Data and parsing parameters
            n: 260,
            alg_type: AlgType::Bin,
            buff_size: BufSize::Small,
            scaled_frequency: true,
            // Model parameters
            model_type: ModelType::Classification,
            activation: Activation::Relu,
            epochs: 100,
            batch_size: 40,
            learning_rate: 0.5,
            train_part: 0.9,
            hidden0: 40,
            hidden1: 10,
            outputs: 1,
        }
    }
}
impl Meta {
    pub fn small(&self) -> String {
        format!("{:?}, {:?}, {}, {}, {}, {}",
                self.model_type,
                self.activation,
                self.epochs,
                self.batch_size,
                self.learning_rate,
                self.train_part
        )
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
#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum Activation {
    Gelu,
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
//    Swiglu,
    Swish,
    HardSwish,
}
impl Activation {
    pub fn forward(&self, xs: &Tensor) -> candle_core::error::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => ops::sigmoid(xs),
            Self::HardSigmoid => ops::hard_sigmoid(xs),
           // Self::Swiglu => ops::swiglu(xs),
            Self::Swish => xs * ops::sigmoid(xs),
            Self::HardSwish => xs * ops::hard_sigmoid(xs),
        }
    }
}
impl Meta {
    /// Saves the metadata to the default and specific file paths
    pub fn save(&self) {
        let file = self.meta_file();
        let parent = file.parent().expect("Parent dir.");
        let _ = create_dir_all(parent);
        let out_string = serde_yaml::to_string(&self).unwrap();
        fs::write(file, &out_string).expect("Unable to write meta file");
        fs::write(DEFAULT, out_string).expect("Unable to write default meta file");
    }
    /// Loads the default metadata from the default file path
    pub fn load_default() ->Meta {
        let meta_path:&Path = DEFAULT.as_ref();
        if !meta_path.exists() { return Meta::default();}
        let buf = fs::read(DEFAULT).unwrap();
        serde_yaml::from_slice(&buf).unwrap()
    }
    /// Generates a data name based on the metadata
    pub fn data_name(&self) -> String {
        let at = first_char(&self.alg_type);
        let sn = &self.n.to_string();
        let bs = first_char(&self.buff_size);
        let sf = if self.scaled_frequency { 'T' } else { 'F' };
        format!("{at}{sn}_{bs}{sf}")
    }
    /// Generates a model name based on the metadata
    pub fn model_name(&self) -> String {
        let mt = first_char(&self.model_type);
        let at = first_char(&self.alg_type);
        let sn = &self.n.to_string();
        let bs = first_char(&self.buff_size);
        let h0 = &self.hidden0.to_string();
        let h1 = &self.hidden1.to_string();
        let sf = if self.scaled_frequency { 'T' } else { 'F' };
        let bcs = &self.batch_size.to_string();
        let act =enum_name(&self.activation);
        format!("{mt}_{h0}_{h1}_{at}{sn}_{bs}{sf}_{bcs}{act}")
    }
    /// Returns the file path for the metadata file
    pub fn meta_file(&self) -> PathBuf {
        let name = &self.model_name();
        Self::named(name, META_NAME)
    }
    /// Returns the file path for the model file
    pub fn model_file(&self) -> PathBuf {
        let name = &self.model_name();
        Self::named(name, MODEL_NAME)
    }
    /// Helper function to generate a file path based on type and file name
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
        assert_eq!(name, "C_40_10_B260_ST_40");
    }
}
