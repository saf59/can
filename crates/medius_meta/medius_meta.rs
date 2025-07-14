use candle_core::safetensors::Load;
use candle_core::{Device, Tensor};
use candle_nn::{ops, VarMap};
use clap::ValueEnum;
use safetensors::SafeTensors;
use std::fmt::Debug;
use std::fs;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::string::ToString;
use utils::{enum_name, first_char};

pub const MODELS_DIR: &str = "./models";
const META_NAME: &str = "model.meta";
const MODEL_NAME: &str = "model.safetensors";
pub const DEFAULT: &str = "./models/model.meta";
pub const DEFAULT_VM: &str = "./models/model.safetensors";

#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Debug)]
pub struct Meta {
    // data && parse
    pub n: usize,
    pub alg_type: AlgType,
    pub buff_size: BufSize,
    pub flag: bool,
    // model
    pub model_type: ModelType,
    pub activation: Activation,
    #[serde(skip)]
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub train_part: f32,
    pub hidden: Option<String>,
    pub outputs: usize,
}
/// Get Meta embed resource
pub fn meta_from_ba(buf: &[u8]) -> anyhow::Result<Meta> {
    serde_yaml::from_slice(buf).map_err(|e| anyhow::anyhow!("Failed to parse Meta: {}", e))
}
pub fn safetensors_from_ba<'a>(buf: &'a [u8]) -> anyhow::Result<SafeTensors<'a>> {
    SafeTensors::deserialize(buf)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize SafeTensors: {}", e))
}
/// Fill VarMap from embed
pub fn fill_safetensors(varmap: &mut VarMap, map: SafeTensors) -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;
    for (k, v) in map.tensors() {
        let _ = varmap.set_one(k, v.load(&dev)?);
        // v.load(&dev)? ->  v.convert(&dev)?  in 0.5 version, but candle 0.8.4 use load
    }
    Ok(())
}

impl Default for Meta {
    /// Provides default values for the `Meta` struct
    fn default() -> Self {
        Self {
            // Data and parsing parameters
            n: 260,
            alg_type: AlgType::Bin,
            buff_size: BufSize::Small,
            flag: true,
            // Model parameters
            model_type: ModelType::Classification,
            activation: Activation::Relu,
            epochs: 100,
            batch_size: 40,
            learning_rate: 0.5,
            train_part: 0.9,
            hidden: Some("40,10".to_string()),
            outputs: 1,
        }
    }
}
impl Meta {
    pub fn small(&self) -> String {
        format!(
            "{:?}, {:?}, {}, {}, {}, {}, {}",
            self.model_type,
            self.activation,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.train_part,
            self.hidden.as_ref().unwrap_or(&"None".to_string())
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
    BinN,
    Mfcc,
    Stat,
    HOM,
}
#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum BufSize {
    Big = 65_536 * 2,
    Small = 65_536,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum Accuracy {
    Loss,
    Percent,
    Percent01,
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
    pub fn load_default() -> Meta {
        let meta_path: &Path = DEFAULT.as_ref();
        if !meta_path.exists() {
            return Meta::default();
        }
        let buf = fs::read(DEFAULT).unwrap();
        serde_yaml::from_slice(&buf).unwrap()
    }
    pub fn load(path: &Path) -> anyhow::Result<Meta> {
        if !path.exists() {
            return Err(anyhow::anyhow!("File does not exist"));
        }
        let buf = fs::read(path.join(META_NAME))?;
        let meta = serde_yaml::from_slice(&buf)?;
        Ok(meta)
    }
    /// Generates a data name based on the metadata
    pub fn data_name(&self) -> String {
        let at = first_char(&self.alg_type);
        let sn = &self.n.to_string();
        let bs = first_char(&self.buff_size);
        let sf = if self.flag { 'T' } else { 'F' };
        format!("{at}{sn}_{bs}{sf}")
    }
    pub fn name_out(&self) -> String {
        let hidden = self
            .hidden
            .as_ref()
            .unwrap_or(&"".to_string())
            .replace(",", "_");
        format!(
            "{:?},{:?},{:?},{},{:?},{:?}",
            &self.model_type, &self.alg_type, &self.n, hidden, &self.activation, &self.batch_size
        )
    }
    /// Generates a model name based on the metadata
    pub fn model_name(&self) -> String {
        let mt = first_char(&self.model_type);
        let at = first_char(&self.alg_type);
        let sn = &self.n.to_string();
        let bs = first_char(&self.buff_size);
        let h = self
            .hidden
            .as_ref()
            .unwrap_or(&"".to_string())
            .replace(",", "_");
        let sf = if self.flag { 'T' } else { 'F' };
        let bcs = &self.batch_size.to_string();
        let act = enum_name(&self.activation);
        format!("{mt}_{h}_{at}{sn}_{bs}{sf}_{bcs}{act}")
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
