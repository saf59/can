use std::fmt::Debug;
use std::fs;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use utils::first_char;

#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq)]
pub struct Meta {
    pub(crate) n: usize,
    pub(crate) model_type: ModelType,
    pub(crate) alg_type: AlgType,
    pub(crate) epochs: usize,
    pub(crate) learning_rate: f64,
    pub(crate) train_part: f32,
    pub(crate) h0: usize,
    pub(crate) h1: usize,
}
impl Default for Meta {
    fn default() -> Self {
        Self {
            n: 260,
            model_type: ModelType::Classification,
            alg_type: AlgType::Bin,
            epochs: 100,
            learning_rate: 0.5,
            train_part: 0.9,
            h0: 40,
            h1: 10,
        }
    }
}
#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq, Debug)]
pub enum ModelType {
    Regression,
    Classification,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq, Debug)]
pub enum AlgType {
    Bin,
    Mfcc,
    Stat,
}
impl Meta {
    fn save(&self) {
        let file = self.file();
        let _ = create_dir_all(file.parent().expect("Parent dir."));
        let out_string = serde_yaml::to_string(&self).unwrap();
        fs::write(file, out_string).expect("Unable to write file");
    }
    fn load(path: &str) -> Self {
        let file = Self::named(path);
        let buf = fs::read(file).unwrap();
        serde_yaml::from_slice(&buf).unwrap()
    }
    fn name(&self) -> String {
        let mt = first_char(&self.model_type);
        let at = first_char(&self.alg_type);
        format!("{mt}_{at}")
    }
    fn file(&self) -> PathBuf {
        let name = &self.name();
        Self::named(name)
    }

    fn named(name: &str) -> PathBuf {
        let dir: &Path = "./models".as_ref();
        let dir = &dir.join(name);
        dir.join("model.meta")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use utils::set_root;
    #[test]
    fn test_name() {
        let meta: Meta = Default::default();
        let name = meta.name();
        assert!(name == "C_B");
    }
    #[test]
    fn test_save_load() {
        set_root();
        let meta: Meta = Default::default();
        let name = meta.name();
        let file = meta.file();
        if !file.exists() {
            meta.save();
        }
        let meta2 = Meta::load(&name);
        assert!(meta == meta2);
    }
}
