use std::fmt::Debug;
use std::fs;
use std::fs::create_dir_all;
use std::path::Path;

#[derive(serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq)]
pub struct Meta {
    pub(crate) n: usize,
    pub(crate) model_type: ModelType,
    pub(crate) alg_type: AlgType,
}
impl Default for Meta {
    fn default() -> Self {
        Self {
            n: 260,
            model_type: ModelType::Classification,
            alg_type: AlgType::Bin,
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
        let binding = self.name();
        let dir: &Path = "./models".as_ref();
        let dir= &dir.join(&binding.as_str());
        let _ = create_dir_all(dir);
        let file = dir.join("model.meta");
        let out_string = serde_json::to_string_pretty(&self).unwrap();
        fs::write(file, out_string).expect("Unable to write file");
    }
    fn load<T: AsRef<Path>>(dir: T) -> Self {
        Default::default()
    }
    fn name(&self) -> String {
        let mt = first_char(&self.model_type);
        let at = first_char(&self.alg_type);
        format!("{mt}_{at}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_name() {
        let meta: Meta = Default::default();
        let name = meta.name();
        assert!(name == "C_B");
    }
    #[test]
    fn test_save() {
        let meta: Meta = Default::default();
        meta.save();
        let name = meta.name();
        let meta2 = Meta::load(&name);
    }
    #[test]
    fn test_load() {}
}

fn first_char<T: Debug>(e: &T) -> char {
    let name = format!("{:?}", e);
    name.chars().next().unwrap()
}
