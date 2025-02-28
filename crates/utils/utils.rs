pub mod statistics;
pub mod fft;

use std::env;
use std::fmt::Debug;

pub fn set_root() {
    let root = project_root::get_project_root().unwrap();
    let _ = env::set_current_dir(&root);
}

pub fn first_char<T: Debug>(e: &T) -> char {
    let name = format!("{:?}", e);
    name.chars().next().unwrap()
}
pub fn enum_name<T: Debug>(e: &T) -> String {
    let name = format!("_{:?}", e);
    match name.as_str() {
        "_Relu" => "".to_string(),
        _ => name.to_lowercase()
    }
}
