use std::env;
use std::fmt::Debug;
use std::path::Path;

pub fn set_root() {
    let root = project_root::get_project_root().unwrap();
    let _ = env::set_current_dir(&root);
}

pub fn first_char<T: Debug>(e: &T) -> char {
    let name = format!("{:?}", e);
    name.chars().next().unwrap()
}
