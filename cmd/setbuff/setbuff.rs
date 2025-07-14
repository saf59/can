use clipboard::{ClipboardContext, ClipboardProvider};
use std::env;
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: setbuff <value>");
        std::process::exit(1);
    }
    let body = &args[1];
    let mut ctx: ClipboardContext = ClipboardProvider::new().unwrap();
    ctx.set_contents(body.to_string()).unwrap();
}
