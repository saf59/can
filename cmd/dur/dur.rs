use clap::Parser;
use clipboard::{ClipboardContext, ClipboardProvider};
use std::time::{Duration, SystemTime};

#[derive(Parser)]
struct Args {
    /// Stop time measurement, else start
    #[arg(long, default_value_t = false)]
    stop: bool,
}

// If the --stop flag is provided,
// it retrieves the start time from the clipboard,
// calculates the duration since the start time, and prints it.
// If the --stop flag is not provided,
// it stores the current time in nanoseconds to the clipboard.
fn main() {
    if Args::parse().stop {
        let mut ctx: ClipboardContext = ClipboardProvider::new().unwrap();
        match ctx.get_contents().unwrap().parse::<u128>() {
            Ok(start_time) => {
                let now_since_epoch: u128 = nano();
                let duration: u128 = now_since_epoch - start_time;
                println!("{:?}", Duration::from_nanos(duration as u64))
            }
            Err(_) => {
                println!("No start time found");
            }
        }
    } else {
        let start_since_epoch = nano();
        let start_time = format!("{:?}", start_since_epoch);
        let mut ctx: ClipboardContext = ClipboardProvider::new().unwrap();
        ctx.set_contents(start_time).unwrap();
    }
}

// Returns the current time in nanoseconds since the UNIX epoch.
fn nano() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
