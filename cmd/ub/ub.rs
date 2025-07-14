use std::time::Duration;

use anyhow::Result;
use chrono::Local;
use crossbeam_channel::{select, tick, unbounded, Receiver};

fn main() -> Result<()> {
    let ctrl_c_events = ctrl_channel()?;
    let duration = Duration::from_millis(10);
    let ticks = tick(duration);
    loop {
        select! {
            recv(ticks) -> _ => {
                let date = Local::now();
                print!("{}\r", date.format("%Y-%m-%d %H:%M:%S"));
            }
            recv(ctrl_c_events) -> _ => {
                println!("\nExit ^C!");
                break;
            }
        }
    }
    Ok(())
}

fn ctrl_channel() -> Result<Receiver<()>, ctrlc::Error> {
    let (sender, receiver) = unbounded(); //bounded(100);
    ctrlc::set_handler(move || {
        let _ = sender.send(());
    })?;
    Ok(receiver)
}
