/// Progress tracker for long-running tasks, with time estimation and formatted output.
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

/// Formats a number with underscores as thousands separators.
pub fn triad(n: i64) -> String {
    let s = n.to_string();
    let mut chars: Vec<char> = s.chars().collect();
    let mut i = chars.len() as isize - 3;
    while i > 0 {
        chars.insert(i as usize, '_');
        i -= 3;
    }
    chars.into_iter().collect()
}

/// Converts milliseconds to a human-readable time string.
pub fn ms_to_time(ms: u128) -> String {
    if ms == 0 {
        return "?".to_string();
    }
    let mut seconds = ms / 1000;
    let hours = seconds / 3600;
    if hours > 1000 {
        return "?".to_string();
    }
    let minutes = (seconds % 3600) / 60;
    seconds %= 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else if minutes > 0 {
        format!("{minutes:02}:{seconds:02}")
    } else {
        format!("{}.{:03} sec.", seconds, ms % 1000)
    }
}

/// Returns the current time as a string in HH:mm.ss format.
pub fn now_time() -> String {
    use chrono::Local;
    Local::now().format("%H:%M.%S").to_string()
}

/// Progress tracker struct.
pub struct Total {
    total: AtomicI64,
    each: i64,
    parsed: AtomicI64,
    past: AtomicI64,
    start_time: Instant,
}

impl Total {
    /// Creates a new `Total` tracker.
    pub fn new(total: i64) -> Self {
        Self::custom(total,1,0)
    }
    /// # Arguments
    /// * `total` - The total number of items to process.
    /// * `each` - How often to update (default: 1).
    /// * `past` - Number of items already processed (default: 0).
    /// * for start not from beginning
    pub fn custom(total: i64, each: i64, past: i64) -> Self {
        Self {
            total: AtomicI64::new(total - past),
            each,
            parsed: AtomicI64::new(0),
            past: AtomicI64::new(past),
            start_time: Instant::now(),
        }
    }
    pub fn next(&self) -> String {
        self.next_add(self.each)
    }
    /// Advances the progress by `add` (default: `each`), returns time info.
    pub fn next_add(&self, add: i64) -> String {
        self.parsed.fetch_add(add, Ordering::SeqCst);
        let from_start = self.start_time.elapsed().as_millis();
        let parsed = self.parsed.load(Ordering::SeqCst) as f64;
        let total = self.total.load(Ordering::SeqCst) as f64;
        let p_speed = if parsed > 0.0 {
            from_start as f64 / parsed
        } else {
            0.0
        };
        let total_ms = (total * p_speed) as u128;
        format!("{}/{}", ms_to_time(from_start), ms_to_time(total_ms))
    }
    pub fn step(&self, prefix: &str) {
        self.step_add(prefix,self.each,false)
    }
    /// Prints a progress step with optional prefix and last flag.
    pub fn step_add(&self, prefix: &str, add: i64, last: bool) {
        if add < 0 {
            self.total.fetch_add(add, Ordering::SeqCst);
            self.past.fetch_add(-add, Ordering::SeqCst);
        } else {
            self.parsed.fetch_add(add, Ordering::SeqCst);
        }
        let past_val = self.past.load(Ordering::SeqCst);
        let past_s = if past_val == 0 {
            "".to_string()
        } else {
            format!("+{}", triad(past_val))
        };
        let parsed = self.parsed.load(Ordering::SeqCst);
        let total = self.total.load(Ordering::SeqCst);
        let msg = format!(
            "{} {}/{}{} {} at {}     ",
            prefix,
            triad(parsed),
            triad(total),
            past_s,
            self.next(),
            now_time()
        );
        if last {
            println!("{msg}");
        } else {
            print!("\r{msg}");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triad() {
        assert_eq!(triad(1234567), "1_234_567");
        assert_eq!(triad(12), "12");
        assert_eq!(triad(0), "0");
    }

    #[test]
    fn test_ms_to_time() {
        assert_eq!(ms_to_time(0), "?");
        assert_eq!(ms_to_time(1000), "1.000 sec.");
        assert_eq!(ms_to_time(61_000), "01:01");
        assert_eq!(ms_to_time(3_661_000), "01:01:01");
    }

    #[test]
    fn test_total_progress() {
        let total = Total::custom(100, 10, 0);
        let msg = total.next_add(10);
        assert!(msg.contains("/"));
        total.step_add("Test", 10, false);
        total.step_add("Test", 10, true);
    }
}