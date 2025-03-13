use pacmog::PcmReader;
use std::collections::HashMap;
use std::fs;

use medius_meta::AlgType;
#[allow(unused_imports)]
use std::path::Path;
use utils::fft::fft_amplitudes;
use utils::mfcc::MFCC;
use utils::statistics::stat3;
use utils::{median_and_multiplier, normalize_array};

const SAMPLE_RATE: usize = 192_000;
#[allow(clippy::excessive_precision)]
const F5: f32 = 37523.4522;
const TOP: f32 = 54000.0;

pub fn parse_wav<P: AsRef<Path>>(
    path: P,
    n: usize,
    frequency: f32,
    buff_size: usize,
    alg_type: AlgType,
) -> anyhow::Result<Vec<f32>> {
    let all = fs::read(&path)?;
    parse_all(&all, n, frequency, buff_size, alg_type)
}

pub fn parse_all(
    all: &[u8],
    n: usize,
    frequency: f32,
    buff_size: usize,
    alg_type: AlgType,
) -> anyhow::Result<Vec<f32>> {
    let nf = frequency / F5;
    let raw = read_wav(all)?;
    match alg_type {
        AlgType::Bin => parse_bin(n, buff_size, nf, &raw),
        AlgType::Mfcc => parse_mfcc(n, 250, buff_size, nf, &raw),
        AlgType::Stat => parse_stat(n, buff_size, nf, &raw),
    }
}

fn parse_bin(n: usize, buff_size: usize, nf: f32, raw: &[f32]) -> anyhow::Result<Vec<f32>> {
    let useful: Vec<f32> = useful3(raw);
    let ampl: Vec<f32> = fft_amplitudes(&useful, buff_size);
    let range_list = build_range_list(TOP, n);
    let (median, multiplayer) = median_and_multiplier(&ampl);
    let norm_row = normalize_array(&ampl, median, multiplayer);
    let out = weighted5_one(&norm_row, n, &range_list, nf);
    Ok(out)
}
fn parse_mfcc(
    n: usize,
    filters: usize,
    buff_size: usize,
    nf: f32,
    raw: &[f32],
) -> anyhow::Result<Vec<f32>> {
    let useful: Vec<f32> = useful3(raw);
    let mfcc = MFCC::new(n, filters, SAMPLE_RATE, buff_size, true, false, false);
    let out = mfcc.calculate_mfcc(&useful, nf, buff_size, buff_size);
    Ok(out)
}

fn parse_stat(_n: usize, _buff_size: usize, _nf: f32, raw: &[f32]) -> anyhow::Result<Vec<f32>> {
    let useful: Vec<f32> = useful3(raw);
    stat3(&useful)
}

fn weighted5_one(row: &[f32], n: usize, f_rangelist: &[std::ops::Range<f32>], nf: f32) -> Vec<f32> {
    let freq = frequencies(row.len());
    let max_f = f_rangelist.last().unwrap().end;

    let mut grouped_data: HashMap<usize, (f32, usize)> = HashMap::with_capacity(n);
    for (i, &d) in row.iter().enumerate() {
        let fi = freq[i] / nf;
        if fi < max_f {
            let range_index = range_index(f_rangelist, fi);
            let entry = grouped_data.entry(range_index).or_insert((0.0, 0));
            entry.0 += d;
            entry.1 += 1;
        }
    }

    let mut result = vec![0.0; n];
    for (k, (sum, count)) in grouped_data {
        if k < n {
            result[k] = sum / count as f32;
        }
    }

    result
}

// Build a list of frequencies with a given window size and sample rate
fn frequencies(window_size: usize) -> Vec<f32> {
    let k = SAMPLE_RATE as f32 / (window_size as f32 * 2.0);
    (0..window_size).map(|i| i as f32 * k).collect()
}
fn range_index(f_rangelist: &[std::ops::Range<f32>], fi: f32) -> usize {
    f_rangelist
        .iter()
        .enumerate()
        .find(|(_, r)| r.contains(&fi))
        .unwrap()
        .0
}
fn build_range_list(tb: f32, n: usize) -> Vec<std::ops::Range<f32>> {
    let step = tb / n as f32;
    (1..=n)
        .map(|i| (i as f32 - 1.0) * step..i as f32 * step)
        .collect()
}
fn read_wav(wav: &[u8]) -> anyhow::Result<Vec<f32>> {
    //let wav: &[u8] = &data;
    let reader = PcmReader::new(wav)?;
    let specs = reader.get_pcm_specs();
    let num_samples = specs.num_samples;
    if specs.sample_rate as usize == SAMPLE_RATE && specs.num_channels == 2 {
        let channel = 0;
        let mut data: Vec<f32> = Vec::with_capacity(num_samples as usize);
        for sample in 0..num_samples {
            data.push(reader.read_sample(channel, sample)?);
        }
        Ok(data)
    } else {
        let mode = if specs.num_channels == 1 {
            "mono"
        } else {
            "stereo"
        };
        Err(anyhow::Error::msg(format!("A stereo file was expected with a sample rate of {}, but a {} file with a sample rate of {} was received!",
                                       SAMPLE_RATE, mode, specs.sample_rate)))
    }
}

fn useful3(raw: &[f32]) -> Vec<f32> {
    // Differences
    let diff: Vec<f32> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| if i == 0 { 0.0 } else { (v - raw[i - 1]).abs() })
        .collect();

    // Assuming SimpleMovingAverage is a struct with a `smooth` method
    let smoothed_diff = SimpleMovingAverage::new(5).smooth(&diff);

    // Calculate mean
    let gm = smoothed_diff.iter().sum::<f32>() / smoothed_diff.len() as f32;

    // Find first and last indices above the mean
    let first = smoothed_diff.iter().position(|&v| v > gm).unwrap_or(0);
    let last = smoothed_diff
        .iter()
        .rposition(|&v| v > gm)
        .unwrap_or(smoothed_diff.len() - 1);

    // Slice the original data
    raw[first..=last].to_vec() // Include last element
}

struct SimpleMovingAverage {
    window: Vec<f32>,
    period: usize,
    sum: f32,
}

impl SimpleMovingAverage {
    fn new(period: usize) -> Self {
        assert!(period > 0, "Period must be a positive integer!");
        Self {
            window: Vec::with_capacity(period),
            period,
            sum: 0.0,
        }
    }

    fn smooth(&mut self, data: &[f32]) -> Vec<f32> {
        let mut ma_data = Vec::with_capacity(data.len());
        for x in data {
            self.new_num(*x);
            ma_data.push(self.avg());
        }
        ma_data
    }

    fn new_num(&mut self, num: f32) {
        self.sum += num;
        self.window.push(num);
        if self.window.len() > self.period {
            self.sum -= self.window.remove(0);
        }
    }

    fn avg(&self) -> f32 {
        if self.window.is_empty() {
            0.0
        } else {
            self.sum / self.window.len() as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use medius_meta::BufSize;
    use std::fs::File;
    use std::io::{BufRead, Write};
    use std::time::Instant;
    use utils::set_root;

    #[allow(clippy::excessive_precision)]
    const FREQUENCY: f32 = 37523.4522;
    const N: usize = 260;
    const SRC111: &str = r"test_data/x1_y1.wav";
    const SRC138: &str = r"test_data/x3_y8.wav";

    /// Same as parse_wav but step by step and from the test data
    #[test]
    fn test111() {
        set_root();
        let buff_size: usize = BufSize::Small as usize;
        let wav_path: &Path = SRC111.as_ref();
        let out = by_steps(buff_size, wav_path);
        println!("out   :{:?}..{:?}", out[0], out.last().unwrap());
        assert!((0.17772603 - out[0]).abs() < f32::EPSILON);
        assert!((0.00020901869 - out.last().unwrap()).abs() < f32::EPSILON);
    }

    fn by_steps(buff_size: usize, wav_path: &Path) -> Vec<f32> {
        let all = fs::read(wav_path).unwrap();
        let raw = read_wav(&all).unwrap();
        println!("raw   :{:?}..{:?}", raw[0], raw.last().unwrap());
        let useful: Vec<f32> = useful3(&raw);
        println!("useful:{:?}..{:?}", useful[0], useful.last().unwrap());
        assert!((0.124176025 - useful[0]).abs() < 1e-8);
        let ampl: Vec<f32> = fft_amplitudes(&useful, buff_size);
        println!("ampl  :{:?}..{:?}", ampl[0], ampl.last().unwrap());
        let range_list = build_range_list(TOP, N);
        let (median, multiplayer) = median_and_multiplier(&ampl);
        let norm_row = normalize_array(&ampl, median, multiplayer);
        let nf = FREQUENCY / F5;
        
        weighted5_one(&norm_row, N, &range_list, nf)
    }

    /// Check parse_wav result from the test data
    #[test]
    fn test111_bin() {
        // 0.17772536181101378, 2.0898706297083427E-4
        test_parse(SRC111.as_ref(), N, 0.17772603, 0.00020901869, AlgType::Bin);
    }
    #[test]
    fn test111_stat() {
        // 0.0006713867..10.4952965
        test_parse(SRC111.as_ref(), N, 0.0006713867, 10.4952965, AlgType::Stat);
    }
    #[test]
    fn test111_mfcc() {
        // 50 Kotlin-154.79848250578078..0.004423994686392349
        // 50->-154.79861..0.004439953
        test_parse(SRC111.as_ref(), 50, -154.79861, 0.004439953, AlgType::Mfcc);
    }
    #[test]
    fn test138_parse() {
        // 0.015270601911173002, 0.14555550691863411
        test_parse(SRC138.as_ref(), 260, 0.016622879, 0.12061991, AlgType::Bin);
    }

    fn test_parse(wav_path: &Path, n: usize, first: f32, last: f32, alg: AlgType) {
        set_root();
        let buf_size: usize = BufSize::Small as usize;
        let start = Instant::now();
        let out = parse_wav(wav_path, n, FREQUENCY, buf_size, alg).unwrap();
        println!(
            "out   :{}->{:?}..{:?} {:5.2?}",
            out.len(),
            out[0],
            out.last().unwrap(),
            Instant::now().duration_since(start)
        );
        //println!("{:?}",&out);
        assert!((first - out[0]).abs() < f32::EPSILON);
        assert!((last - out.last().unwrap()).abs() < f32::EPSILON);
    }
    #[test]
    #[ignore]
    fn build_data() -> anyhow::Result<()> {
        build_by_alg(AlgType::Mfcc, 50)
        //build_by_alg(AlgType::Stat,10);
        //build_by_alg(AlgType::Bin,260);
    }

    fn build_by_alg(alg: AlgType, n: usize) -> anyhow::Result<()> {
        let src_file = File::open("../../data/src.csv").unwrap();
        let x_file = File::create("../../data/x.csv").unwrap();
        let y_file = File::create("../../data/y.csv").unwrap();
        let src = std::io::BufReader::new(src_file);
        let mut x = std::io::BufWriter::new(x_file);
        let mut y = std::io::BufWriter::new(y_file);
        let buf_size: usize = BufSize::Small as usize;
        let mut result = 0;
        for line in src.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() == 3 {
                let s = parts[0].to_string();
                let id = s.chars().nth(41).unwrap_or('_');
                let freq: f32 = parts[1].parse()?;
                let wp: f32 = parts[2].parse()?;
                let out = parse_wav(&s, n, freq, buf_size, alg.clone()).unwrap();
                let row = out
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(",");
                x.write_all(row.as_bytes())?;
                x.write_all(b"\n")?;
                y.write_all((wp / -0.1).abs().to_string().as_bytes())?;
                y.write_all(b",")?;
                y.write_all(id.to_string().as_bytes())?;
                y.write_all(b"\n")?;
                result += 1;
                print!("{}\r ", result);
            } else {
                return Err(anyhow::anyhow!("Invalid CSV format"));
            }
        }
        x.flush()?;
        y.flush()?;
        println!("{:?}", result); // 5 min 15 sec
        Ok(())
    }

    #[test]
    fn test_weighted5_one() {
        let ampl = generate_vec();
        let n = 10;
        let range_list = build_range_list(TOP, n);
        let (median, multiplayer) = median_and_multiplier(&ampl);
        println!("{median},{multiplayer}");
        let row = normalize_array(&ampl, median, multiplayer);

        let result = weighted5_one(&row, n, &range_list, 1.0);
        // since the real median after normalization is 0.0,
        // the average of most low groups is negative
        let must_be: [f32; 10] = [
            -0.5184687,
            -0.45620948,
            -0.39448705,
            -0.3327646,
            -0.2710421,
            -0.20931965,
            -0.14706048,
            -0.085338004,
            -0.02361555,
            0.038106915,
        ];
        println!("{:?}", result);
        assert_eq!(result, must_be);
    }
    fn generate_vec() -> Vec<f32> {
        let len = 1024;
        let step = 1.0 / (len - 1) as f32;
        (0..len).map(|i| i as f32 * step).collect()
    }
}
