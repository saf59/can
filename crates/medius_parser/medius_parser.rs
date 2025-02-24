use candle_core::cpu::erf::erf;
use pacmog::PcmReader;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::fs;

#[allow(unused_imports)]
use std::path::Path;

const SAMPLE_RATE: usize = 192_000;
#[allow(clippy::excessive_precision)]
const F5: f32 = 37523.4522;
const TOP: f32 = 54000.0;

pub fn parse_wav<P: AsRef<Path>>(
    path: P,
    n: usize,
    frequency: f32,
    buff_size: usize,
) -> anyhow::Result<Vec<f32>> {
    let all = fs::read(&path)?;
    parse_all(&all, n, frequency, buff_size)
}

pub fn parse_all(
    all: &[u8],
    n: usize,
    frequency: f32,
    buff_size: usize,
) -> anyhow::Result<Vec<f32>> {
    let nf = frequency / F5;
    let raw = read_wav(all)?;
    let useful: Vec<f32> = useful3(&raw);
    let ampl: Vec<f32> = fft_amplitudes(&useful, buff_size);
    let range_list = build_range_list(TOP, n);
    let (median, multiplayer) = median_and_multiplier(&ampl);
    let norm_row = normalize_array(&ampl, median, multiplayer);
    let out = weighted5_one(&norm_row, n, &range_list, nf);
    Ok(out)
}

fn fft_amplitudes(raw: &[f32], buf_size: usize) -> Vec<f32> {
    let mut fft_planner = FftPlanner::<f32>::new();
    let data = align(raw, buf_size);
    let fft = fft_planner.plan_fft_forward(buf_size);
    let mut buffer = f32_to_complex_vec(&data, buf_size);
    fft.process(&mut buffer);
    let half = buf_size / 2;
    let n = half as f32;
    buffer
        .iter()
        .take(half)
        .map(|item| {
            let real = item.re.powi(2);
            let imag = item.im.powi(2);
            ((real + imag).sqrt()) / n
        })
        .collect()
}
fn align(data: &[f32], buf_size: usize) -> Vec<f32> {
    if data.len() < buf_size {
        let mut aligned = vec![0.0; buf_size];
        aligned[..data.len()].copy_from_slice(data);
        aligned
    } else {
        data[..buf_size].to_vec()
    }
}
fn f32_to_complex_vec(data: &[f32], buf_size: usize) -> Vec<Complex<f32>> {
    data.iter()
        .take(buf_size)
        .map(|&item| Complex::new(item, 0.0))
        .collect()
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

fn median_and_multiplier(values: &[f32]) -> (f32, f32) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = values.len();
    let mid = len / 2;

    let median = if len % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    let centered_data: Vec<f32> = values.iter().map(|v| v - median).collect();
    let mean: f32 = centered_data.iter().sum::<f32>() / centered_data.len() as f32;
    let variance: f32 = centered_data
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>()
        / centered_data.len() as f32;
    let std_dev = variance.sqrt();

    let scale_factor = 1.0 / (2.0 * std_dev);
    let scaled_data: Vec<f32> = centered_data.iter().map(|v| v * scale_factor).collect();

    let normal_distribution_share: f32 = (1.0 - erf((1.0 / 2.0_f32.sqrt()).into())) as f32;
    let emissions_share = scaled_data
        .into_iter()
        .filter(|v| *v < 0.0 || *v > 1.0)
        .count() as f32
        / centered_data.len() as f32;

    let multiplier = if emissions_share <= normal_distribution_share {
        scale_factor
    } else {
        scale_factor * (normal_distribution_share / emissions_share)
    };

    (median, multiplier)
}
fn normalize_array(data: &[f32], median: f32, multiplier: f32) -> Vec<f32> {
    data.iter()
        .map(|v| normalize(*v, median, multiplier))
        .collect()
}
fn normalize(value: f32, median: f32, multiplier: f32) -> f32 {
    (value - median) * multiplier
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
    use std::time::Instant;
    use utils::set_root;

    #[allow(clippy::excessive_precision)]
    const FREQUENCY: f32 = 37523.4522;
    const N: usize = 260;
    const SRC: &str = r"test_data/x1_y1.wav";

    /// Same as parse_wav but step by step and from the test data
    #[test]
    fn test111() {
        set_root();
        let buff_size: usize = BufSize::Small as usize;
        let wav_path: &Path = SRC.as_ref();
        let out = by_steps(buff_size, wav_path);
        println!("out   :{:?}..{:?}", out[0], out.last().unwrap());
        assert!((0.17772603 - out[0]).abs() < 1e-9);
        assert!((0.00020901869 - out.last().unwrap()).abs() < 1e-8);
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
        let out = weighted5_one(&norm_row, N, &range_list, nf);
        out
    }

    /// Check parse_wav result from the test data
    #[test]
    fn test111_parse() {
        set_root();
        let buf_size: usize = BufSize::Small as usize;
        let start = Instant::now();
        let wav_path: &Path = SRC.as_ref();
        let out = parse_wav(wav_path, N, FREQUENCY, buf_size).unwrap();
        println!(
            "out   :{:?}..{:?} {:5.2?}",
            out[0],
            out.last().unwrap(),
            Instant::now().duration_since(start)
        );
        assert!((0.17772603 - out[0]).abs() < 1e-8);
        assert!((0.00020901869 - out.last().unwrap()).abs() < 1e-8);
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
            -0.5184687, -0.45620948, -0.39448705, -0.3327646, -0.2710421,
            -0.20931965, -0.14706048, -0.085338004, -0.02361555, 0.038106915,
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
