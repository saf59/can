use candle_core::cpu::erf::erf;
use pacmog::PcmReader;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const SAMPLE_RATE: usize = 192_000;
const F5: f32 = 37523.4522;
const TOP: f32 = 54000.0;
pub fn parse_wav<P: AsRef<Path>>(path: P, n: usize, frequency:f32, to_low:bool) -> anyhow::Result<Vec<f32>> {
    let nf = frequency / F5;
    let buf_size:usize = if to_low {65_536} else {65_536 * 2};
    let all = fs::read(&path).unwrap();
    let raw = read_wav(all).unwrap();
    let useful: Vec<f32> = useful3(&raw);
    let ampl: Vec<f32> = fft_amplitudes(&useful,buf_size);
    let range_list = build_range_list(TOP, n);
    let out = weighted5_one(&ampl, n, &range_list, nf);
    Ok(out)
}

fn fft_amplitudes(data: &[f32],buf_size:usize) -> Vec<f32> {
    let mut fft_planner = FftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(buf_size);

    let mut buffer = f32_to_complex_vec(&data,buf_size);
    fft.process(&mut buffer);
    let n: f32 = (buf_size / 2usize) as f32;
    let mut amplitudes: Vec<f32> = Vec::with_capacity(buffer.len() / 2);
    for i in 0..buffer.len() / 2 {
        let real = buffer[i].re.powi(2);
        let imag = buffer[i].im.powi(2);
        amplitudes.push(((real + imag).sqrt()) / n);
    }
    amplitudes
}

fn f32_to_complex_vec(data: &[f32],buf_size:usize) -> Vec<Complex<f32>> {
    let mut complex_data = Vec::with_capacity(buf_size);
    let max = data.len();
    for i in 0..buf_size {
        if i < max {
            complex_data.push(Complex::new(data[i], 0.0))
        } else {
            complex_data.push(Complex::new(0.0, 0.0))
        };
    }
    complex_data
}

fn weighted5_one(row: &[f32], n: usize, f_rangelist: &[std::ops::Range<f32>], nf: f32) -> Vec<f32> {
    let (median, multiplayer) = median_and_multiplier(&row);
    let norm_row = normalize_array(&row, median, multiplayer);
    let mut result = vec![0.0; n]; // Initialize result with zeros
    let freq = frequencies(row.len());
    //println!("freq  :{:?}..{:?}", freq[0], freq.last().unwrap(), );
    let max_f = f_rangelist.last().unwrap().end; // Assuming last is Some

    let filtered_data = norm_row
        .iter()
        .enumerate()
        .filter(|(i, _)| freq[*i] / nf < max_f);

    let mut grouped_data: HashMap<usize, Vec<f32>> = HashMap::new();
    for (i, d) in filtered_data {
        let range_index = range_index(f_rangelist, freq[i] / nf);
        grouped_data
            .entry(range_index)
            .or_insert(Vec::new())
            .push(*d);
    }

    for (k, v) in grouped_data.iter() {
        if *k < n {
            result[*k] = v.iter().sum::<f32>() / v.len() as f32; // Average
        }
    }

    result
}

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
fn read_wav(data: Vec<u8>) -> anyhow::Result<Vec<f32>> {
    let wav: &[u8] = &data;
    let reader = PcmReader::new(wav)?;
    let specs = reader.get_pcm_specs();
    let num_samples = specs.num_samples;
    return if specs.sample_rate as usize == SAMPLE_RATE && specs.num_channels == 2 {
        let channel = 0;
        let mut data: Vec<f32> = Vec::with_capacity(num_samples as usize);
        for sample in 0..num_samples {
            data.push(reader.read_sample(channel, sample).unwrap());
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
    };
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
    let first = smoothed_diff
        .clone()
        .into_iter()
        .enumerate()
        .find(|(_, v)| *v > gm)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let last = smoothed_diff
        .clone()
        .into_iter()
        .rev()
        .enumerate()
        .find(|(_, v)| *v > gm)
        .map(|(i, _)| smoothed_diff.len() - i - 1)
        .unwrap_or(smoothed_diff.len() - 1);

    // Slice the original data
    let aligned = raw[first..last + 1].to_vec(); // Include last element

    aligned
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
    use std::time::Instant;
    use super::*;
    const frequency: f32 = 37523.4522;
    const n:usize = 260;
    const to_low:bool = true;
    const src: &str =
    r"W:\data\medius\Audiodaten Scanner\Sample_1\Matrix_1_WP-0.4\x1_y1.wav";

    #[test]
    fn test111() {
        let nf = frequency / F5;
        let buf_size:usize = if to_low {65_536} else {65_536 * 2};
        let all = fs::read(src.as_ref() as &Path).unwrap();
        let raw = read_wav(all).unwrap();
        //raw  :0.0005187988..0.0012817383
        println!("raw  :{:?}..{:?}", raw[0], raw.last().unwrap());
        let useful: Vec<f32> = useful3(&raw);
        println!("useful:{:?}..{:?}", useful[0], useful.last().unwrap());
        assert!((0.124176025 - useful[0]).abs() < 1e-9);
        let ampl: Vec<f32> = fft_amplitudes(&useful,buf_size);
        println!("ampl  :{:?}..{:?}", ampl[0], ampl.last().unwrap());
        //assert!((2.974548e-6 - ampl.last().unwrap()).abs() < 1e-9);
        let range_list = build_range_list(TOP, n);
        let out = weighted5_one(&ampl, n, &range_list, nf);
        println!("out  :{:?}..{:?}", out[0], out.last().unwrap());
        assert!((0.17772603 - out[0]).abs() < 1e-9);
        assert!((0.00020901869 - out.last().unwrap()).abs() < 1e-9);
    }
    #[test]
    fn test111_parse() {
        let out = parse_wav(src.as_ref() as &Path, n, frequency, to_low).unwrap();
        let start = Instant::now();
        println!("out  :{:?}..{:?} {:5.2?}", out[0], out.last().unwrap(), Instant::now().duration_since(start));
        assert!((0.17772603 - out[0]).abs() < 1e-9);
        assert!((0.00020901869 - out.last().unwrap()).abs() < 1e-9);
    }
}

