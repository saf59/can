use ho::HigherOrderMomentsAnalyzer;
use medius_meta::AlgType;
use pacmog::PcmReader;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::f32::consts::PI;
use std::fs;
#[allow(unused_imports)]
use std::path::Path;
use utils::fft::{asize, fft_amplitudes};
use utils::mfcc::MFCC;
use utils::statistics::{iqr, mean, population_variance, skew, stat3, to_db};
use utils::{median_and_multiplier, normalize_array};

const SAMPLE_RATE: usize = 192_000;
const SAMPLE_RATE_F32: f32 = SAMPLE_RATE as f32;
#[allow(clippy::excessive_precision)]
const F5: f32 = 37523.4522;
const TOP: f32 = 54000.0;

// call only from builder
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
// call also from medius_utils.detect()
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
        AlgType::BinN => parse_bin_n(n, &raw),
        AlgType::Mfcc => parse_mfcc(n, 250, buff_size, nf, &raw),
        AlgType::Stat => parse_stat(n, buff_size, nf, &raw),
        AlgType::ATen => parse_aten(&raw, SAMPLE_RATE_F32, n),
        AlgType::HOM => Err(anyhow::anyhow!("HOM algorithm is not implemented yet")),
        AlgType::Ten => Err(anyhow::anyhow!("Ten algorithm is not implemented yet")),
    }
}
pub fn parse_aten(raw: &[f32], sampling_rate: f32, n: usize) -> anyhow::Result<Vec<f32>> {
    let n_list: Vec<usize> = vec![9, 17, 18, 24, 30, 34, 38, 52];
    if !n_list.contains(&n) {
        return Err(anyhow::anyhow!("Aten algorithm requires N in {n_list:?} !"));
    }
    // useful
    let useful = usefull(raw);
    let useful = scale(useful);
    // fft
    let buff_size = asize(useful.len()).0 * 2;
    let ampl = fft_amplitudes(&useful, buff_size);
    // 10 bins
    // 18 total
    let vx = match n {
        // 9 && 18 are OLD approach
        9 => {
            let bins = bin_harmonics(&ampl, 10_000.0, 1_500.0, sampling_rate);
            let skew: Vec<f32> = bins[1..]
                .iter()
                .map(|bin| to_db(bin))
                .map(|db| skew(&db))
                .collect();
            if skew.len() != 9 {
                return Err(anyhow::anyhow!("Aten skew data requires N=9!"));
            }
            skew
        }
        18 => {
            let bins = bin_harmonics(&ampl, 10_000.0, 1_500.0, sampling_rate);
            let skew: Vec<f32> = bins[1..]
                .iter()
                .map(|bin| to_db(bin))
                .map(|db| skew(&db))
                .collect();
            let avg: Vec<f32> = bins[1..].iter().map(|bin| mean(bin)).collect();
            let mut vx = vec![0.0; n];
            vx[..9].copy_from_slice(&avg);
            vx[9..].copy_from_slice(&skew);
            vx
        }
        // all others via bin_n_harmonics()
        17 => {
            let bins = bin_n_harmonics(&ampl, 1 , 10_000.0, 1_500.0, sampling_rate);
            let db_vec:Vec<Vec<f32>> = bins.iter().map(|bin| to_db(bin)).collect();
            let skew: Vec<f32> = db_vec.iter().map(|db_bin| skew(db_bin)).collect();
            let variance: Vec<f32> = db_vec.iter().map(|db_bin| population_variance(db_bin)).collect();
            let irq: Vec<f32> = db_vec.iter().map(|db_bin| iqr(db_bin)).collect();
            let mut vx = vec![0.0; n];
            vx[..7].copy_from_slice(&skew[2..]);
            vx[7]=variance[1];
            vx[8..13].copy_from_slice(&variance[4..]);
            vx[13..].copy_from_slice(&irq[5..]);
            vx
        }
        24 => {
            let bins = bin_n_harmonics(&ampl, 3 , 10_000.0, 1_500.0, sampling_rate);
            let db_vec:Vec<Vec<f32>> = bins.iter().map(|bin| to_db(bin)).collect();
            let skew: Vec<f32> = db_vec.iter().map(|db_bin| skew(db_bin)).collect();
            skew[2..].to_vec()
        }
        30 => {
            let bins = bin_n_harmonics(&ampl, 2 , 10_000.0, 1_500.0, sampling_rate);
            let db_vec:Vec<Vec<f32>> = bins.iter().map(|bin| to_db(bin)).collect();
            let skew: Vec<f32> = db_vec.iter().map(|db_bin| skew(db_bin)).collect();
            let variance: Vec<f32> = db_vec.iter().map(|db_bin| population_variance(db_bin)).collect();
            let irq: Vec<f32> = db_vec.iter().map(|db_bin| iqr(db_bin)).collect();
            let mut vx = vec![0.0; n];
            vx[..6].copy_from_slice(&skew[4..10]);
            vx[6..12].copy_from_slice(&skew[11..]);
            vx[12]=variance[4];
            vx[13..22].copy_from_slice(&variance[8..]);
            vx[22..].copy_from_slice(&irq[9..]);
            vx
        }
        34 => {
            let bins = bin_n_harmonics(&ampl,2, 10_000.0, 1_500.0, sampling_rate);
            let skew: Vec<f32> = bins
                .iter()
                .map(|bin| to_db(bin))
                .map(|db| skew(&db))
                .collect();
            let avg: Vec<f32> = bins.iter().map(|bin| mean(bin)).collect();
            let mut vx = vec![0.0; n];
            vx[..17].copy_from_slice(&avg);
            vx[17..].copy_from_slice(&skew);
            vx
        }
        38 => {
            let bins = bin_n_harmonics(&ampl, 3 , 10_000.0, 1_500.0, sampling_rate);
            let db_vec:Vec<Vec<f32>> = bins.iter().map(|bin| to_db(bin)).collect();
            let skew: Vec<f32> = db_vec.iter().map(|db_bin| skew(db_bin)).collect();
            let variance: Vec<f32> = db_vec.iter().map(|db_bin| population_variance(db_bin)).collect();
            let irq: Vec<f32> = db_vec.iter().map(|db_bin| iqr(db_bin)).collect();
            let mut vx = vec![0.0; n];
            vx[..24].copy_from_slice(&skew[2..26]);
            vx[24]=variance[4];
            vx[25]=variance[6];
            vx[26]=variance[10];
            vx[27]=variance[14];
            vx[28]=variance[15];
            vx[29]=variance[17];
            vx[30]=variance[18];
            vx[31..36].copy_from_slice(&irq[14..19]);
            vx[36..].copy_from_slice(&irq[24..]);
            vx
        }
        52 => {
            let bins = bin_n_harmonics(&ampl,3, 10_000.0, 1_500.0, sampling_rate);
            let skew: Vec<f32> = bins
                .iter()
                .map(|bin| to_db(bin))
                .map(|db| skew(&db))
                .collect();
            let avg: Vec<f32> = bins.iter().map(|bin| mean(bin)).collect();
            let mut vx = vec![0.0; n];
            vx[..26].copy_from_slice(&avg);
            vx[26..].copy_from_slice(&skew);
            vx
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Aten algorithm requires {n} in [{n_list:?}] !"
            ))
        }
    };
    Ok(vx)
}

fn scale(useful: Vec<f32>) -> Vec<f32> {
    let max = useful
        .iter()
        .map(|&x| x.abs())
        .max_by(f32::total_cmp)
        .unwrap();
    let scale = if max > 0.0 { 1.0 / max } else { 1.0 };
    let useful: Vec<f32> = useful.iter().map(|x| x * scale).collect();
    useful
}

pub fn parse_hom(raw: &[f32]) -> anyhow::Result<Vec<Vec<f32>>> {
    let peaks = all_peaks(raw, 4000);
    let pulses: Vec<Vec<f32>> = peaks
        .iter()
        .map(|p| impuls(raw, *p))
        .collect::<Vec<Vec<f32>>>();
    let bands = 1;
    let band_width = 3000.0;
    let centers = vec![85_000.0];
    let window = ""; //"hanning";
    let mut analyzer = HigherOrderMomentsAnalyzer::default();
    let results: Vec<Vec<f32>> = pulses
        .iter()
        .map(|pulse| hob(pulse, bands, band_width, window, &centers, &mut analyzer))
        .collect::<Vec<_>>();
    Ok(results)
}
pub fn hob(
    data: &[f32],
    bands: usize,
    band_width: f32,
    window: &str,
    centers: &[f32],
    analyzer: &mut HigherOrderMomentsAnalyzer,
) -> Vec<f32> {
    // Ensure signal is sized to 4096
    let signal = to_sized(data, 4096);
    let signal = f32_slice_to_f64_vec(&signal);
    // Analyze full spectrum
    let full_result = analyzer.analyze_full_spectrum(&signal, window);
    let mut result = analyzer.extract_features_for_ml(&full_result);

    // Get custom frequency bands
    let centers = f32_slice_to_f64_vec(centers);
    let custom_bands = analyzer.get_custom_frequency_bands(&centers, bands, band_width as f64);
    //let band_results = analyzer.analyze_frequency_bands(signal, bands, "hanning");
    let band_results = analyzer.analyze_frequency_bands(&signal, &custom_bands, window);

    for (_band_name, band_result) in band_results {
        let band_features = analyzer.extract_features_for_ml(&band_result);
        result.extend(band_features);
    }
    f64_slice_to_f32_vec(&result)
}
// &[f32] to Vec<f64>
fn f32_slice_to_f64_vec(input: &[f32]) -> Vec<f64> {
    input.iter().map(|&x| x as f64).collect()
}

// &[f64] to Vec<f32>
fn f64_slice_to_f32_vec(input: &[f64]) -> Vec<f32> {
    input.iter().map(|&x| x as f32).collect()
}

// Helper: resize or pad/truncate to 4096
fn to_sized(data: &[f32], size: usize) -> Vec<f32> {
    let mut out = vec![0.0; size];
    let n = data.len().min(size);
    out[..n].copy_from_slice(&data[..n]);
    out
}

fn parse_bin_n(n: usize, raw: &[f32]) -> anyhow::Result<Vec<f32>> {
    let peaks = all_peaks(raw, 4000);
    let pulses: Vec<Vec<f32>> = peaks
        .iter()
        .map(|p| impuls(raw, *p))
        .collect::<Vec<Vec<f32>>>();
    let ampls: Vec<Vec<f32>> = pulses
        .iter()
        .map(|pulse| fft_amplitudes(pulse, 4096))
        .collect::<Vec<Vec<f32>>>();
    let bins = ampls
        .iter()
        .map(|ampl| bin(ampl, n))
        .collect::<Vec<Vec<f32>>>();
    let avg_by_bins = column_averages(&bins);
    Ok(avg_by_bins)
}

fn avg(data: Vec<f32>) -> f32 {
    if data.is_empty() {
        0.0
    } else {
        data.iter().sum::<f32>() / data.len() as f32
    }
}

fn column_averages(data: &[Vec<f32>]) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    let rows = data.len();
    (0..cols)
        .map(|col| {
            let sum: f32 = data.iter().map(|row| row[col]).sum();
            sum / rows as f32
        })
        .collect()
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
pub fn read_wav(wav: &[u8]) -> anyhow::Result<Vec<f32>> {
    read_wav_ch(wav, 2)
}
fn read_wav_ch(wav: &[u8], channels: u16) -> anyhow::Result<Vec<f32>> {
    let mut input = wav; // &wav[..];
    let reader = PcmReader::new(&mut input)?;
    let specs = reader.get_pcm_specs();
    let num_samples = specs.num_samples;
    if specs.sample_rate as usize == SAMPLE_RATE && specs.num_channels == channels {
        let channel = 0;
        let mut data: Vec<f32> = Vec::with_capacity(num_samples as usize);
        for sample in 0..num_samples {
            data.push(reader.read_sample(channel, sample)?);
        }
        Ok(data)
    } else {
        let (mode, must) = if specs.num_channels == 1 {
            ("mono", "stereo")
        } else {
            ("stereo", "mono")
        };
        Err(anyhow::Error::msg(format!("A {must} file was expected with a sample rate of {}, but a {} file with a sample rate of {} was received!",
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
pub fn usefull(raw_data: &[f32]) -> Vec<f32> {
    let peak_list = all_peaks(raw_data, 4000);
    if peak_list.is_empty() {
        return vec![];
    }
    let u_start = peak_list[0].saturating_sub(2);
    let mut e_peak = *peak_list.last().unwrap();
    let mut size = ((e_peak - peak_list[0]) as f32 / (peak_list.len() - 1) as f32) as usize;
    if (e_peak + size) > raw_data.len() && peak_list.len() >= 2 {
        e_peak = peak_list[peak_list.len() - 2];
        size = ((e_peak - peak_list[0]) / (peak_list.len() - 2)) as usize;
    }

    let u_end = e_peak + size;
    raw_data.get(u_start..=u_end).unwrap_or(&[]).to_vec()
}
fn bin(data: &[f32], n: usize) -> Vec<f32> {
    let (part, left) = left_part(n);
    let size = ((SAMPLE_RATE_F32 / 2.0 + left) / part).ceil() as usize;
    let mut result: Vec<Vec<f32>> = vec![Vec::new(); size];
    let k = SAMPLE_RATE_F32 / (data.len() as f32 * 2.0);

    for (i, &v) in data.iter().enumerate() {
        let f = i as f32 * k;
        let bin = ((f + left) / part) as usize;
        if bin < result.len() {
            result[bin].push(v);
        }
    }
    result.into_iter().map(avg).collect()
}

fn left_part(n: usize) -> (f32, f32) {
    let nn = ((n as f32) / 10.0).round();
    let part = 10_000.0 / nn;
    let left = part / 2.0;
    (part, left)
}

pub fn all_peaks(data: &[f32], n_after: usize) -> Vec<usize> {
    let abs_data: Vec<f32> = data.iter().map(|x| x.abs()).collect();
    let (mean, std) = mean_and_std(&abs_data);
    let top = mean + std * 3.0;
    let mut i = 0;
    let mut peaks = Vec::new();
    // Important !!!
    // This fix does not allow to find peaks on short data
    let last = data.len() - n_after;
    while i < data.len() {
        if data[i] > top && i < last {
            peaks.push(i);
            i += n_after;
        } else {
            i += 1;
        }
    }
    peaks
}

// Helper function to compute mean and standard deviation
fn mean_and_std(data: &[f32]) -> (f32, f32) {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    (mean, var.sqrt())
}

pub fn impuls(data: &[f32], peak: usize) -> Vec<f32> {
    // Calculate mean of positive values in the range [peak, peak+1000)
    let mean: f32 = data
        .iter()
        .skip(peak)
        .take(1000)
        .filter(|&&x| x > 0.0)
        .copied()
        .sum::<f32>()
        / data
            .iter()
            .skip(peak)
            .take(1000)
            .filter(|&&x| x > 0.0)
            .count()
            .max(1) as f32;

    let last = back_search(data, peak, mean);
    // Slice from (peak-2)..=last, handling bounds
    let start = peak.saturating_sub(2);
    let end = last + 1; // inclusive range
    data.get(start..end).unwrap_or(&[]).to_vec()
}
pub fn bin_harmonics(
    data: &[f32],
    base_freq: f32,
    edge_width: f32,
    sample_rate: f32,
) -> Vec<Vec<f32>> {
    let n_bins = ((sample_rate / 2.0) / base_freq).round() as usize;
    let mut result = vec![Vec::new(); n_bins];
    let k = sample_rate / (data.len() as f32 * 2.0);

    for (i, &v) in data.iter().enumerate() {
        let f = i as f32 * k;
        let bin_idx = (f / base_freq).round() as usize;
        if bin_idx < n_bins {
            let bin_start = bin_idx as f32 * base_freq + edge_width;
            let bin_end = (bin_idx as f32 + 1.0) * base_freq - edge_width;
            if f >= bin_start && f < bin_end {
                result[bin_idx].push(v);
            }
        }
    }
    result
}

fn back_search(data: &[f32], peak: usize, border: f32) -> usize {
    let start = peak + 4096 - 2;
    for i in (peak..=start).rev() {
        if i < data.len() && data[i] > border {
            return i;
        }
    }
    peak.saturating_sub(1)
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
use std::cmp;

/**
 * Splits FFT amplitude data into bins based on harmonics of a base frequency.
 * By Google Studio Team, adapted for Rust.
 *
 * @param amplitudes The input amplitude data from a Fourier Transform.
 * @param n_sub_bins The number of sub-bins to split each main bin into.
 * @param base_freq The fundamental frequency for calculating harmonics.
 * @param edge_width The frequency width around each harmonic to be ignored.
 * @param sample_rate The sample rate of the original signal.
 * @return A vector of bins, where each bin is a vector of amplitude values.
 */
pub fn bin_n_harmonics(
    amplitudes: &[f32],
    n_sub_bins: usize,
    base_freq: f32,
    edge_width: f32,
    sample_rate: f32,
) -> Vec<Vec<f32>> {
    let mut result_bins: Vec<Vec<f32>> = Vec::new();
    let nyquist_freq = sample_rate / 2.0;

    // Return an empty vector if the input is invalid or empty.
    if base_freq <= 0.0 || amplitudes.is_empty() {
        return result_bins;
    }

    // Calculate the frequency represented by each index in the amplitudes array.
    let freq_resolution = nyquist_freq / (amplitudes.len() - 1) as f32;

    // Determine the highest harmonic number below the Nyquist frequency.
    let max_harmonic_n = (nyquist_freq / base_freq) as usize;

    // Iterate through the regions between harmonics to create the main bins.
    for n in 1..max_harmonic_n {
        let start_freq = (n as f32 * base_freq) + edge_width;
        let end_freq = ((n + 1) as f32 * base_freq) - edge_width;

        if start_freq >= end_freq || start_freq >= nyquist_freq {
            continue;
        }

        let start_index = (start_freq / freq_resolution).ceil() as usize;
        let end_index = (end_freq / freq_resolution).floor() as usize;

        if start_index >= end_index {
            continue;
        }

        let main_bin_data = &amplitudes[start_index..=end_index];

        if !main_bin_data.is_empty() {
            if n_sub_bins > 1 {
                let sub_bin_size = (main_bin_data.len() as f32 / n_sub_bins as f32).ceil() as usize;
                if sub_bin_size > 0 {
                    for chunk in main_bin_data.chunks(sub_bin_size) {
                        result_bins.push(chunk.to_vec());
                    }
                }
            } else {
                result_bins.push(main_bin_data.to_vec());
            }
        }
    }

    // Handle the last bin.
    if max_harmonic_n > 0 {
        let start_freq = (max_harmonic_n as f32 * base_freq) + edge_width;

        if start_freq < nyquist_freq {
            let start_index = (start_freq / freq_resolution).ceil() as usize;

            if start_index < amplitudes.len() {
                let last_main_bin_data = &amplitudes[start_index..];

                if !last_main_bin_data.is_empty() {
                    let num_sub_bins_for_last = if n_sub_bins > 1 {
                        cmp::max(1, n_sub_bins - 1)
                    } else {
                        1
                    };
                    let sub_bin_size = (last_main_bin_data.len() as f32
                        / num_sub_bins_for_last as f32)
                        .ceil() as usize;
                    if sub_bin_size > 0 {
                        for chunk in last_main_bin_data.chunks(sub_bin_size) {
                            result_bins.push(chunk.to_vec());
                        }
                    }
                }
            }
        }
    }

    result_bins
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
    const SRCRAW: &str = r"test_data/458a3473-ad9a-4c0a-8a2e-7b3a8e634ae6.wav";
    const SRCRAW0: &str = r"test_data/0_0.wav";
    #[test]
    #[ignore]
    fn test_raw_to_bin11() {
        let wav_path: &Path = SRCRAW.as_ref();
        #[rustfmt::skip]
        let real = vec![5.911_861E-4,6.021_952E-4,0.002_478_469_9,0.001_267_877,0.001_978_414_1,9.065_647_6E-4,0.001_497_84,7.194_772E-4,9.831_714E-4,8.784_834E-4,1.527_001_1E-4];
        test_to_bin11(wav_path, real, 2);
    }
    #[test]
    #[ignore]
    fn test_interim_to_bin11() {
        // SRCRAW0 is a very short file
        // and last version of all_peaks(raw, 4000) does not find any peaks
        // but without this fix broken wav file will panic
        let wav_path: &Path = SRCRAW0.as_ref();
        #[rustfmt::skip]
        let real = vec![0.00087014644, 0.00075164676, 0.002676331, 0.001098881, 0.0022909078, 0.00084123365, 0.0017334798, 0.00078802824, 0.001121256, 0.0008936477, 0.00022057218];
        test_to_bin11(wav_path, real, 1);
    }

    fn test_to_bin11(wav_path: &Path, real: Vec<f32>, channels: u16) {
        set_root();
        let all = fs::read(wav_path).unwrap();
        let raw = read_wav_ch(&all, channels).unwrap();
        let data = parse_bin_n(11, &raw).unwrap();
        println!("data.len  :{:?},real:{:?}", data.len(), real.len());
        assert_eq!(data.len(), real.len());
        for i in 0..data.len() {
            assert!((real[i] - data[i]).abs() < f32::EPSILON);
        }
    }
    // for correctness of the *bin_n left and part detection
    #[test]
    fn test_left_part() {
        test_one_left_part(11, 5_000.0, 10_000.0);
        test_one_left_part(20, 5_000.0 / 2.0, 10_000.0 / 2.0);
        test_one_left_part(30, 5_000.0 / 3.0, 10_000.0 / 3.0);
        test_one_left_part(39, 5_000.0 / 4.0, 10_000.0 / 4.0);
        test_one_left_part(49, 5_000.0 / 5.0, 10_000.0 / 5.0);
    }
    fn test_one_left_part(n: usize, real_left: f32, real_part: f32) {
        let (part, left) = left_part(n);
        assert!((real_part - part).abs() < f32::EPSILON);
        assert!((real_left - left).abs() < f32::EPSILON);
    }
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
        let src_file = File::open("../../data/src.csv")?;
        let x_file = File::create("../../data/x.csv")?;
        let y_file = File::create("../../data/y.csv")?;
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
                let out = parse_wav(&s, n, freq, buf_size, alg.clone())?;
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
                print!("{result}\r ");
            } else {
                return Err(anyhow::anyhow!("Invalid CSV format"));
            }
        }
        x.flush()?;
        y.flush()?;
        println!("{result:?}"); // 5 min 15 sec
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
        assert_eq!(result, must_be);
    }
    fn generate_vec() -> Vec<f32> {
        let len = 1024;
        let step = 1.0 / (len - 1) as f32;
        (0..len).map(|i| i as f32 * step).collect()
    }
    #[test]
    fn test_hob_single_band() {
        use ho::HigherOrderMomentsAnalyzer;

        let data = sample_hom_signal();
        let bands = 1;
        let band_width = 3000.0;
        let centers = vec![85_000.0];
        let window = ""; //"hanning";
        println!("Signal: {:?}..{:?}", data.first(), data.last());
        // Create analyzer (adjust as needed for your constructor)
        let mut analyzer = HigherOrderMomentsAnalyzer::default();
        let result = hob(&data, bands, band_width, window, &centers, &mut analyzer);
        println!("HOB size: {:?}", result.len());
        println!("HOB result 1st: {:?}", result[0..17].to_vec());
        println!("HOB result 2nd: {:?}", result[17..].to_vec());
    }

    fn sample_hom_signal() -> Vec<f32> {
        let signal_length = 4096;
        let sampling_rate = 192000.0;
        let signal: Vec<f32> = (0..signal_length)
            .map(|i| {
                let t = i as f32 / sampling_rate;
                // Harmonic components
                let fundamental = (2.0 * PI * 10000.0 * t).sin();
                let second_harmonic = 0.5 * (2.0 * PI * 20000.0 * t).sin();
                let third_harmonic = 0.3 * (2.0 * PI * 85000.0 * t).sin();
                // Nonlinear component (creates interesting higher-order moments)
                let nonlinear = 0.1 * (2.0 * PI * 10000.0 * t).sin().powf(3.0);
                fundamental + second_harmonic + third_harmonic + nonlinear //+ noise
            })
            .collect();
        signal
    }
    /// Calls the main test logic with various input data sizes.
    #[test]
    fn test_n() {
        test_bin_n_harmonics_for_size(128);
        test_bin_n_harmonics_for_size(256);
        test_bin_n_harmonics_for_size(4096);
    }

    /// Contains the core testing logic, adapted from the Kotlin example.
    fn test_bin_n_harmonics_for_size(size: usize) {
        // Generate sample data where each element is its index.
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        // Use the same default parameters as the Kotlin function.
        let base_freq = 10_000.0;
        let edge_width = 1_500.0;
        let sample_rate = 192_000.0;

        // Generate bins for n=1, 2, and 3.
        let bins1 = bin_n_harmonics(&data, 1, base_freq, edge_width, sample_rate);
        let bins2 = bin_n_harmonics(&data, 2, base_freq, edge_width, sample_rate);
        let bins3 = bin_n_harmonics(&data, 3, base_freq, edge_width, sample_rate);

        // --- Sanity Checks ---
        // Ensure that the resulting lists of bins are not empty.
        assert!(!bins1.is_empty(), "List bins1 should not be empty");
        assert!(!bins2.is_empty(), "List bins2 should not be empty");
        assert!(!bins3.is_empty(), "List bins3 should not be empty");

        // Ensure that all bins within each list contain data.
        assert!(
            bins1.iter().all(|b| !b.is_empty()),
            "All bins in bins1 should have size > 0"
        );
        assert!(
            bins2.iter().all(|b| !b.is_empty()),
            "All bins in bins2 should have size > 0"
        );
        assert!(
            bins3.iter().all(|b| !b.is_empty()),
            "All bins in bins3 should have size > 0"
        );

        // --- Bin Count Checks ---
        // Check if the number of sub-bins matches the expected formula.
        // The last main bin is split into max(1, n-1) sub-bins.
        let main_bin_count = bins1.len();
        assert_eq!(
            bins2.len(),
            (main_bin_count - 1) * 2 + 1,
            "Bin count for n=2 should match expected"
        );
        assert_eq!(
            bins3.len(),
            (main_bin_count - 1) * 3 + 2,
            "Bin count for n=3 should match expected"
        );

        // --- Summation Checks ---
        // Verify that the sum of sub-bins equals the sum of the original main bins.
        let sums1: Vec<f32> = bins1.iter().map(|bin| bin.iter().sum()).collect();

        // For bins2, chunk sub-bins by 2 and sum them to reconstruct the original main bin sums.
        let sums2: Vec<f32> = sum_by_chunks(bins2, 2);

        // For bins3, chunk sub-bins by 3.
        let sums3: Vec<f32> = sum_by_chunks(bins3, 3);

        // Since the input data and operations are exact, we can directly compare the resulting sums.
        assert_eq!(sums1, sums2, "Bins for n=2 by sum should match expected");
        assert_eq!(sums1, sums3, "Bins for n=3 by sum should match expected");
    }

    fn sum_by_chunks(bins3: Vec<Vec<f32>>, chunks:usize) -> Vec<f32> {
        bins3
            .chunks(chunks)
            .map(|chunk| -> f32 {
                chunk
                    .iter()
                    .map(|sub_bin| sub_bin.iter().sum::<f32>())
                    .sum()
            })
            .collect()
    }
}
