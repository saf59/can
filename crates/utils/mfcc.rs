use crate::fft::fft_amplitudes;
use crate::{median_and_multiplier, normalize_array};

pub struct MFCC {
    num_coefficients: usize,
    num_filters: usize,
    sample_rate: usize,
    fft_size: usize,
    pre_emphasized: bool,
    norm: bool,
    scaled: bool,
}

impl MFCC {
    pub fn new(
        num_coefficients: usize,
        num_filters: usize,
        sample_rate: usize,
        fft_size: usize,
        pre_emphasized: bool,
        norm: bool,
        scaled: bool,
    ) -> Self {
        Self {
            num_coefficients,
            num_filters,
            sample_rate,
            fft_size,
            pre_emphasized,
            norm,
            scaled,
        }
    }

    pub fn calculate_mfcc(
        &self,
        raw: &[f32],
        scale: f32,
        frame_size: usize,
        frame_step: usize,
    ) -> Vec<f32> {
        let mut signal = vec![0.0; self.fft_size];
        signal[..raw.len().min(self.fft_size)]
            .copy_from_slice(&raw[..raw.len().min(self.fft_size)]);

        let pre = if self.pre_emphasized {
            self.pre_emphasize(&signal)
        } else {
            signal.clone()
        };

        let norm_data = if self.norm { self.normalize(&pre) } else { pre };

        let frames = self.frame_signal(&norm_data, frame_size, frame_step);
        //out("frames", &frames[0]);
        let power_spectrum = self.calculate_power_spectrum(&frames);
        //out("power", &power_spectrum[0]);
        let mel_spectrum = self.apply_mel_filter_banks(&power_spectrum, scale);
        //out("mel", &mel_spectrum[0]);
        let log_mel_spectrum = self.log_mel_spectrum(&mel_spectrum);
        //out("log", &log_mel_spectrum[0]);
        self.dct(&log_mel_spectrum)
    }

    fn pre_emphasize(&self, signal: &[f32]) -> Vec<f32> {
        let alpha = 0.97;
        let mut pre_emphasized = vec![0.0; signal.len()];
        for i in 1..signal.len() {
            pre_emphasized[i] = signal[i] - alpha * signal[i - 1];
        }
        pre_emphasized
    }

    fn frame_signal(&self, signal: &[f32], frame_size: usize, frame_step: usize) -> Vec<Vec<f32>> {
        let num_frames = (signal.len() - frame_size) / frame_step + 1;
        let mut frames = vec![vec![0.0; frame_size]; num_frames];
        for i in 0..num_frames {
            frames[i].copy_from_slice(&signal[i * frame_step..i * frame_step + frame_size]);
        }
        frames
    }

    fn calculate_power_spectrum(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        frames
            .iter()
            .map(|frame| {
                let fft_result = self.fft(frame);
                fft_result
                    .iter()
                    .map(|&x| x.powi(2) / self.fft_size as f32)
                    .collect()
            })
            .collect()
    }

    fn apply_mel_filter_banks(&self, power_spectrum: &[Vec<f32>], scale: f32) -> Vec<Vec<f32>> {
        let samp_rate = if self.scaled {
            self.sample_rate as f32 * scale
        } else {
            self.sample_rate as f32
        };
        let mel_filter_banks = self.create_mel_filter_banks(samp_rate);
        power_spectrum
            .iter()
            .map(|frame| {
                mel_filter_banks
                    .iter()
                    .map(|filter| frame.iter().zip(filter).map(|(&ps, &mf)| ps * mf).sum())
                    .collect()
            })
            .collect()
    }

    fn log_mel_spectrum(&self, mel_spectrum: &[Vec<f32>]) -> Vec<Vec<f32>> {
        mel_spectrum
            .iter()
            .map(|frame| frame.iter().map(|&x| (x + f32::EPSILON).log10()).collect())
            .collect()
    }

    fn dct(&self, log_mel_spectrum: &[Vec<f32>]) -> Vec<f32> {
        let num_filters = log_mel_spectrum[0].len();
        let num_frames = log_mel_spectrum.len();
        if num_filters == self.num_coefficients {
            return self.avg_col(log_mel_spectrum);
        }
        let mut mfccs = vec![vec![0.0; self.num_coefficients]; num_frames];
        for i in 0..num_frames {
            for k in 0..self.num_coefficients {
                let mut sum = 0.0;
                for n in 0..num_filters {
                    sum += log_mel_spectrum[i][n]
                        * (std::f32::consts::PI * k as f32 * (2 * n + 1) as f32
                            / (2.0 * num_filters as f32))
                            .cos();
                }
                mfccs[i][k] = sum * (2.0 / num_filters as f32).sqrt();
            }
        }
        self.avg_col(&mfccs)
    }

    fn create_mel_filter_banks(&self, samp_rate: f32) -> Vec<Vec<f32>> {
        let mel_min = 0.0;
        let mel_max = 2595.0 * (1.0 + samp_rate / 2.0 / 700.0).log10();
        let mel_points: Vec<f32> = (0..=self.num_filters + 1)
            .map(|i| mel_min + i as f32 * (mel_max - mel_min) / (self.num_filters + 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points
            .iter()
            .map(|&mel| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0))
            .collect();
        let bin: Vec<usize> = hz_points
            .iter()
            .map(|&hz| (hz * (self.fft_size + 1) as f32 / samp_rate).round() as usize)
            .collect();

        let mut mel_filter_banks = vec![vec![0.0; self.fft_size / 2 + 1]; self.num_filters];
        for i in 1..=self.num_filters {
            for j in bin[i - 1]..bin[i] {
                mel_filter_banks[i - 1][j] = (j - bin[i - 1]) as f32 / (bin[i] - bin[i - 1]) as f32;
            }
            for j in bin[i]..bin[i + 1] {
                mel_filter_banks[i - 1][j] = (bin[i + 1] - j) as f32 / (bin[i + 1] - bin[i]) as f32;
            }
        }
        mel_filter_banks
    }

    fn fft(&self, frame: &[f32]) -> Vec<f32> {
        fft_amplitudes(frame, self.fft_size)
    }

    fn normalize(&self, data: &[f32]) -> Vec<f32> {
        let (median, multiplayer) = median_and_multiplier(data);
        normalize_array(data, median, multiplayer)
    }

    fn avg_col(&self, data: &[Vec<f32>]) -> Vec<f32> {
        let num_rows = data.len();
        let num_cols = data[0].len();
        let mut column_averages = vec![0.0; num_cols];
        for col in 0..num_cols {
            let sum: f32 = data.iter().map(|row| row[col]).sum();
            column_averages[col] = sum / num_rows as f32;
        }
        column_averages
    }
}

/*
fn out(name: &str, out: &[f32]) {
    println!(
        "{name}:{:?}->{:?}..{:?}",
        out.len(),
        out[0],
        out.last().unwrap(),
    );
}
*/