use ndarray::Array1;
use num_complex::Complex;
use plotly::common::Mode;
use plotly::Plot;
use plotly::Scatter;
use rustfft::FftPlanner;

fn bandlimited_signal(
    frequencies: &[f32],
    amplitudes: &[f32],
    sampling_rate: f32,
    duration: f32,
) -> Vec<f32> {
    let num_samples = (duration * sampling_rate) as usize;
    let time = Array1::linspace(0.0, duration, num_samples);

    time.iter()
        .map(|t| {
            frequencies
                .iter()
                .zip(amplitudes)
                .map(|(freq, amp)| {
                    amp * Complex::from_polar(1.0, 2.0 * std::f32::consts::PI * freq * t)
                })
                .sum::<Complex<f32>>()
                .re
        })
        .collect()
}

fn bicks_reconstruction(signal: &[f32], sampling_rate: f32, max_frequency: f32) -> Vec<f32> {
    let num_samples = signal.len();
    //let time = Array1::linspace(0.0, (num_samples as f32) / sampling_rate, num_samples);
    let time: Vec<f32> = (1..=num_samples)
        .map(|i| i as f32 / sampling_rate)
        .collect();
    // Calculate the Discrete Fourier Transform (DFT)
    let mut fft_planner = FftPlanner::new();
    let fft = fft_planner.plan_fft_forward(num_samples);
    let mut fft_buffer: Vec<Complex<f32>> = prepare_data_for_fft(signal, signal.len());
    fft.process(&mut fft_buffer);

    // Extract relevant harmonics
    let mut reconstructed_signal = vec![0.0; num_samples];
    //let nyquist_freq = sampling_rate / 2.0;
    for (i, _item) in fft_buffer.iter().enumerate().take(num_samples / 2) {
        let frequency = i as f32 * sampling_rate / num_samples as f32;
        if frequency <= max_frequency {
            let real = fft_buffer[i].re;
            let imag = fft_buffer[i].im;
            for j in 0..num_samples {
                reconstructed_signal[j] += real
                    * f32::cos(2.0 * std::f32::consts::PI * frequency * time[j])
                    + imag * f32::sin(2.0 * std::f32::consts::PI * frequency * time[j]);
            }
        }
    }
    let scale = (num_samples / 2) as f32;
    // Scale the reconstructed signal
    for value in &mut reconstructed_signal {
        *value /= scale;
    }
    reconstructed_signal
}
fn prepare_data_for_fft(data: &[f32], fft_size: usize) -> Vec<Complex<f32>> {
    let mut complex_data = vec![Complex::new(0.0, 0.0); fft_size];
    for (i, &sample) in data.iter().enumerate() {
        complex_data[i] = Complex::new(sample, 0.0);
    }
    complex_data
}
fn main() {
    // Example usage:
    let sampling_rate = 100.0; // Hz
    let duration = 1.0; // seconds

    // Generate a sample band-limited signal
    let frequencies = vec![5.0, 10.0, 15.0];
    let amplitudes = vec![1.0, 0.5, 0.2];
    let original_signal = bandlimited_signal(&frequencies, &amplitudes, sampling_rate, duration);

    // Reconstruct the signal using BICKS
    show(sampling_rate, &original_signal);
}

#[cfg(target_os = "windows")]
fn show(sampling_rate: f32, original_signal: &[f32]) {
    let freqs = [20.0, 30.0, 50.0];
    let x: Vec<f32> = (0..100).map(|x| x as f32).collect();
    let mut plot = Plot::new();
    let original = Scatter::new(x.clone(), original_signal.to_owned())
        .mode(Mode::Lines)
        .name("original");
    println!(
        "src  {:?} ->{:?}",
        original_signal.len(),
        original_signal.split_at(5).0
    );
    plot.add_trace(original);
    for max_frequency in freqs.iter() {
        let reconstructed_signal =
            bicks_reconstruction(original_signal, sampling_rate, *max_frequency);
        let trace = Scatter::new(x.clone(), reconstructed_signal.clone())
            .mode(Mode::Lines)
            .name(format!("bicks {:?}", max_frequency));
        plot.add_trace(trace);
        println!(
            "bicks {max_frequency} ->{:?}",
            reconstructed_signal.split_at(5).0
        );
    }
    //plot.show();
    plot.write_html("bicks.html");
}
#[cfg(target_os = "linux")]
fn show(sampling_rate: f32, original_signal: &Vec<f32>) {
    let freqs = [20.0, 30.0, 50.0];
    let x: Vec<f32> = (0..100).map(|x| x as f32).collect();
    let original = Scatter::new(x.clone(), original_signal.clone())
        .mode(Mode::Lines)
        .name("original");
    println!(
        "src  {:?} ->{:?}",
        original_signal.len(),
        original_signal.split_at(5).0
    );
    for max_frequency in freqs.iter() {
        let reconstructed_signal =
            bicks_reconstruction(&original_signal, sampling_rate, *max_frequency);
        println!(
            "bicks {max_frequency} ->{:?}",
            reconstructed_signal.split_at(5).0
        );
    }
}
