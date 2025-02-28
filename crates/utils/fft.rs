use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// full fft of a signal with a buffer size
pub fn fft_amplitudes(raw: &[f32], buf_size: usize) -> Vec<f32> {
    let data = align(raw, buf_size);
    let buffer = fft_forward(&data, buf_size);
    to_amplitudes(buffer, buf_size)
}

fn to_amplitudes(buffer: Vec<Complex<f32>>, buf_size: usize) -> Vec<f32> {
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

pub fn fft_forward(data: &[f32], buf_size: usize) -> Vec<Complex<f32>> {
    let mut fft_planner = FftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(buf_size);
    let mut buffer = f32_to_complex_vec(&data, buf_size);
    fft.process(&mut buffer);
    buffer
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
