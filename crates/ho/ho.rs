use realfft::num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Result structure containing all higher-order spectral moments and related metrics
#[derive(Debug, Clone)]
pub struct HigherOrderMomentsResult {
    /// First moment - spectral centroid (center of mass of the spectrum)
    pub spectral_mean: f64,
    /// Second central moment - spectral spread
    pub spectral_variance: f64,
    /// Third standardized moment - spectral asymmetry
    pub spectral_skewness: f64,
    /// Fourth standardized moment - spectral peakedness
    pub spectral_kurtosis: f64,
    /// Ratio of geometric to arithmetic mean - measure of spectral uniformity
    pub spectral_flatness: f64,
    /// Shannon entropy of the power spectral density
    pub spectral_entropy: f64,
    /// Normalized moment ratios for feature extraction
    pub moment_ratios: Vec<f64>,
    /// Cumulants derived from raw moments
    pub cumulants: Vec<f64>,
    /// Per-harmonic moment analysis for harmonic content
    pub harmonic_moments: HashMap<usize, Vec<f64>>,
}

/// Frequency band definition for band-specific analysis
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    /// Start frequency of the band in Hz
    pub start_freq: f64,
    /// End frequency of the band in Hz
    pub end_freq: f64,
    /// Human-readable name for the frequency band
    pub name: String,
}

impl FrequencyBand {
    pub fn new(start_freq: f64, end_freq: f64, name: &str) -> Self {
        Self {
            start_freq,
            end_freq,
            name: name.to_string(),
        }
    }
}

/// Main analyzer for computing higher-order spectral moments and related features
pub struct HigherOrderMomentsAnalyzer {
    /// Audio sampling rate in Hz
    sampling_rate: f64,
    /// Maximum order of moments to compute (N)
    n: usize,
    /// Cache for precomputed window functions to avoid recomputation
    window_cache: HashMap<(usize, String), Vec<f64>>,
    // FFT planner for efficient Fourier transforms
    //fft_planner: FftPlanner<f64>,
}

impl HigherOrderMomentsAnalyzer {
    /// Create a new analyzer with specified parameters
    ///
    /// # Arguments
    /// * `sampling_rate` - Audio sampling rate in Hz (default: 192000.0)
    /// * `n` - Maximum order of moments to compute (default: 5)
    pub fn new(sampling_rate: f64, n: usize) -> Self {
        Self {
            sampling_rate,
            n,
            window_cache: HashMap::new(),
            //fft_planner: FftPlanner::new(),
        }
    }

    /// Create analyzer with default parameters (192kHz, 5th order moments)
    pub fn new_default() -> Self {
        Self::new(192000.0, 5)
    }

    /// Get or compute window function with caching for performance
    ///
    /// # Arguments
    /// * `n` - Window length in samples
    /// * `window_type` - Type of window ("hanning", "hamming", "blackman", or "rectangular")
    fn get_window(&mut self, n: usize, window_type: &str) -> Vec<f64> {
        let key = (n, window_type.to_lowercase());

        if let Some(cached_window) = self.window_cache.get(&key) {
            return cached_window.clone();
        }

        // Compute window function based on type
        let window: Vec<f64> = (0..n)
            .map(|i| {
                let i_f = i as f64;
                let n_f = n as f64;
                match window_type.to_lowercase().as_str() {
                    "hanning" => 0.5 * (1.0 - (2.0 * PI * i_f / (n_f - 1.0)).cos()),
                    "hamming" => 0.54 - 0.46 * (2.0 * PI * i_f / (n_f - 1.0)).cos(),
                    "blackman" => {
                        0.42 - 0.5 * (2.0 * PI * i_f / (n_f - 1.0)).cos()
                            + 0.08 * (4.0 * PI * i_f / (n_f - 1.0)).cos()
                    }
                    _ => 1.0, // Rectangular window
                }
            })
            .collect();

        self.window_cache.insert(key, window.clone());
        window
    }

    /// Apply windowing function to signal to reduce spectral leakage
    ///
    /// # Arguments
    /// * `signal` - Input signal samples
    /// * `window_type` - Type of window to apply
    fn apply_window(&mut self, signal: &[f64], window_type: &str) -> Vec<f64> {
        if window_type.is_empty() {
            return signal.to_vec();
        }
        let window = self.get_window(signal.len(), window_type);
        signal
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect()
    }

    /// Compute FFT using rustfft for optimal performance
    /// Handles any input size efficiently with proper zero-padding if needed
    ///
    /// # Arguments
    /// * `input` - Real-valued input signal
    ///
    /// # Returns
    /// * Complex-valued FFT output with same length as input
    fn compute_fft(&mut self, input: &[f64]) -> Vec<Complex64> {
        utils::fft::fft64_forward(input, input.len())
    }

    /// Compute Power Spectral Density from windowed signal using rustfft
    ///
    /// # Arguments
    /// * `signal` - Input time-domain signal
    /// * `window` - Window type to apply before FFT
    fn compute_psd(&mut self, signal: &[f64], window: &str) -> Vec<f64> {
        // Apply windowing to reduce spectral leakage
        let windowed_signal = self.apply_window(signal, window);

        // Compute FFT of windowed signal using rustfft
        let spectrum = self.compute_fft(&windowed_signal);
        let n = spectrum.len();

        // Convert to one-sided PSD (positive frequencies only)
        let mut psd = vec![0.0; n / 2 + 1];
        for i in 0..psd.len() {
            let magnitude = spectrum[i].norm(); // rustfft Complex32 has norm() method
            let mut power = (magnitude * magnitude) / (self.sampling_rate * n as f64);

            // Double the power for positive frequencies (except DC and Nyquist)
            if i > 0 && i < n / 2 {
                power *= 2.0;
            }
            psd[i] = power;
        }
        psd
    }

    /// Generate frequency bins corresponding to FFT output
    ///
    /// # Arguments
    /// * `n` - Length of original signal (determines frequency resolution)
    fn get_frequencies(&self, n: usize) -> Vec<f64> {
        let df = self.sampling_rate / n as f64; // Frequency resolution
        (0..=n / 2).map(|i| i as f64 * df).collect()
    }

    /// Analyze the full spectrum and compute all higher-order moments
    ///
    /// # Arguments
    /// * `signal` - Input time-domain signal
    /// * `window` - Window type for preprocessing
    pub fn analyze_full_spectrum(
        &mut self,
        signal: &[f64],
        window: &str,
    ) -> HigherOrderMomentsResult {
        let psd = self.compute_psd(signal, window);
        let frequencies = self.get_frequencies(signal.len());
        self.compute_higher_order_moments(&psd, &frequencies)
    }

    /// Analyze specific frequency bands separately
    ///
    /// # Arguments
    /// * `signal` - Input time-domain signal
    /// * `bands` - List of frequency bands to analyze
    /// * `window` - Window type for preprocessing
    pub fn analyze_frequency_bands(
        &mut self,
        signal: &[f64],
        bands: &[FrequencyBand],
        window: &str,
    ) -> HashMap<String, HigherOrderMomentsResult> {
        let psd = self.compute_psd(signal, window);
        let frequencies = self.get_frequencies(signal.len());
        let mut results = HashMap::new();

        for band in bands {
            // Find frequency bins within the band
            let band_indices: Vec<usize> = frequencies
                .iter()
                .enumerate()
                .filter(|(_, &freq)| freq >= band.start_freq && freq <= band.end_freq)
                .map(|(i, _)| i)
                .collect();
            //println!("Analyzing band: {} (from {} {} indices)", band.name, band_indices[0], band_indices.len());
            if !band_indices.is_empty() {
                // Extract PSD and frequencies for this band
                let band_psd: Vec<f64> = band_indices.iter().map(|&i| psd[i]).collect();
                let band_freqs: Vec<f64> = band_indices.iter().map(|&i| frequencies[i]).collect();
                //println!("Band PSD: {:?}", band_psd);
                //println!("Band freq: {:?}", band_freqs);
                let result = self.compute_higher_order_moments(&band_psd, &band_freqs);
                results.insert(band.name.clone(), result);
            }
        }
        results
    }

    /// Analyze spectral content around harmonic frequencies
    /// Used for analyzing signals with known fundamental frequency
    ///
    /// # Arguments
    /// * `signal` - Input time-domain signal
    /// * `fundamental_freq` - Fundamental frequency in Hz
    /// * `max_harmonic` - Maximum harmonic number to analyze
    /// * `bandwidth_hz` - Bandwidth around each harmonic to include
    pub fn analyze_around_harmonics(
        &mut self,
        signal: &[f64],
        fundamental_freq: f64,
        max_harmonic: usize,
        bandwidth_hz: f64,
    ) -> HigherOrderMomentsResult {
        let psd = self.compute_psd(signal, "hanning");
        let frequencies = self.get_frequencies(signal.len());
        let mut harmonic_moments = HashMap::new();
        let mut all_harmonic_indices = Vec::new();

        // Analyze each harmonic separately
        for harmonic in 1..=max_harmonic {
            let harmonic_freq = fundamental_freq * harmonic as f64;
            let start_freq = harmonic_freq - bandwidth_hz / 2.0;
            let end_freq = harmonic_freq + bandwidth_hz / 2.0;

            // Find frequency bins around this harmonic
            let harmonic_indices: Vec<usize> = frequencies
                .iter()
                .enumerate()
                .filter(|(_, &freq)| freq >= start_freq && freq <= end_freq)
                .map(|(i, _)| i)
                .collect();

            if !harmonic_indices.is_empty() {
                let harmonic_psd: Vec<f64> = harmonic_indices.iter().map(|&i| psd[i]).collect();
                let harmonic_freqs: Vec<f64> =
                    harmonic_indices.iter().map(|&i| frequencies[i]).collect();

                // Compute raw moments for this harmonic
                let moments = self.compute_raw_moments(&harmonic_psd, &harmonic_freqs, self.n + 1);
                harmonic_moments.insert(harmonic, moments);

                // Collect indices for combined analysis
                all_harmonic_indices.extend(harmonic_indices);
            }
        }

        // Combine all harmonic regions for overall analysis
        let combined_psd: Vec<f64> = all_harmonic_indices.iter().map(|&i| psd[i]).collect();
        let combined_freqs: Vec<f64> = all_harmonic_indices
            .iter()
            .map(|&i| frequencies[i])
            .collect();

        let mut result = self.compute_higher_order_moments(&combined_psd, &combined_freqs);
        result.harmonic_moments = harmonic_moments;
        result
    }

    /// Core function to compute all higher-order spectral moments and derived metrics
    ///
    /// # Arguments
    /// * `psd` - Power spectral density values
    /// * `frequencies` - Corresponding frequency values
    fn compute_higher_order_moments(
        &self,
        psd: &[f64],
        frequencies: &[f64],
    ) -> HigherOrderMomentsResult {
        let total_power: f64 = psd.iter().sum();

        // Normalize PSD to create probability density function
        let normalized_psd: Vec<f64> = if total_power > 0.0 {
            psd.iter().map(|&p| p / total_power).collect()
        } else {
            psd.to_vec()
        };
        // Compute raw moments (m_k = sum(p_i * f_i^k))
        let raw_moments = self.compute_raw_moments(&normalized_psd, frequencies, self.n + 1);
        let mean = raw_moments[1]; // First moment is the spectral centroid
                                   // Compute central moments around the mean
        let central_moments = self.compute_central_moments(&normalized_psd, frequencies, mean, 4);
        // Compute cumulants from raw moments
        let cumulants = self.compute_cumulants(&raw_moments);
        // Extract standard spectral features
        let spectral_mean = mean;
        let spectral_variance = central_moments[2];
        // Skewness: normalized third central moment (asymmetry)
        let spectral_skewness = if spectral_variance > 0.0 {
            central_moments[3] / spectral_variance.powf(1.5)
        } else {
            0.0
        };
        // Kurtosis: normalized fourth central moment minus 3 (excess kurtosis)
        let spectral_kurtosis = if spectral_variance > 0.0 {
            central_moments[4] / spectral_variance.powf(2.0) - 3.0
        } else {
            0.0
        };
        // Spectral flatness: ratio of geometric to arithmetic mean
        let geometric_mean = if normalized_psd.iter().all(|&p| p > 0.0) {
            let log_sum: f64 = normalized_psd.iter().map(|&p| p.ln()).sum();
            (log_sum / normalized_psd.len() as f64).exp()
        } else {
            0.0
        };
        let arithmetic_mean: f64 = normalized_psd.iter().sum::<f64>() / normalized_psd.len() as f64;
        let spectral_flatness = if arithmetic_mean > 0.0 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        };
        // Spectral entropy: Shannon entropy of the PSD
        let spectral_entropy: f64 = normalized_psd
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        // Moment ratios for feature extraction (normalized moments)
        let moment_ratios: Vec<f64> = (0..self.n)
            .map(|i| {
                if i == 0 || raw_moments[1] == 0.0 {
                    0.0
                } else {
                    raw_moments[i + 1] / raw_moments[1].powf((i + 1) as f64)
                }
            })
            .collect();

        HigherOrderMomentsResult {
            spectral_mean,
            spectral_variance,
            spectral_skewness,
            spectral_kurtosis,
            spectral_flatness,
            spectral_entropy,
            moment_ratios,
            cumulants,
            harmonic_moments: HashMap::new(),
        }
    }

    /// Compute raw moments: m_k = sum(p_i * f_i^k)
    ///
    /// # Arguments
    /// * `psd` - Normalized power spectral density
    /// * `frequencies` - Frequency bins
    /// * `max_order` - Maximum moment order to compute
    fn compute_raw_moments(&self, psd: &[f64], frequencies: &[f64], max_order: usize) -> Vec<f64> {
        let mut moments = vec![0.0; max_order + 1];

        for (order, moment) in moments.iter_mut().enumerate().take(max_order + 1) {
            let sum: f64 = psd
                .iter()
                .zip(frequencies.iter())
                .map(|(&p, &f)| p * f.powf(order as f64))
                .sum();
            *moment = sum;
        }
        moments
    }

    /// Compute central moments: m_k = sum(p_i * (f_i - mean)^k)
    ///
    /// # Arguments
    /// * `psd` - Normalized power spectral density
    /// * `frequencies` - Frequency bins
    /// * `mean` - Spectral centroid (first moment)
    /// * `max_order` - Maximum moment order to compute
    fn compute_central_moments(
        &self,
        psd: &[f64],
        frequencies: &[f64],
        mean: f64,
        max_order: usize,
    ) -> Vec<f64> {
        let mut central_moments = vec![0.0; max_order + 1];

        for (order, central_moment) in central_moments.iter_mut().enumerate().take(max_order + 1) {
            let sum: f64 = psd
                .iter()
                .zip(frequencies.iter())
                .map(|(&p, &f)| p * (f - mean).powf(order as f64))
                .sum();
            *central_moment = sum;
        }
        central_moments
    }
    /// Compute cumulants from raw moments using the moment-cumulant relationship
    /// Cumulants are more robust to outliers than moments
    ///
    /// # Arguments
    /// * `moments` - Raw moments array
    fn compute_cumulants(&self, moments: &[f64]) -> Vec<f64> {
//        let moments = Self::f64_slice_to_f64_vec(moments);
        let n = self.n.min(moments.len());
        let mut cumulants: Vec<f64> = vec![0.0; n];

        // Cumulant relationships (up to 5th order)
        if !moments.is_empty() {
            cumulants[0] = moments[0]; // κ₁ = μ₁
        }
        if moments.len() >= 2 {
            //cumulants[1] = moments[1]; // κ₂ = μ₂
            cumulants[1] = moments[1] - moments[0].powf(2.0); // κ₂ = μ₂ - μ₁²
        }
        if moments.len() >= 3 {
            cumulants[2] = moments[2] - moments[1].powf(2.0); // κ₃ = μ₃ - μ₁²
        }
        if moments.len() >= 4 {
            // κ₄ = μ₄ - 3μ₂μ₁ + 2μ₁³
            cumulants[3] = moments[3] - 3.0 * moments[2] * moments[1] + 2.0 * moments[1].powf(3.0);
        }
        if moments.len() >= 5 {
            // κ₅ = μ₅ - 4μ₄μ₁ - 3μ₂² + 12μ₂μ₁² - 6μ₁⁴
            cumulants[4] = moments[4] - 4.0 * moments[3] * moments[1] - 3.0 * moments[2].powf(2.0)
                + 12.0 * moments[2] * moments[1].powf(2.0)
                - 6.0 * moments[1].powf(4.0);
        }
        //Self::f64_slice_to_f64_vec(&cumulants)
        cumulants
    }
    /// Get standard frequency bands for audio analysis
    pub fn get_standard_frequency_bands(&self) -> Vec<FrequencyBand> {
        vec![
            FrequencyBand::new(0.0, 1000.0, "LowFreq"),
            FrequencyBand::new(1000.0, 5000.0, "MidLowFreq"),
            FrequencyBand::new(5000.0, 15000.0, "MidFreq"),
            FrequencyBand::new(15000.0, 30000.0, "MidHighFreq"),
            FrequencyBand::new(30000.0, 60000.0, "HighFreq"),
            FrequencyBand::new(60000.0, self.sampling_rate / 2.0, "VeryHighFreq"),
        ]
    }

    /// Create custom frequency bands around specified center frequencies
    ///
    /// # Arguments
    /// * `centers` - Center frequencies for band groups
    /// * `bands` - Number of bands per center frequency
    /// * `band_width` - Width of each band in Hz
    pub fn get_custom_frequency_bands(
        &self,
        centers: &[f64],
        bands: usize,
        band_width: f64,
    ) -> Vec<FrequencyBand> {
        let mut result = Vec::new();

        for &center in centers {
            let start = center - band_width * (bands as f64) / 2.0;

            for i in 0..bands {
                let start_freq = start + band_width * i as f64;
                let end_freq = start_freq + band_width;
                let name = format!("Band{}_Center{}", i + 1, center as usize);

                result.push(FrequencyBand::new(start_freq, end_freq, &name));
            }
        }
        result
    }

    /// Extract features optimized for machine learning applications
    /// Creates a fixed-size feature vector from the analysis results
    ///
    /// # Arguments
    /// * `result` - Analysis results to extract features from
    pub fn extract_features_for_ml(&self, result: &HigherOrderMomentsResult) -> Vec<f64> {
        // Calculate feature vector size: 6 basic + 2*(N-1) moments/cumulants + 3 derived = 2*(N-1) + 9
        let mut features = Vec::with_capacity(2 * (self.n - 1) + 9);

        // Basic spectral features (6 features)
        features.push(result.spectral_mean);
        features.push(result.spectral_variance);
        features.push(result.spectral_skewness);
        features.push(result.spectral_kurtosis);
        features.push(result.spectral_flatness);
        features.push(result.spectral_entropy);

        // Higher-order moment ratios (N-1 features, skip the 0th)
        for i in 1..self.n {
            if i < result.moment_ratios.len() {
                features.push(result.moment_ratios[i]);
            } else {
                features.push(0.0);
            }
        }

        // Cumulants (N-1 features, skip the 0th)
        for i in 1..self.n {
            if i < result.cumulants.len() {
                features.push(result.cumulants[i]);
            } else {
                features.push(0.0);
            }
        }

        // Derived features for better ML performance (3 features)
        features.push(result.spectral_skewness.abs()); // Absolute skewness
        features.push(result.spectral_kurtosis + 3.0); // Non-excess kurtosis
        features.push(result.spectral_variance.abs().sqrt()); // Standard deviation

        features
    }
}

// Additional utility functions and implementations

impl Default for HigherOrderMomentsAnalyzer {
    fn default() -> Self {
        Self::new_default()
    }
}

impl HigherOrderMomentsResult {
    /// Create a new empty result for initialization
    pub fn new_empty() -> Self {
        Self {
            spectral_mean: 0.0,
            spectral_variance: 0.0,
            spectral_skewness: 0.0,
            spectral_kurtosis: 0.0,
            spectral_flatness: 0.0,
            spectral_entropy: 0.0,
            moment_ratios: Vec::new(),
            cumulants: Vec::new(),
            harmonic_moments: HashMap::new(),
        }
    }

    /// Get a summary of the most important features
    pub fn get_summary(&self) -> String {
        format!(
            "Mean: {:.1}Hz, Var: {:.3}, Skew: {:.2}, Kurt: {:.2}, Flat: {:.3}, Ent: {:.2}",
            self.spectral_mean,
            self.spectral_variance,
            self.spectral_skewness,
            self.spectral_kurtosis,
            self.spectral_flatness,
            self.spectral_entropy
        )
    }
}

/// Example usage and demonstration
/// This would typically be in a separate example file or documentation
pub fn example_usage() {
    println!("=== Higher-Order Moments Analyzer Example ===");

    // Create analyzer with custom parameters
    let mut analyzer = HigherOrderMomentsAnalyzer::new(96000.0, 6);

    // Generate example signal: chirp from 1kHz to 5kHz
    let signal: Vec<f64> = (0..8192)
        .map(|i| {
            let t = i as f64 / 96000.0;
            let freq = 1000.0 + 4000.0 * t; // Linear chirp
            (2.0 * PI * freq * t).sin()
        })
        .collect();

    println!("Analyzing chirp signal ({} samples)...", signal.len());

    // Perform analysis
    let result = analyzer.analyze_full_spectrum(&signal, "blackman");

    println!("Analysis complete!");
    println!("Summary: {}", result.get_summary());

    // Extract features for classification/ML
    let features = analyzer.extract_features_for_ml(&result);
    println!("Extracted {} features for ML", features.len());
}

#[cfg(test)]
mod tests {
    //    use realfft::num_complex::{Complex, Complex64};
    //    use realfft::RealFftPlanner;
    use super::*;

    /// Test function converted from Kotlin main() - creates synthetic signal and analyzes it
    #[test]
    fn test_higher_order_moments_analysis() {
        let signal = sample_hom_signal();

        // Run comprehensive analysis
        analyze_signal(&signal);

        // Basic verification that analysis runs without panicking
        let mut analyzer = HigherOrderMomentsAnalyzer::new_default();
        let result = analyzer.analyze_full_spectrum(&signal, "hanning");

        // Verify that we get reasonable values
        assert!(result.spectral_mean > 0.0);
        assert!(result.spectral_variance >= 0.0);
        assert!(!result.spectral_skewness.is_nan());
        assert!(!result.spectral_kurtosis.is_nan());
        assert!(result.spectral_flatness >= 0.0 && result.spectral_flatness <= 1.0);
        assert!(result.spectral_entropy >= 0.0);
        assert!(!result.moment_ratios.is_empty());
        assert!(!result.cumulants.is_empty());
        println!("Analysis result: {result:?}");
        println!("✅ All tests passed! Analysis completed successfully.");
        let custom_bands = analyzer.get_custom_frequency_bands(&[85_000.0], 1, 3000.0);
        //let band_results = analyzer.analyze_frequency_bands(signal, bands, "hanning");
        let band_results = analyzer.analyze_frequency_bands(&signal, &custom_bands, "hanning");
        println!("Bands result: {band_results:?}");
    }

    fn sample_hom_signal() -> Vec<f64> {
        // Create test signal with multiple harmonics and noise
        let signal_length = 4096;
        let sampling_rate = 192000.0;

        // Generate complex test signal with:
        // - Fundamental frequency at 10kHz
        // - Second harmonic at 20kHz (half amplitude)
        // - Third harmonic at 30kHz (30% amplitude)
        // - Nonlinear distortion (creates higher-order moments)
        // - Gaussian noise
        let signal: Vec<f64> = (0..signal_length)
            .map(|i| {
                let t = i as f64 / sampling_rate;
                // Harmonic components
                let fundamental = (2.0 * PI * 10000.0 * t).sin();
                let second_harmonic = 0.5 * (2.0 * PI * 20000.0 * t).sin();
                let third_harmonic = 0.3 * (2.0 * PI * 30000.0 * t).sin();
                // Nonlinear component (creates interesting higher-order moments)
                let nonlinear = 0.1 * (2.0 * PI * 10000.0 * t).sin().powf(3.0);
                fundamental + second_harmonic + third_harmonic + nonlinear //+ noise
            })
            .collect();
        signal
    }

    /// Test individual components of the analyzer
    #[test]
    fn test_analyzer_components() {
        let mut analyzer = HigherOrderMomentsAnalyzer::new(48000.0, 4);

        // Test window function generation
        let window = analyzer.get_window(1024, "hanning");
        assert_eq!(window.len(), 1024);
        assert!(window[0] >= 0.0 && window[0] <= 1.0);
        assert!(window[512] >= 0.0 && window[512] <= 1.0);

        // Test FFT functionality with simple signal
        let test_signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * PI * i as f64 / 64.0).sin())
            .collect();
        let fft_result = analyzer.compute_fft(&test_signal);
        assert_eq!(fft_result.len(), 64);

        // Test frequency band creation
        let bands = analyzer.get_standard_frequency_bands();
        assert!(!bands.is_empty());
        assert_eq!(bands[0].name, "LowFreq");

        // Test custom bands
        let custom_bands = analyzer.get_custom_frequency_bands(&[1000.0, 2000.0], 2, 100.0);
        assert_eq!(custom_bands.len(), 4); // 2 centers × 2 bands each

        println!("✅ Component tests passed!");
    }
    /// Test feature extraction for ML
    #[test]
    fn test_ml_features() {
        let mut analyzer = HigherOrderMomentsAnalyzer::new_default();

        // Simple test signal
        let signal: Vec<f64> = (0..1024)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        let result = analyzer.analyze_full_spectrum(&signal, "hanning");
        let features = analyzer.extract_features_for_ml(&result);

        // Should have 2*(N-1) + 9 features where N=5, so 2*4 + 9 = 17 features
        assert_eq!(features.len(), 17);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite: {feature}");
        }

        println!("✅ ML feature extraction test passed!");
    }
    /// Print detailed analysis results in a formatted way
    fn print_ho_results(name: &str, result: &HigherOrderMomentsResult) {
        println!("--- {name} ---");
        println!("Spectral Mean: {:.2} Hz", result.spectral_mean);
        println!("Spectral Variance: {:.6}", result.spectral_variance);
        println!("Spectral Skewness: {:.4}", result.spectral_skewness);
        println!("Spectral Kurtosis: {:.4}", result.spectral_kurtosis);
        println!("Spectral Flatness: {:.6}", result.spectral_flatness);
        println!("Spectral Entropy: {:.4}", result.spectral_entropy);

        let cumulants_str: Vec<String> = result
            .cumulants
            .iter()
            .map(|&x| format!("{x:.4}"))
            .collect();
        println!("Cumulants: {}", cumulants_str.join(", "));
        println!();
    }

    /// Analyze frequency bands and print results for significant bands only
    fn bands_stat(
        analyzer: &mut HigherOrderMomentsAnalyzer,
        signal: &[f64],
        bands: &[FrequencyBand],
        window: &str,
    ) {
        //let band_results = analyzer.analyze_frequency_bands(signal, bands, "hanning");
        let band_results = analyzer.analyze_frequency_bands(signal, bands, window);

        for (band_name, result) in band_results {
            // Only show bands with significant spectral variance
            if result.spectral_variance > 1e-10 {
                print_ho_results(&band_name, &result);
            }
        }
    }

    /// Main analysis function demonstrating all capabilities
    fn analyze_signal(signal: &[f64]) {
        let mut analyzer = HigherOrderMomentsAnalyzer::new_default();

        // Full spectrum analysis
        println!("=== Full Spectrum Higher-Order Moments Analysis ===");
        let full_result = analyzer.analyze_full_spectrum(signal, "hanning");
        print_ho_results("Full Spectrum", &full_result);

        // Standard frequency band analysis
        println!("\n=== Standard Frequency Bands Analysis ===");
        let bands = analyzer.get_standard_frequency_bands();
        bands_stat(&mut analyzer, signal, &bands, "hanning");

        // Custom frequency band analysis
        println!("\n=== Custom Frequency Bands Analysis ===");
        let custom_bands = analyzer.get_custom_frequency_bands(&[85000.0], 1, 3000.0);
        bands_stat(&mut analyzer, signal, &custom_bands, "hanning");

        // Harmonic analysis
        println!("\n=== Harmonic Frequency Analysis ===");
        let harmonic_result = analyzer.analyze_around_harmonics(signal, 10000.0, 5, 100.0);
        print_ho_results("Harmonic Regions", &harmonic_result);

        // Machine learning feature extraction
        println!("\n=== Machine Learning Features ===");
        let features = analyzer.extract_features_for_ml(&full_result);
        println!("Number of features: {}", features.len());

        let features_str: Vec<String> = features.iter().map(|&x| format!("{x:.4}")).collect();
        println!("Features: {}", features_str.join(", "));
    }

    // to compare with App4.checkComputeFft()
    #[test]
    #[ignore]
    fn test_compute_fft() {
        // Test with a simple sine wave, but not 2.0 * PI -> 2.123 * PI
        let test_signal: Vec<f64> = (0..8)
            .map(|i| (2.123 * PI * i as f64 / 8.0).sin())
            .collect();
        // the same
        // [0.0, 0.0980171, 0.195090, 0.290284, 0.382683, 0.4713967, 0.555570 ...
        //let fft_result = analyzer.compute_fft(&test_signal);
        // different only Complex.re, and it is very small values
        let fft_result = utils::fft::fft64_forward(&test_signal, test_signal.len());
        println!("FFT Result:");
        fft_result.iter().for_each(|c| {
            println!("{} + {}i", c.re, c.im);
        });
        //let size = fft_result.len();
        println!(
            "amplitudes: {:?})",
            utils::fft::to64_amplitudes(&fft_result, test_signal.len())
        );
        println!(
            "phases: {:?})",
            utils::fft::to64_phases(&fft_result, test_signal.len())
        );
    }
}
