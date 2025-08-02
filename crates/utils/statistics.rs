#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]

pub fn summary_statistics(data: &[f32]) -> anyhow::Result<Vec<f32>> {
    let stats: Vec<f32> = vec![
        min(data),
        max(data),
        sum(data),
        mean(data),
        geometric_mean(data),
        sample_variance(data),
        population_variance(data),
        second_moment(data),
        sum_sq(data),
        sample_standard_deviation(data),
        sum_of_logs(data),
        population_skewness(data),
        kurtosis(data),
    ];
    Ok(stats)
}
pub fn stat3(data: &[f32]) -> anyhow::Result<Vec<f32>> {
    let stats: Vec<f32> = vec![
        median(data),
        sample_variance(data),
        std_deviation(data),
        population_skewness(data),
        kurtosis(data),
        rms(data),
        mad(data),
        crest(data),
        energy(data),
        entropy(data),
    ];
    Ok(stats)
}
#[test]
fn test_stats() {
    let data2 = vec![1.0E-5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result2 = vec![
        1.0E-5,
        10.0,
        55.00001,
        5.000_001,
        1.386_106_6,
        10.999_99,
        9.999_99,
        109.999_9,
        385.0000000001,
        3.316_623_2,
        3.591_487_2,
        0.0,
        1.78,
    ];
    let data = vec![1.0, 2.0, 5.0, 5.0, 8.0, 8.0, 9.0];
    let result = vec![
        1.0,
        9.0,
        38.0,
        5.428_571,
        4.335_668_6,
        9.619_047,
        8.244_898,
        57.714_287,
        264.0,
        3.101_459,
        10.268_13,
        -0.288_166_7,
        1.624_522_1,
    ];
    let _ = compare_stats(data, result);
    let _ = compare_stats(data2, result2);

    fn compare_stats(data: Vec<f32>, estimated: Vec<f32>) -> anyhow::Result<()> {
        let stat_name = vec![
            "min",
            "max",
            "sum",
            "mean",
            "geometric mean",
            "variance",
            "population variance",
            "second moment",
            "sum of squares",
            "standard deviation",
            "sum of logs",
            "skew",
            "kurtosis",
        ];
        let real = summary_statistics(&data)?;
        //println!("real:{:?}",real);
        //println!("estimated:{:?}",estimated);
        for i in 0..real.len() {
            //println!("{:19} real:{:9.5} estimated:{:9.5} {}", stat_name[i], real[i], estimated[i], (real[i] - estimated[i]).abs() < 0.0001);
            let msg = format!(
                "{:19} real:{:9.5} estimated:{:9.5}",
                stat_name[i], real[i], estimated[i]
            );
            assert!((real[i] - estimated[i]).abs() < 0.0001, "{}", &msg); //, );
        }
        //println!("");
        Ok(())
    }
}

pub fn min(data: &[f32]) -> f32 {
    data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
}

pub fn max(data: &[f32]) -> f32 {
    data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
}

pub fn sum(data: &[f32]) -> f32 {
    data.iter().fold(0.0, |sum, v| sum + v)
}

pub fn mean(data: &[f32]) -> f32 {
    sum(data) / (data.len() as f32)
}

pub fn sample_variance(data: &[f32]) -> f32 {
    sum_squared_deltas(data) / ((data.len() - 1) as f32)
}

pub fn population_variance(data: &[f32]) -> f32 {
    sum_squared_deltas(data) / (data.len() as f32)
}

fn sum_squared_deltas(data: &[f32]) -> f32 {
    let mean = mean(data);
    let mut ssd = 0.0;
    data.iter().for_each(|v| {
        let delta = v - mean;
        ssd += delta * delta;
    });
    ssd
}

pub fn sample_standard_deviation(data: &[f32]) -> f32 {
    f32::sqrt(sample_variance(data))
}

pub fn geometric_mean(data: &[f32]) -> f32 {
    let sum_of_logs = data.iter().fold(0.0_f32, |sum, num| sum + num.ln());
    (sum_of_logs / data.len() as f32).exp()
}

pub fn sum_of_logs(data: &[f32]) -> f32 {
    data.iter().fold(0.0_f32, |sum, num| {
        sum + (if num != &0.0_f32 { num } else { &1.0E-5_f32 }).ln()
    })
}

pub fn sum_sq(data: &[f32]) -> f32 {
    data.iter().fold(0.0_f32, |sum, num| sum + num * num)
}

fn second_moment(data: &[f32]) -> f32 {
    let mean = mean(data);
    let mut ssd = 0.0_f32;
    data.iter().for_each(|v| {
        let delta = v - mean;
        ssd += delta * delta;
    });
    ssd
}

pub fn population_skewness(data: &[f32]) -> f32 {
    let mean = mean(data);
    let sum3 = data.iter().fold(0.0, |sum, v| {
        let delta = v - mean;
        sum + delta * delta * delta
    });

    let ssv = population_variance(data);
    let n = data.len() as f32;
    let variance = f32::sqrt(ssv);
    if variance == 0.0 {
        return 0f32;
    }
    sum3 / n / (variance * variance * variance)
}

pub fn to_db(data: &[f32]) -> Vec<f32> {
    data.iter().map(|amplitude|10.0 * (amplitude.abs().log10())).collect()
}
pub fn average_by_column(data: &[Vec<f32>]) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    let cols = data[0].len();
    (0..cols)
        .map(|col| {
            data.iter().map(|row| row[col]).sum::<f32>() / data.len() as f32
        })
        .collect()
}
pub fn skew(data: &[f32]) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let mean = data.iter().copied().sum::<f32>() / n as f32;
    let std_dev = {
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        var.sqrt()
    };
    if std_dev == 0.0 {
        return 0.0;
    }
    let mut skew = 0.0;
    for &x in data {
        skew += ((x - mean) / std_dev).powi(3);
    }
    skew / n as f32
}

pub fn kurtosis(values: &[f32]) -> f32 {
    let mut kurtosis = 0.0;
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let std_deviation = std_deviation(values);
    if !values.is_empty() && std_deviation > 0.0 {
        // values.iter().for_each(|&value| kurtosis += ((value - mean) / std_deviation).powf(4.0));
        for value in values {
            kurtosis += ((value - mean) / std_deviation).powf(4.0);
        }
        kurtosis /= values.len() as f32;
    }
    kurtosis
}

pub fn std_deviation(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let mut variance = 0.0;
    // values.iter().for_each(|&value| variance += (value - mean).powf(2.0));
    for value in values {
        variance += (value - mean).powf(2.0);
    }
    variance /= values.len() as f32;
    variance.sqrt()
}

fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = values.len();
    let mid = len / 2;
    if len % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn rms(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum_of_squares: f32 = data.iter().map(|&x| x * x).sum();
    (sum_of_squares / data.len() as f32).sqrt()
}

fn mad(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let mad: f32 = data.iter().map(|&x| (x - mean).abs()).sum::<f32>() / data.len() as f32;
    mad
}
fn crest(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let rms = rms(data);
    if rms == 0.0 {
        return 0.0;
    }
    let peak = data.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    peak / rms
}

fn energy(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum_of_squares: f32 = data.iter().map(|&x| x * x).sum();
    sum_of_squares / data.len() as f32
}

fn entropy(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut psd: Vec<f32> = data.iter().map(|&x| x * x / data.len() as f32).collect();
    let psd_sum: f32 = psd.iter().sum();

    if psd_sum == 0.0 {
        return 0.0;
    }

    psd.iter_mut().for_each(|x| *x /= psd_sum);

    let mut entropy = 0.0;
    for &p in &psd {
        if p != 0.0 {
            entropy += p * p.ln();
        }
    }

    -entropy
}
