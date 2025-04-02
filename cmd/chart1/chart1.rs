#![cfg(target_os = "windows")]
#![allow(unused_imports)]
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;

#[derive(Serialize, Deserialize)]
struct Timeseries {
    data: Vec<Timepoint>,
    unit: String,
    label: String,
    #[serde(rename = "predictTime")]
    predict_time: u64,
}

#[derive(Serialize, Deserialize)]
struct Timepoint {
    timestamp: u64,
    value: f64,
}


fn main() -> std::io::Result<()> {
    let data_dir = "T:/EnerReg/14102024/memo/";
    let h_file = "generation@id=318_L2h.json";
    let a_file = "generation@id=318_L2a.json";
    let real_file = "data_1730365200000_1730523600000.csv";

    let (a_time, a_data) = get_timeseries_data(data_dir, a_file)?;
    let (h_time, h_data) = get_timeseries_data(data_dir, h_file)?;
    let (real_time, real_data) = get_real_data(data_dir, real_file)?;

    let layout = plotly::Layout::new()
        .title("Result Comparison")
        .x_axis(plotly::layout::Axis::new().title("Time"))
        .y_axis(plotly::layout::Axis::new().title("Energy"));

    let a_trace = date_time_scatter_trace("318_L2a", &a_time, &a_data);
    let h_trace = date_time_scatter_trace("318_L2h", &h_time, &h_data);
    let real_trace = date_time_scatter_trace("Real", &real_time, &real_data);

    let mut plot = plotly::Plot::new();
    plot.add_trace(a_trace);
    plot.add_trace(h_trace);
    plot.add_trace(real_trace);
    plot.set_layout(layout);
    plot.write_html("charts/chart1.html");
    plot.show();
    Ok(())
}
fn get_timeseries_data(data_dir: &str, file_name: &str) -> Result<(Vec<DateTime<Utc>>, Vec<f64>), std::io::Error> {
    let mut file_path = String::from(data_dir);
    file_path.push_str(file_name);

    let mut file_content = String::new();
    File::open(file_path)?.read_to_string(&mut file_content)?;

    let ts: Timeseries = serde_json::from_str(&file_content)?;
    let x: Vec<DateTime<Utc>> = ts.data.iter().map(|tp| Utc.timestamp_opt(tp.timestamp as i64 / 1000, 0).unwrap()).collect();
    let y: Vec<f64> = ts.data.iter().map(|tp| tp.value).collect();

    Ok((x, y))
}
fn get_real_data(data_dir: &str, file_name: &str) -> Result<(Vec<DateTime<Utc>>, Vec<f64>), std::io::Error> {
    let mut file_path = String::from(data_dir);
    file_path.push_str(file_name);

    let mut file_content = String::new();
    File::open(file_path)?.read_to_string(&mut file_content)?;

    let lines: Vec<&str> = file_content.lines().skip(1).collect();
    let x: Vec<DateTime<Utc>> = lines.iter().map(|line| {
        let parts: Vec<&str> = line.split(',').collect();
        Utc.timestamp_opt(parts[0].parse::<i64>().unwrap() / 1000, 0).unwrap()
    }).collect();
    let y: Vec<f64> = lines.iter().map(|line| {
        let parts: Vec<&str> = line.split(',').collect();
        parts[1].parse::<f64>().unwrap()
    }).collect();

    Ok((x, y))
}

fn date_time_scatter_trace(name: &str, x_data: &[DateTime<Utc>], y_data: &[f64]) -> Box<dyn plotly::Trace> {
    plotly::Scatter::new(x_data.to_vec(), y_data.to_vec())
        .mode(plotly::common::Mode::Lines)
        .name(name)
}