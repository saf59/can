#![allow(unused_imports)]
#![cfg(target_os = "windows")]
use medius_data::read_medius_x;
use plotly::{
    common::{Marker, Mode},
    layout::LayoutScene,
    Layout, Plot, Scatter3D,
};
use std::collections::HashMap;
use std::error::Error;
use utils::umap::umap;

pub fn plot_3d_scatter(
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    labels: Vec<String>,
    output_file: &str,
) -> Result<(), Box<dyn Error>> {
    // Check that all vectors have the same length
    let len = x.len();
    if y.len() != len || z.len() != len || labels.len() != len {
        return Err("All data vectors and labels must have the same length".into());
    }
    // Group data points by their labels
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        groups.entry(label).or_default().push(i);
    }
    // Define a list of colors for different labels (cycle through these)
    let colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
        "#FF97FF", "#FECB52", "#FF9DA6", "#72B7B2",
    ];
    let mut color_iter = colors.iter().cycle();
    // Create the plot
    let mut plot = Plot::new();

    for (label, indices) in &groups {
        // Extract coordinates for this group
        let x_vals: Vec<f32> = indices.iter().map(|&i| x[i]).collect();
        let y_vals: Vec<f32> = indices.iter().map(|&i| y[i]).collect();
        let z_vals: Vec<f32> = indices.iter().map(|&i| z[i]).collect();
        // Get the next color from the color list
        let color = *color_iter.next().unwrap();
        // Create a trace for this group with its own color and label
        let trace = Scatter3D::new(x_vals, y_vals, z_vals)
            .mode(Mode::Markers)
            .name(label) // Name appears in the legend
            .marker(Marker::new().size(6).opacity(0.8).color(color));
        // Add the trace to the plot
        plot.add_trace(trace);
    }
    // Configure layout with axis labels
    let layout = Layout::new().scene(
        LayoutScene::new(), //.hover_mode(plotly::layout::HoverMode::Closest)
                            //.x_axis(Axis::new().title("X Axis"))
                            //.y_axis(Axis::new().title("Y Axis"))
                            //.z_axis(Axis::new().title("Z Axis")),
    );
    plot.set_layout(layout);
    plot.write_html(output_file);
    //plot.show();
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let dir = "./data/B260_ST";
    //let labels = read_medius_y(dir.as_ref())?;
    let raw = read_medius_x(dir.as_ref())?;
    let data: Vec<Vec<f32>> = raw.chunks(260).map(|x| x.to_vec()).collect();
    let result = umap(&data, 40, 3, 0.5, 10, 0.8, "euclidean").unwrap();
    let x = result.iter().map(|x| x[0]).collect();
    let y = result.iter().map(|x| x[1]).collect();
    let z = result.iter().map(|x| x[2]).collect();
    // working positions
    //let labels:Vec<String> = labels.iter().map(|x| ((*x as f32) * -0.1 ).to_string()).collect();
    // samples
    let labels = samples(9, 80);
    plot_3d_scatter(x, y, z, labels, "chart2.html")?;
    Ok(())
}
fn samples(n: usize, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for i in 1..=n {
        for _j in 0..width {
            out.push(format!("Sample {i}"));
        }
    }
    out
}
