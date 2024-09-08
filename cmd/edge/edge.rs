use can_data::data_set;
use can_utils::l2_loss;
use candle_core::{DType, Device, Module};
use candle_nn::Init::Const;
use candle_nn::{Conv2d, Conv2dConfig, Init, Optimizer, VarBuilder, VarMap};
use std::error::Error;
use std::time::Instant;

const LEARNING_RATE: f64 = 0.1;
const FILE: &str = "checkpoint.safetensors";
const INIT_WS: Init = Const(123.); // To repeat must be Const. Else: DEFAULT_KAIMING_NORMAL;

// the same as 6.2.4. Learning a Kernel: https://d2l.djl.ai/chapter_convolutional-neural-networks/conv-layer.html
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let start = Instant::now();
    let _ = build_model(&device, &varmap);
    println!("{:?}, Cuda:{:?}", Instant::now().duration_since(start), &device.is_cuda());
    println!("save: {:?}", varmap.data());
    varmap.save(FILE)?; // save is Ok
    restore(FILE);
    Ok(())
}

fn restore<P: AsRef<std::path::Path>>(path: P) {
    // restore only after set VarBuilder configuration (ws2)
    let device = Device::cuda_if_available(0).unwrap();
    let mut varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F64, &device);
    let ws2 = vb2.get_with_hints((1, 1, 1, 2), "weight", INIT_WS).unwrap();
    let conv2dn = Conv2d::new(ws2, None, Conv2dConfig::default()); //to print same result
    varmap2.load(&path).unwrap(); // back
    println!("load: {:?}", varmap2.data());
    print_tensor(conv2dn);
}

fn build_model(device: &Device, varmap: &VarMap) -> Result<(), Box<dyn Error>> {
    let vb = VarBuilder::from_varmap(varmap, DType::F64, device);
    let (x, y) = data_set(device);
    // filter: in_channels, out_channels, kernel_height, kernel_width
    let ws = vb.get_with_hints((1, 1, 1, 2), "weight", INIT_WS)?;
    let conv2d = Conv2d::new(ws, None, Conv2dConfig::default());
    let mut optimizer = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    for _i in 0..400 {
        let y_hat = conv2d.forward(&x)?;
        let loss = l2_loss(&y_hat, &y);
        let _ = optimizer.backward_step(&loss);
        //        if (i + 1) % 2 == 0 { print!("batch {} loss: {}\r", i + 1, loss); }
    }
    // target is: [1.0, -1.0]
    print_tensor(conv2d);
    Ok(())
}

fn print_tensor(conv2d: Conv2d) {
    println!(
        "Tensor{:.1?}",
        conv2d
            .weight()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()
    );
}
