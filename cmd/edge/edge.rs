use can_data::data_set;
use can_utils::l2_loss;
use candle_core::{DType, Device, Module};
use candle_nn::Init::Const;
use candle_nn::{Conv2d, Conv2dConfig, Optimizer, VarBuilder, VarMap};

const LEARNING_RATE: f64 = 0.05;
const FILE: &str = "checkpoint.safetensors";

// the same as 6.2.4. Learning a Kernel: https://d2l.djl.ai/chapter_convolutional-neural-networks/conv-layer.html
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let (x, y) = data_set(&device);
    let init_ws = Const(12345.); // To repeat must be Const. Else: DEFAULT_KAIMING_NORMAL;
    // filter: in_channels, out_channels, kernel_height, kernel_width
    let ws = vb.get_with_hints((1, 1, 1, 2,), "weight", init_ws)?;
    let conv2d = Conv2d::new(ws, None, Conv2dConfig::default());
    let mut optimizer = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    for _i in 0..500 {
        let y_hat = conv2d.forward(&x)?;
        let loss = l2_loss(&y_hat, &y);
        let _ = optimizer.backward_step(&loss);
//        if (i + 1) % 2 == 0 { print!("batch {} loss: {}\r", i + 1, loss); }
    }
    // target is: [1.0, -1.0]
    println!("result: {:?}", conv2d.weight().flatten_all()?);

    // save model
    println!("save: {:?}", varmap.data());
    varmap.save(&FILE)?;  // save is Ok

    // restore only after set VarBuilder configuration (ws2)
    let mut varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F64, &device);
    let ws2 = vb2.get_with_hints((1, 1, 1, 2,), "weight", init_ws)?;
    let conv2dn = Conv2d::new(ws2, None, Conv2dConfig::default()); //to print same result
    varmap2.load(&FILE)?; // back
    println!("\nkernel: {:?}", conv2dn.weight().flatten_all()?);
    println!("load: {:?}", varmap2.data());
    Ok(())
}

