cargo run --bin train -- --model-type regression --batch-size 1 --train-part 1.0 -e 0 --activation relu --hidden "100,40,10" --alg-type bin --buff-size small --norm false --scale true -n 260
cargo build --release --bin test3 --bin detect3

