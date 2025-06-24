cargo run --bin train -- --model-type classification --batch-size 100 --train-part 1.0 -e 0 --activation relu --hidden "100,40,10" --alg-type hom --buff-size small --flag true -n 34
cargo build --release --bin test3 --bin detect4

