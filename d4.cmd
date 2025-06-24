set JOINED=-j
cargo run --bin detect4 -- %JOINED% test_data/4/in.wav
cargo run --bin detect4 -- %JOINED% test_data/4/below.wav
cargo run --bin detect4 -- %JOINED% test_data/4/above.wav
