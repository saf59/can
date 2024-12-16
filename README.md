Как работают трансформеры: разбираем математику / Хабр
https://habr.com/ru/articles/785474/

# ENV
- CUDA_TOOLKIT_ROOT_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
- CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"

```
error: failed to run custom build command for `cudarc v0.12.0`
note: To improve backtraces for build dependencies, set the CARGO_PROFILE_DEV_BUILD_OVERRIDE_DEBUG=true environment variable to enable debug information generation.
Caused by:
process didn't exit successfully: `D:\projects\rust\can\target\debug\build\cudarc-7b87b9b6059fabbd\build-script-build` (exit code: 101)
--- stdout
cargo:rerun-if-changed=build.rs
cargo:rerun-if-env-changed=CUDA_ROOT
cargo:rerun-if-env-changed=CUDA_PATH
cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR
--- stderr
thread 'main' panicked at C:\Users\alsh\.cargo\registry\src\index.crates.io-6f17d22bba15001f\cudarc-0.12.0\build.rs:82:14:
Unsupported cuda toolkit version: `10.2`. Please raise a github issue.
stack backtrace:
0: std::panicking::begin_panic_handler
at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\std\src\panicking.rs:652
1: core::panicking::panic_fmt
at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\core\src\panicking.rs:72
2: <core::ops::control_flow::ControlFlow<B,C> as core::ops::try_trait::Try>::branch
3: <core::ops::control_flow::ControlFlow<B,C> as core::ops::try_trait::Try>::branch
4: core::ops::function::FnOnce::call_once
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```
test

