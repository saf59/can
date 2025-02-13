## Запуск с CUDA ##
1) Установить CUDA 12.2
	https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
> ### ВНИМАНИЕ ###
	установка на атомате скорее всего не пройдет, НО
    можно перед выходом переименовать C:\Temp\cuda напимер в C:\Temp\cuda_setup
    и во время следующего запуска setup: custom setup выбрать только 3 пункта - runtime && compiler & VS tools integration
> проверить CUDA_PATH
2) Установить CUDNN (соответствующий CUDA https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html)
    т.е. 9.6 для 12.*
	https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
	станет норм, но пути не исправит
> проверить CUDNN_PATH

3) В PATH добавить пути к bin из CUDA && CUDNN
4) Для runtime достаточно, но Rust нужна еще и компиляция, поэтому надо
	- или скопировать lib, include из CUDNN в CUDA
	- или все это тоже добавить в PATH
5) В Rust добавить dependency 
```
cudarc = { version = "0.12.2", features = ["std", "cublas", "cublaslt", "curand", "driver", "nvrtc", "f16", "cuda-version-from-build-system", "dynamic-linking"], default-features=false }

```
6) В любой участвующий в сборке src добавить 
```
#[allow(unused_imports)]
#[cfg(feature = "cuda")]
use cudarc;

#[allow(unused_imports)]
#[cfg(feature = "cudnn")]
use cudarc::cudnn;
```
иначе соберет без них и будет на Cpu

!!! Похоже не так.
Важно, чтобы candle-core собралось с --features cuda,cudnn
!!! 
```https://doc.rust-lang.org/cargo/reference/features.html
Features for the package being built can be enabled on the command-line with flags such as --features. 
Features for dependencies can be enabled in the dependency declaration in Cargo.toml.
```
Т.е. use уже не имеет смысла так как у cudarc cuda отсутствует
И вообще важна только фича в toml: candle-core = { version = "0.8.1", features = ["cuda"] }
ее можно добавить в головной toml или в дочерние и, возможно, в сецию [features]
И еще!
Добавление --features cuda,cudnn после соответствующей сборки candle-core не имеет смысла
!!! Надо делать cargo clean

7) теперь при сбоке, если есть --features cuda,cudnn по одной или вместе, соберется с использованием cuda|cudnn, а если нету то на Cpu
Если при сборке не найдет dll-h-lib ... кричит и бросает
8) Rust runtime должен все найти, нет - будет Cpu
Рекомендации в хвосте https://github.com/huggingface/candle не помогли
Все собирается, но пишет ##Cpu##
!!!CUDA ПОКА НЕ РАБОТАЕТ!!! Искать ПОЧЕМУ?
ага, если --features cuda то candle-core = "0.8.1" собирается БЕЗ CUDA
а если в toml: candle-core = { version = "0.8.1", features = ["cuda"] }
то
  nvcc fatal   : Cannot find compiler 'cl.exe' in PATH
Но, если добавить путь к "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64"
  cannot find struct, variant or union type `cudaExternalMemoryHandleDesc_st__bindgen_ty_1__bindgen_ty_1` in module `sys`
или
```
error[E0425]: cannot find function `cudaImportExternalMemory` in module `sys`
 --> C:\Users\alsh\.cargo\registry\src\index.crates.io-6f17d22bba15001f\cudarc-0.12.2\src\runtime\result.rs:997:14
997 |   sys::cudaImportExternalMemory(external_memory.as_mut_ptr(), &handle_description)
 |    ^^^^^^^^^^^^^^^^^^^^^^^^ not found in `sys`
```
т.е. под WIN в sys не хватает многого