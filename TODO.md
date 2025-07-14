
Можно сразу и на Kotlin 
+ +проверить 4 hom impulse без 85, но с нормализацией   
  **BAD**
+ проверить 4 V2 hom impulse без 85, но с нормализацией   
  **BAD**
+ проверить hom на RAW (без пиков и 85) но с нормализацией 
  not20000RawFilter как buildHOM, но 4
+ проверить hom на RAW Stage 3 (без пиков и 85) но с нормализацией
  все есть по W:\data\medius\stage3\sample\src.csv
  val samples = Stage3.samples(), getUseful(sample: Int, xi: Int, yi: Int,..
  **SOME** v1 and v2 are without difference, but with normalization  
  a lot of **correlations close or more than 0.5 !!!**
  so model on whole useful data **is possible**
+ Возможно в 3 добавить scale, например в виде sampleRate scale  
  val scaledRate = RATE_192_000.toDouble() * 37523.4522 / sr.frequency.toDouble()  
  для инициализации HigherOrderMomentsAnalyzer(samplingRate = scaledRate))
  **SOME** v1 and v2 are without difference, but with normalization
- HOM V1 и V2 нужно сравнить - похоже в них просто сдвиг
-----------
- поменять все действия с fft c f32 на \<T\>
  по ходу ho будет ясно чего не хватает
- ho -> f64, f32 только в parser hob(параметры)
  1. переносим f32_slice_to_f64_vec в parser
  2. ho в f64  + резервируем в отдельный файл
  3. hob signal в f64
  4. result в f64
  5. tests
- добавить массивы нормализации в
  - отдельный файл
  - или внутрь модели


