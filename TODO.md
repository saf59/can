
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
+ поменять все действия с fft c f32 на \<T\>
  по ходу ho будет ясно чего не хватает
+ ho -> f64, f32 только в parser hob(параметры)
  1. переносим f32_slice_to_f64_vec в parser
  2. ho в f64  + резервируем в отдельный файл
  3. hob signal в f64
  4. result в f64
  5. tests

+ добавить массивы нормализации в
  + отдельный файл (сырой вид при генерации данных)
  + внутрь safetensors - при train (C_100_40_10_H34_ST_100 32_508 -> **32916** )  
    при любом трайн, т.е. ВСЕГДА перезапись norm
    т.е. заполнили и сохранили
  + а в detect_by_single_vec поступают уже нормализированные данные
    можно нормализовать уже там (94), но тогда нужно будет из get_model возвращить и m,m
    нужен ридер из varmap после fill  в 
 
+ ПРОБЛЕМА - flag = надо разделить на 2 разных
  - для HOB  это нормализация norm: `N|▋`
  - для остальных это scale frquency: scale - `T|F`

+ ПРОБЛЕМА - Big|Small - разделить на 2 enum и оба с None
  - для HOB  это impulse|raw - свойство датасета datatype `I|R|▋`
  - для остальных это buffer is Big |Small - свойство буфера при анализе `B|S|▋`

- После правок (было 2, станет 4) нужно будет:
  + добавить и поменять в meta именах,
  + сделать тесты на имена
  + поменяь в hob meta на data_type & norm, flag default = scale 
  + переименвать данные и H34 модели ST->RN, SF->R, BT->IN
  + замеить константы в исходниках
  - добавить и поменять в cmd для H34,
+ БАГ не заполняет MM - (вернее исчезают после load)
  fill_from_file(440) varmap.load(model_path)?;
  loading weights from ./models\C_100_40_10_H34_IN_100\model.safetensors
  если в вармапе есть, а в файле нету -> self.data.iter -> item.load(name, var.device())?;
  т.е. заполнение должно быть после!!
  !!! но при detect_by error: process didn't exit successfully: `target\debug\detect4.exe test_data/4/in.wav` (exit code: 0xc0000005, STATUS_ACCESS_VIOLATION)
  Исправлено fill_norm после fill и с get внутри !!! 
- генератор hob для 3 + signal normalize и scale
+ тестирование текущее на 3 - 100%
- тестирование hob на 3
 
  
