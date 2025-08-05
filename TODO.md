
Можно сразу и на Kotlin 
+ +проверить 4 hom impulse без 85, но с нормализацией   
  **BAD** buildHOM4()
+ проверить 4 V2 hom impulse без 85, но с нормализацией   
  **BAD** buildHOM4()
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
3: --buff-size small --norm false --scale true  // --data-type none
4: --buff-size none --norm true --data-type raw // --scale none
  
- После правок (было 2, станет 4) нужно будет:
  + добавить и поменять в meta именах,
  + сделать тесты на имена
  + поменяь в hob meta на data_type & norm, flag default = scale 
  + переименвать данные и H34 модели ST->RN, SF->R, BT->IN
  + замеить константы в исходниках
  + добавить и поменять в cmd для H34,
+ БАГ не заполняет MM - (вернее исчезают после load)
  fill_from_file(440) varmap.load(model_path)?;
  loading weights from ./models\C_100_40_10_H34_IN_100\model.safetensors
  если в вармапе есть, а в файле нету -> self.data.iter -> item.load(name, var.device())?;
  т.е. заполнение должно быть после!!
  !!! но при detect_by error: process didn't exit successfully: `target\debug\detect4.exe test_data/4/in.wav` (exit code: 0xc0000005, STATUS_ACCESS_VIOLATION)
  Исправлено fill_norm после fill и с get внутри !!! 
+ builder ten 18 by pulses+avg raw
+ type = Ten + 18 -> добавить общность с HOB
+ train - только enum && match 73.3% для raw и 44.86% для impulse
+ builder ten 18, но по всем бинам БЕЗ puslses && avg -> только raw
  36,18,9 ->relu 88.22% = 4557/5075 (518 из 5075 - 10.2% не предсказано)
  40,20,10,5 -> relu6 91.72% = 4655/5075 (420 из 5075 - 8.3% не предсказано)
  30,30,30 -> relu6 95.94% = 4869/5075 (206 из 5075 = 4.1% не предсказано)
  36,36,36 -> relu6 97.48% = 4930/5075 (127 из 5075 = 2.5% не предсказано)
  36,36,36,12 -> relu6 97.85% = 4966/5075 (109 из 5075 = 2.1% не предсказано)
+ ? test - парсинг типа dbuilder
+ добавить в мета номер версии, по умолчанию "", а дальше V1...
и вставить его в конец во все Meta имена - данных, модели, out 
 
  
