This is a local work position detector with a built-in model.
Can be compiled for any platform.
The detection model is inside the file and to change it you need to rebuild the file.

```
**Command line arguments for the detecting program**
```
Usage: detect3.exe [OPTIONS] <WAV>
```
Arguments:
```
<WAV>  Path to stereo *.wav file with sample rate 192_000
```
Options:
```
  -v                  Verbose mode
  -f <FREQUENCY>      Laser frequency [default: 37523.4522]                                       
  -h, --help          Print help
``` 
**Copyright 2025 The ISPredict.  
Licensed under the Apache License, Version 2.0 or the MIT license at your option.**