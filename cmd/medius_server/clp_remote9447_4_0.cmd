curl -X "POST" ^
  "http://app.ispredict.com:9447/detect4" ^
  -H "accept: */*" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@D:\projects\rust\can\test_data\4\in.wav;type=audio/wav" ^
  -F "signature=75462951157"