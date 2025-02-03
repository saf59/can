curl -X "POST" ^
  "http://localhost:8080/detect3" ^
  -H "accept: */*" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@D:\projects\rust\can\test_data\x1_y1.wav;type=audio/wav" ^
  -F "signature=123456728" ^
  -F "frequency=123.456"