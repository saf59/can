curl -X "POST" ^
  "http://localhost:9447/detect4" ^
  -H "accept: */*" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@D:\projects\rust\can\test_data\4\below.wav;type=audio/wav" ^
  -F "signature=90809089590"