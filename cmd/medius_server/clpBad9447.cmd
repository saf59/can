curl -X "POST" ^
  "http://localhost:9447/detect3" ^
  -H "accept: */*" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@read.md;type=audio/wav" ^
  -F "signature=27719681143" ^
  -F "frequency=12123.456"