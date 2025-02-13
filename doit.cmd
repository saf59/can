@Echo Off
set RUST_BACKTRACE=1
set CP=D:\projects\rust\can\target\release\

set NAME=%1_%2_%3
ECHO Start %NAME%

echo %1 %2 %3> %NAME%.csv 
%CP%dur.exe
if %1 == c (
	call :class %1 %2 %3
) else if %1 == r (
	call :regr %1 %2 %3
) else (
    ECHO Bad type "%1" , exit 
	%CP%dur.exe --stop
    EXIT /B
)
%CP%dur.exe --stop >> %NAME%.csv
%CP%test3 >> %NAME%.csv 
rem meta MUST be embed to detect3
cargo build --release --bin detect3
%CP%dur.exe
%CP%detect3 -f 37523.4522 test_data/x1_y1.wav >> %NAME%.csv
%CP%dur.exe --stop >> %NAME%.csv
ECHO End %1 %2 %3 %NAME%.csv

rem Join %NAME%.csv to one row and add it to the total.csv
setlocal EnableDelayedExpansion
set O=
for /f "tokens=*" %%a in (%NAME%.csv) do set "O=!O!%%a "
echo %O% >> total.csv
endlocal 

GOTO: eof

:regr
echo Train regression %1 %2 %3
%CP%train.exe --model-type regression --batch-size %2 --train-part 1.0 -e 0 --activation %3
%CP%train.exe --learning-rate 0.005 -e 1000
%CP%train.exe --learning-rate 0.0005 -e 1000
%CP%train.exe --learning-rate 0.00005 -e 1000
%CP%train.exe --learning-rate 0.000005 -e 1000
EXIT /B

:class
echo Train classification %1 %2 %3
%CP%train.exe --model-type classification --batch-size %2 --train-part 1.0 -e 0 --activation %3
%CP%train.exe --learning-rate 0.5 -e 1000
%CP%train.exe --learning-rate 0.05 -e 1000
%CP%train.exe --learning-rate 0.005 -e 1000
%CP%train.exe --learning-rate 0.0005 -e 1000
EXIT /B

:eof