@Echo Off
set RUST_BACKTRACE=1
set CP=D:\projects\rust\can\target\release\
SET hidden=%4
IF NOT DEFINED hidden SET hidden="40,10"
SET hidden=%hidden:"=%
set epochs=%5
IF NOT DEFINED epochs SET "epochs=1000"

set NAME=%1_%2_%3_%hidden:,=_%
ECHO Start %NAME%

echo %1 %2 %3 %hidden:,=_%> %NAME%.csv 
%CP%dur.exe
if %1 == c (
	set rate="0.5"
	echo Train classification %NAME%
	%CP%train.exe --model-type classification --batch-size %2 --train-part 1.0 -e 0 --activation %3 --hidden %hidden%
	call :train
) else if %1 == r (
	set rate="0.005"
	echo Train regression %NAME%
	%CP%train.exe --model-type regression --batch-size %2 --train-part 1.0 -e 0 --activation %3 --hidden %hidden%
	call :train
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
EXIT /B

:train 
%CP%train.exe --learning-rate %rate:"=% -e %epochs%
set rate=%rate:0.=0.0%
%CP%train.exe --learning-rate %rate:"=% -e %epochs%
set rate=%rate:0.=0.0%
%CP%train.exe --learning-rate %rate:"=% -e %epochs%
set rate=%rate:0.=0.0%
%CP%train.exe --learning-rate %rate:"=% -e %epochs%
EXIT /B
