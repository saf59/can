@Echo Off
set RUST_BACKTRACE=1
set CP=D:\projects\rust\can\target\release\
echo  1=%1 2=%2 3=%3 4=%4 5=%5 6=%6 7=%7 8=%8 9=%9
rem  1=c 2=1 3=relu 4=18 5=2000 6=ten 7=none 8=none 9=raw
SET CR=%1
IF NOT DEFINED CR SET "CR=c"
SET BATCH=%2
IF NOT DEFINED BATCH SET "BATCH=1"
SET ACTIVATION=%3
IF NOT DEFINED ACTIVATION SET "ACTIVATION=relu"
SET N=%4
IF NOT DEFINED N SET "N=260"
set epochs=%5
IF NOT DEFINED epochs SET "epochs=1500"
SET alg=%6
IF NOT DEFINED alg SET "alg=bin"
SET buff=%7
IF NOT DEFINED buff SET "buff=none"
SET scaled=%8
IF NOT DEFINED scaled SET "scaled=none"
SET dt=%9
IF NOT DEFINED dt SET "dt=none"
shift
SET norm=%9
IF NOT DEFINED norm SET "norm=none"


IF NOT DEFINED DETECT SET "DETECT=detect3"
IF NOT DEFINED hidden SET hidden="100,40,10"
SET hidden=%hidden:"=%

set NAME=%CR%_%BATCH%_%ACTIVATION%_%N%_%alg%_%buff%_%scaled%_%hidden:,=_%
ECHO Start %NAME%

echo %CR% %BATCH% %ACTIVATION% %N% %alg% %buff% %scaled% %hidden:,=_%> %NAME%.csv 

rem BUILD
%CP%dur.exe
set rate="0.005"
rem set default meta by -e 0
if %CR% == c (
	echo Train classification %NAME%
	%CP%train.exe --model-type classification --batch-size %BATCH% --train-part 1.0 -e 0 --activation %ACTIVATION% --hidden %hidden% --alg-type %alg% --buff-size %buff% --norm %norm% --scale %scaled% --data-type %dt% -n %N%
	call :train
) else if %CR% == r (
	echo Train regression %NAME%
	%CP%train.exe --model-type regression --batch-size %BATCH% --train-part 1.0 -e 0 --activation %ACTIVATION% --hidden %hidden% --alg-type %alg% --buff-size %buff% --norm %norm% --scale %scaled% --data-type %dt% -n %N%
	call :train
) else (
    ECHO Bad C/R type "%CR%" , exit 
	%CP%dur.exe --stop
    EXIT /B
)
%CP%dur.exe --stop >> %NAME%.csv

rem meta MUST be embed to %DETECT%
cargo build --release --bin %DETECT% --bin test3
rem CHECK
call %DETECT%
ECHO %NAME%

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
