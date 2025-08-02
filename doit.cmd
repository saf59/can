@Echo Off
set RUST_BACKTRACE=1
set CP=D:\projects\rust\can\target\release\

SET N=%4
IF NOT DEFINED N SET "N=260"
set epochs=%5
IF NOT DEFINED epochs SET "epochs=1500"
SET alg=%6
IF NOT DEFINED alg SET "alg=bin"
SET buff=%7
IF NOT DEFINED buff SET "buff=small"
SET scaled=%8
IF NOT DEFINED scaled SET "scaled=true"
SET scaled=%9
IF NOT DEFINED dt SET "dt=none"
shift
SET norm=%9
IF NOT DEFINED norm SET "norm=none"


IF NOT DEFINED DETECT SET "DETECT=detect3"
IF NOT DEFINED hidden SET hidden="100,40,10"
SET hidden=%hidden:"=%

set NAME=%1_%2_%3_%N%_%alg%_%buff%_%scaled%_%hidden:,=_%
ECHO Start %NAME%

echo %1 %2 %3 %N% %alg% %buff% %scaled% %hidden:,=_%> %NAME%.csv 

rem BUILD
%CP%dur.exe
set rate="0.005"
rem set default meta by -e 0
if %1 == c (
	echo Train classification %NAME%
	%CP%train.exe --model-type classification --batch-size %2 --train-part 1.0 -e 0 --activation %3 --hidden %hidden% --alg-type %alg% --buff-size %buff% --norm %norm% --scale %scaled% --data-type %dt%-n %N%
	call :train
) else if %1 == r (
	echo Train regression %NAME%
	%CP%train.exe --model-type regression --batch-size %2 --train-part 1.0 -e 0 --activation %3 --hidden %hidden% --alg-type %alg% --buff-size %buff% --norm %norm% --scale %scaled% --data-type %dt%-n %N%
	call :train
) else (
    ECHO Bad type "%1" , exit 
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
