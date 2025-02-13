@Echo Off
del total.csv >nul
call :size c
call :size r
GOTO :eof
	
:size 
call :act %1 1
call :act %1 40
EXIT /B

:act
call doit %1 %2 gelu
call doit %1 %2 new-gelu
call doit %1 %2 relu
call doit %1 %2 relu2
call doit %1 %2 relu6
call doit %1 %2 silu
call doit %1 %2 sigmoid
call doit %1 %2 hard-sigmoid
call doit %1 %2 swish
call doit %1 %2 hard-swish
EXIT /B

:eof