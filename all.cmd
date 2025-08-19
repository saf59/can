@Echo Off
cargo build --release --bin train --bin test3
@REM N epochs type size freq|norm
@rem   4     5  6   7    8      9  10(need shift -> 9)
@rem   N epochs alg buff scaled dt norm   
@rem 4
@remSET "DEF=34 200 hom none none raw true"
rem SET "DEF=18 4000 ten none none raw false"
SET "DEF=52 800 a-ten none none raw false"
rem SET "DEF=18 200 ten none none impulse false"

@rem 3
@rem SET "DEF=260 1500 bin small true none false"

@rem SET hidden="100,40,10"
@rem SET hidden="40,20,10,5"
@rem SET hidden="30,30,30"
@rem SET hidden="36,36,36,12"
@rem SET hidden="60,30,15"
@rem SET hidden="60,60,60,15"
SET hidden="104,104,104,20"
@rem SET hidden="48,48,48,12"
@rem SET hidden="60,60,60,60,15"
@rem SET hidden="36,18,9"
SET "DETECT=detect4"
REM set "JOINED=-- -j"
 
rem del total.csv >nul
call :size c
rem call :size r
GOTO :eof
	
:size 
@REM batchsize
call :act %1 1
rem call :act %1 40
rem call :act %1 100
EXIT /B

:act
@REM alg
call doit %1 %2 relu6 %DEF%
EXIT /B
call doit %1 %2 hard-sigmoid %DEF%
call doit %1 %2 relu %DEF%
call doit %1 %2 gelu %DEF%
call doit %1 %2 sigmoid %DEF%
call doit %1 %2 new-gelu %DEF%
call doit %1 %2 relu2 %DEF%
call doit %1 %2 silu %DEF%
call doit %1 %2 hard-sigmoid %DEF%
call doit %1 %2 swish %DEF%
rem call doit %1 %2 hard-swish %DEF%
EXIT /B

:act1
call doit %1 %2 hard-swish %DEF%
EXIT /B

:eof