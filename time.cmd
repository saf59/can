@Echo Off
set RUST_BACKTRACE=1
set CP=D:\projects\rust\can\target\release\
rem cargo build --release --bin detect3
%CP%dur.exe
%CP%detect3 -f 37523.4522 test_data/x1_y1.wav
%CP%dur.exe --stop 
