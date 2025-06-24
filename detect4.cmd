%CP%test3 >> %NAME%.csv 
%CP%dur.exe
%CP%%DETECT% %JOINED% test_data/4/in.wav >>%NAME%.csv
%CP%%DETECT% %JOINED% test_data/4/below.wav >> %NAME%.csv
%CP%%DETECT% %JOINED% test_data/4/above.wav >> %NAME%.csv
%CP%dur.exe --stop >> %NAME%.csv
