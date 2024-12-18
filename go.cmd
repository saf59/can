@echo on 
for /f "tokens=*" %%a in ('D:\projects\rust\can\target\release\detect3.exe -f %2 %1 ') do set wd=%%a
echo %1,%2,%3,%wd% >>reg_2400.csv