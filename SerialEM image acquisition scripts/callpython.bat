@echo off

set root=C:\ProgramData\Anaconda3
set environment=DarEM

call %root%\Scripts\activate.bat %environment%

call python %1 %2 %3 %4


echo Execution ended, will close this prompt in 10 seconds...

timeout /T 20

exit
