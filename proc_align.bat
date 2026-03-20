@echo off
set cfg=%1
rem set bindir=c:\VivoFollow\bin\
set bindir=d:\development\LTS\bin\
%bindir%\DistCorr_64.exe -cfg:%cfg%
rem %bindir%\DistCorr_64-Debug.exe -cfg:%cfg% -sleep:10000
