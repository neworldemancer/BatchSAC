@echo off

set exe=d:\development\LTS\bin\DistCorr_64.exe 
rem set exe=d:\development\LTS\bin\DistCorr_64-Debug.exe
set cfg=%~dp0align_2tf_bgrf.cfg

%exe% -cfg:%cfg%
