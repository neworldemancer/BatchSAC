@echo off

set exe=d:\development\LTS\bin\DistCorr_64.exe 
rem set exe=d:\development\LTS\bin\DistCorr_64-Debug.exe
rem set cfg=%~dp0align_2tf_bgr.cfg
set cfg=%~dp0align_2tf_grf.cfg

%exe% -cfg:%cfg%
