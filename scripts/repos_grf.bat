@echo off

set exe=d:\development\LTS\bin\DistCorr_64.exe 
rem set exe=d:\development\LTS\bin\DistCorr_64-Debug.exe
rem set cfg=%~dp0repos_seq_bgr.cfg
set cfg=%~dp0repos_seq_grf.cfg

%exe% -cfg:%cfg%
