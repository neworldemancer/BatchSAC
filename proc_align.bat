@echo off
set workdir=%1
set dsname=%2
set runname=%3

rem set proc_sfx=DC_%year.%month.%date %%hour-%%min-%%sec
set proc_sfx=%4

set algn_col=%5

set bindir=c:\VivoFollow\bin\
set cfg=d:\Trafficking_proc\BatchSAC\cfg\DistCorr_align_%algn_col%.cfg

%bindir%\DistCorr_64.exe -cfg:%cfg%
rem %bindir%\DistCorr_64-Debug.exe -cfg:%bindir%\DistCorr_align.cfg -sleep:10000
