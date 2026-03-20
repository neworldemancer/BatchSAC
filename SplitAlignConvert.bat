@echo off
set spath=%~dp0
pushd %spath%
uv run python %spath%Split_Align_convert_GUI.py
popd
