@echo off
rem Launch SkyRoads in the portable DOSBox Staging next to this file.
cd /d "%~dp0"
if not exist "skyroads\" (
  echo Put the SkyRoads game files in "%~dp0skyroads\" first.
  pause
  exit /b 1
)
start "" "dosbox\dosbox.exe" -conf "skyroads.conf"
