@echo off
echo === Cleaning up Recall build ===

:: Kill recall.exe if running
taskkill /IM recall.exe /F >nul 2>&1
if %ERRORLEVEL% NEQ 0 echo recall.exe not running

:: Kill electron.exe if running
taskkill /IM electron.exe /F >nul 2>&1
if %ERRORLEVEL% NEQ 0 echo electron.exe not running

:: Reset errorlevel
ver >nul

:: Clean old build artifacts
echo Removing old build files...
call npx rimraf C:\Users\raghu\AppData\Local\recall
call npx rimraf C:\Users\raghu\AppData\Roaming\recall

:: Rebuild and make
echo Starting rebuild...
call npm run clean-rebuild-make

:: If build succeeded, run the installer
if %ERRORLEVEL% EQU 0 (
    echo Build successful. Launching installer...
    start "" "out\make\squirrel.windows\x64\recall-1.0.0 Setup.exe"
) else (
    echo Build failed. Not launching installer.
)
