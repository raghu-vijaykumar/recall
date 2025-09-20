@echo off
echo Killing processes on ports 8000, 8001, 8002...

for %%p in (8000 8001 8002) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p') do (
        echo Attempting to kill process %%a on port %%p
        taskkill /PID %%a /F
    )
)

echo Starting Electron Forge...
electron-forge start