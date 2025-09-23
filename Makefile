.PHONY: help setup build dev dist test clean start install build-frontend build-electron build-backend-pyinstaller

# Default target
help:
	@echo "Available commands:"
	@echo "  setup    - Install dependencies and set up Python virtual environment"
	@echo "  build    - Build the application (runs tests first)"
	@echo "  dev      - Run in development mode"
	@echo "  dist     - Create distributable package"
	@echo "  test     - Run backend tests only"
	@echo "  clean    - Clean build artifacts"
	@echo "  start    - Start development with process cleanup"
	@echo "  install  - Full clean, rebuild, and install"
	@echo "  help     - Show this help message"

# Setup dependencies
setup:
	npm install
	cd backend && python -m venv .venv
	cd backend && .venv\Scripts\python.exe -m pip install -r requirements.txt

# Build the application
build: test
	make build-frontend
	make build-electron
	make build-backend-pyinstaller
	npx copyfiles -u 1 "frontend/components/**/*.html" dist/frontend
	npx copyfiles -u 1 "backend/dist/recall-backend.exe" dist/backend

# Run in development mode
dev:
	@taskkill /IM recall-backend.exe /F >nul 2>&1 || echo recall-backend.exe not running
	npx concurrently "npm run dev:frontend" "make build-electron && npx electron-forge start"

# Create distributable package
dist: clean build
	npx electron-forge make

# Run tests only
test:
	cd backend && .\.venv\Scripts\activate.bat && set PYTHONPATH=%cd% && python -m pytest tests/ -v

# Build frontend
build-frontend:
	npx vite build

# Build electron
build-electron:
	npx tsc -p tsconfig.electron.json

# Build backend with PyInstaller
build-backend-pyinstaller:
	cd backend && python -m venv .venv-build && .\.venv-build\Scripts\python.exe -m ensurepip --default-pip && .\.venv-build\Scripts\pip.exe install pyinstaller && .\.venv-build\Scripts\pip.exe install -r requirements.txt && .\.venv-build\Scripts\python.exe -m PyInstaller --onefile --name recall-backend main.py --hidden-import uvicorn --collect-all uvicorn --add-data "app/static;app/static" --add-data "migrations;migrations"

# Clean build artifacts
clean:
	npx rimraf dist
	npx rimraf backend/dist
	npx rimraf backend/build
	npx rimraf backend/.venv-build

# Start development with cleanup
start: build
	@echo Killing processes on ports 8000, 8001, 8002...
	-@for %%p in (8000 8001 8002) do ( \
		for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p') do ( \
			@echo Attempting to kill process %%a on port %%p \
			taskkill /PID %%a /F 2>nul || echo Failed to kill process %%a \
		) \
	)
	@echo Killing recall-backend.exe if running...
	-@taskkill /IM recall-backend.exe /F >nul 2>&1 || echo recall-backend.exe not running
	@echo Starting Electron Forge...
	npx electron-forge start

# Full clean, rebuild, and install
install: dist
	@echo Killing processes...
	@taskkill /IM recall.exe /F >nul 2>&1 || echo recall.exe not running
	@taskkill /IM electron.exe /F >nul 2>&1 || echo electron.exe not running
	@taskkill /IM recall-backend.exe /F >nul 2>&1 || echo recall-backend.exe not running
	@for %%p in (8000 8001 8002) do ( \
		for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p') do ( \
			taskkill /PID %%a /F >nul 2>&1 \
			@echo Killed process %%a on port %%p \
		) \
	)
	@echo Removing old app data...
	@npx rimraf "C:\Users\%USERNAME%\AppData\Local\recall"
	@npx rimraf "C:\Users\%USERNAME%\AppData\Roaming\recall"
	@echo Build successful. Launching installer...
	@start "" "out\make\squirrel.windows\x64\recall-1.0.0 Setup.exe"
