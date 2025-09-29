.PHONY: help setup build dev dist test clean start install build-frontend build-electron build-backend-pyinstaller wiki-build wiki-serve lint-duplicates
.ONESHELL:

# Default target
help:
	@echo "Available commands:"
	@echo "  setup      - Install dependencies and set up Python virtual environment"
	@echo "  build      - Build the application (runs tests first)"
	@echo "  dev        - Run in development mode"
	@echo "  dist       - Create distributable package"
	@echo "  test       - Run backend tests only"
	@echo "  clean      - Clean build artifacts"
	@echo "  start      - Start development with process cleanup"
	@echo "  install    - Full clean, rebuild, and install"
	@echo "  lint-duplicates - Check for duplicate code blocks using pylint"
	@echo "  wiki-build - Build the wiki documentation site"
	@echo "  wiki-serve - Serve the wiki documentation site locally"
	@echo "  help       - Show this help message"

# Setup dependencies
setup:
	@echo off
	setlocal enabledelayedexpansion
	for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set start=%%i
	npm install
	cd backend && python -m venv .venv
	cd backend && .venv\Scripts\python.exe -m pip install -r requirements.txt
	cd backend && .venv\Scripts\python.exe -m pip install mkdocs
	for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set end=%%i
	set /a duration=!end! - !start!
	set /a minutes=!duration! / 60
	set /a seconds=!duration! % 60
	echo setup took !minutes! minutes and !seconds! seconds

# Build the application
build: test kill
	@echo off
	setlocal enabledelayedexpansion
	for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set start=%%i
	make build-frontend
	make build-electron
	make build-backend-pyinstaller
	npx copyfiles -u 1 "frontend/components/**/*.html" dist/frontend
	npx copyfiles -u 1 "backend/dist/recall-backend.exe" dist/backend
	for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set end=%%i
	set /a duration=!end! - !start!
	set /a minutes=!duration! / 60
	set /a seconds=!duration! % 60
	echo build took !minutes! minutes and !seconds! seconds

# Run in development mode
dev:
	@echo off
	setlocal enabledelayedexpansion
	for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set start=%%i
	@taskkill /IM recall-backend.exe /F >nul 2>&1 || echo recall-backend.exe not running
	npx concurrently "npm run dev:frontend" "make build-electron && npx electron-forge start"
	for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set end=%%i
	set /a duration=!end! - !start!
	set /a minutes=!duration! / 60
	set /a seconds=!duration! % 60
	echo dev took !minutes! minutes and !seconds! seconds

# Create distributable package
dist: clean build
	@echo off
	setlocal enabledelayedexpansion
	for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set start=%%i
	npx electron-forge make
	for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set end=%%i
	set /a duration=!end! - !start!
	set /a minutes=!duration! / 60
	set /a seconds=!duration! % 60
	echo dist took !minutes! minutes and !seconds! seconds

# Run tests only
test:
	make test-backend
	make test-frontend

# Run backend tests only
test-backend:
ifdef BYPASS_COVERAGE
	cd backend && .\.venv\Scripts\activate.bat && set PYTHONPATH=%cd% && python -m pytest --cache-clear tests/ --tb=short -v
else
	cd backend && .\.venv\Scripts\activate.bat && set PYTHONPATH=%cd% && python -m pytest --cache-clear --tb=short --cov=app --cov-fail-under=100 tests/ -v
endif
# Run frontend tests only
test-frontend:
ifdef BYPASS_COVERAGE
	npm run test:frontend
else
	npm run test:frontend:coverage
endif

# Build frontend
build-frontend:
	npx vite build

# Build electron
build-electron:
	npx tsc -p tsconfig.electron.json

# Build backend with PyInstaller
build-backend-pyinstaller:
	cd backend && python -m venv .venv-build && .\.venv-build\Scripts\python.exe -m ensurepip --default-pip && .\.venv-build\Scripts\pip.exe install pyinstaller && .\.venv-build\Scripts\pip.exe install -r requirements.txt && .\.venv-build\Scripts\python.exe -m PyInstaller --onefile --name recall-backend main.py --hidden-import uvicorn --hidden-import aiosqlite --collect-all uvicorn --add-data "app/static;app/static" --add-data "migrations;migrations"

# Clean build artifacts
clean:
	npx rimraf dist
	npx rimraf backend/dist
	npx rimraf backend/build
	npx rimraf backend/.venv-build

# Start development with cleanup
start: build
	@echo Starting Electron Forge...
	npx electron-forge start

# Start backend only
start-backend:
	@echo Starting backend server...
	cd backend && .\.venv\Scripts\activate.bat && set FLASK_DEBUG=1 && set PYTHONPATH=%cd% && python main.py

# Full clean, rebuild, and install
install: dist
	@echo Removing old app data...
	@npx rimraf "C:\Users\%USERNAME%\AppData\Local\recall"
	@npx rimraf "C:\Users\%USERNAME%\AppData\Roaming\recall"
	@echo Build successful. Launching installer...
	@start "" "out\make\squirrel.windows\x64\recall-1.0.0 Setup.exe"


kill:
	@echo Killing processes...
	-@taskkill /IM recall.exe /F >nul 2>&1
	-@taskkill /IM electron.exe /F >nul 2>&1
	-@taskkill /IM recall-backend.exe /F >nul 2>&1
	-@for %%p in (8000 8001 8002) do ( \
		for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p') do ( \
			taskkill /PID %%a /F >nul 2>&1 \
		) \
	)
	@echo Process cleanup complete

# Check for duplicate code using pylint similarities
lint-duplicates:
	cd backend && .\.venv\Scripts\activate.bat && set PYTHONPATH=%cd% && pylint --rcfile=.pylintrc app/

# Build wiki documentation site
wiki-build:
	@echo "Building wiki documentation..."
	cd backend && .\.venv\Scripts\activate.bat && cd .. && mkdocs build

# Serve wiki documentation site locally
wiki-serve:
	@echo "Serving wiki documentation on http://127.0.0.1:8000"
	@echo "Press Ctrl+C to stop the server"
	cd backend && .\.venv\Scripts\activate.bat && cd .. && mkdocs serve --dev-addr=127.0.0.1:8000
