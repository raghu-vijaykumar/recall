# Build Process

This document outlines the steps to build the Recall application. The application consists of a Python Flask backend, a React/TypeScript frontend, and an Electron wrapper for the desktop application.

## 1. Backend Build

The backend is a Python Flask application. It doesn't require a separate "build" step in the traditional sense, but rather relies on dependency installation.

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Install Python dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    ./.venv/Scripts/activate  # On Windows
    source ./.venv/bin/activate # On macOS/Linux
    pip install -r requirements.txt
    ```

3.  **Run the backend (for development):**
    ```bash
    python main.py
    ```

## 2. Frontend Build

The frontend is a React/TypeScript application.

1.  **Navigate to the project root directory:**
    ```bash
    cd ..
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    # or yarn install
    ```

3.  **Build the frontend for production:**
    ```bash
    npm run build
    # This will typically output to a `dist` or `build` directory within `frontend/`
    ```

## 3. Electron Build

The Electron application wraps the frontend and communicates with the backend.

1.  **Ensure frontend is built:**
    The Electron application will serve the built frontend assets. Make sure you have run `npm run build` in the frontend.

2.  **Run Electron in development mode:**
    ```bash
    npm run electron:dev
    ```

3.  **Build Electron for production (packaging):**
    This step is usually combined with packaging. Refer to `PACKAGING.md` for details on creating distributable packages.
    ```bash
    npm run electron:build