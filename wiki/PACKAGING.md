e# Packaging the Recall Application

This document details the process of packaging the Recall application into distributable formats for various operating systems. The packaging process primarily uses Electron Forge.

## Prerequisites

*   Ensure all build steps for the frontend and backend are completed. Refer to `BUILD.md`.
*   Node.js and npm/yarn installed.
*   Python 3.x and pip installed.

## Electron Forge Configuration

Electron Forge uses `forge.config.js` for its configuration. This file defines the makers (for different package formats) and other packaging options.

## Packaging Steps

1.  **Install Electron Forge dependencies (if not already installed):**
    ```bash
    npm install --save-dev @electron-forge/cli
    npx electron-forge import
    ```

2.  **Run the package command:**
    This command will trigger the Electron Forge to build and package the application according to the `forge.config.js` configuration.
    ```bash
    npm run make
    ```
    This command will:
    *   Compile the Electron main process code.
    *   Bundle the frontend assets (which should already be built).
    *   Create distributable packages (e.g., `.exe` for Windows, `.dmg` for macOS, `.deb`/`.rpm` for Linux) in the `out/` directory.

## Platform-Specific Notes

*   **Windows (`.exe`):** The `make` command will generate an executable installer.
*   **macOS (`.dmg`):** A disk image will be created.
*   **Linux (`.deb`, `.rpm`):** Debian and RPM packages will be generated, suitable for respective Linux distributions.

## Customizing Packaging

To customize the packaging process (e.g., change icons, add splash screens, modify installer options), you will need to edit the `forge.config.js` file. Refer to the Electron Forge documentation for detailed configuration options.