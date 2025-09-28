# Backend README

This document provides instructions for setting up, building, and packaging the backend of the Recall project.

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/raghu-vijaykumar/recall.git
   cd recall/backend
   ```
2. **Create a virtual environment (recommended):**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Backend
- To start the backend server (replace `app.py` with your main file if different):
  ```sh
  python app.py
  ```

## Build & Packaging
1. **Freeze dependencies:**
   ```sh
   pip freeze > requirements.txt
   ```
2. **Create a distributable package:**
   - If using setuptools, ensure you have a `setup.py` file. Then run:
     ```sh
     python setup.py sdist bdist_wheel
     ```
   - The built packages will be in the `dist/` directory.

3. **Distribute or deploy:**
   - Share the `requirements.txt` and your source code, or
   - Upload the package to PyPI (if public) or your private repository.

## Notes
- Update this README with any additional environment variables, configuration, or special instructions as your backend evolves.
- For troubleshooting, consult the error messages or open an issue in the repository.


DELETE FROM concepts;
DELETE FROM concept_files;
DELETE FROM relationships;
DELETE FROM files;
COMMIT;