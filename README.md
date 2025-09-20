# Recall

Recall is an application designed to help users learn and retain information from their files by generating quizzes and tracking progress.

## Features

*   **Workspace Management:** Organize your files into workspaces for focused learning.
*   **File Analysis:** Analyze various file types to extract key information.
*   **Quiz Generation:** Automatically generate quizzes based on the content of your files.
*   **Progress Tracking:** Monitor your learning progress and identify areas for improvement.

## Getting Started

### Prerequisites

*   Node.js (for frontend and Electron)
*   Python 3.x (for backend)
*   npm or yarn (for frontend dependencies)
*   pip (for backend dependencies)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/recall.git
    cd recall
    ```

2.  **Backend Setup:**
    ```bash
    cd backend
    pip install -r requirements.txt
    python main.py
    ```

3.  **Frontend Setup:**
    ```bash
    cd ..
    npm install
    npm run dev
    ```

## Project Structure

*   [`backend/`](backend/): Contains the Python Flask backend application.
*   [`electron/`](electron/): Contains the Electron-specific code for the desktop application.
*   [`frontend/`](frontend/): Contains the React/TypeScript frontend application.
*   [`database/`](database/): Contains database schema and migration scripts.
*   [`wiki/`](wiki/): Contains documentation on building and packaging.
*   [`TODO.md`](TODO.md): A list of improvements and future tasks.

## Contributing

We welcome contributions! Please see our `TODO.md` for areas of improvement and feel free to submit pull requests.

## License

This project is licensed under the MIT License.