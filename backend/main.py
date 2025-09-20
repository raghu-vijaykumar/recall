import uvicorn
import os
import sys  # Import sys to check for frozen environment
import logging
from pathlib import Path
from app.app import app  # Import the FastAPI app instance


# Configure logging
def configure_logging():
    log_dir = None
    database_path = os.getenv("DATABASE_PATH")
    if database_path:
        log_dir = Path(database_path).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "backend.log" if log_dir else "backend.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[{asctime}] [{levelname}] {name}: {message}",
        style="{",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),  # Also log to console
        ],
    )
    # Suppress uvicorn access logs to avoid duplication with stdout
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False


if __name__ == "__main__":
    configure_logging()
    logger = logging.getLogger("backend")
    logger.info("Starting backend application...")

    port = int(os.getenv("PORT", 8000))
    # Disable reload when running in a PyInstaller bundle
    is_frozen = getattr(sys, "frozen", False)
    uvicorn.run(
        "app.app:app",
        host="127.0.0.1",
        port=port,
        reload=not is_frozen,
        log_level="info",  # Uvicorn's log_level will still control its own messages
    )
