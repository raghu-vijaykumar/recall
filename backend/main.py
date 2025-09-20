import uvicorn
import os
from app.app import app  # Import the FastAPI app instance

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.app:app", host="127.0.0.1", port=port, reload=True, log_level="info"
    )
