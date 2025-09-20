from .workspaces import router as workspaces_router
from .files import router as files_router
from .quiz import router as quiz_router
from .progress import router as progress_router

# Re-export routers with consistent naming
workspaces = workspaces_router
files = files_router
quiz = quiz_router
progress = progress_router
