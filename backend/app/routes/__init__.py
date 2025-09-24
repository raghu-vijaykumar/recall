from .workspaces import router as workspaces_router
from .files import router as files_router
from .quiz import router as quiz_router
from .progress import router as progress_router
from .search import router as search_router
from .knowledge_graph import router as knowledge_graph_router
from .quiz_improvements import router as quiz_improvements_router

# Re-export routers with consistent naming
workspaces = workspaces_router
files = files_router
quiz = quiz_router
progress = progress_router
search = search_router
knowledge_graph = knowledge_graph_router
quiz_improvements = quiz_improvements_router
