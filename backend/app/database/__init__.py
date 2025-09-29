# Database package
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

# Database path configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", str(Path.home() / ".recall" / "recall.db"))

# Convert SQLite path to async URL
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    future=True,
)

# Create async session factory
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """
    Dependency function to get database session for FastAPI routes
    """
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
