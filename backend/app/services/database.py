"""
Database service for SQLite operations with migration support
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager, asynccontextmanager
import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from .migration_service import MigrationService

logger = logging.getLogger(__name__)


class DatabaseService:

    def __init__(self, db_path: str = None):
        # Use environment variable if provided, otherwise default to user data
        if db_path is None:
            db_path = os.getenv(
                "DATABASE_PATH",
                os.path.join(os.path.expanduser("~"), ".recall", "recall.db"),
            )
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize async SQLAlchemy components
        database_url = f"sqlite+aiosqlite:///{self.db_path}"

        # Create async engine
        self.engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL query logging
            future=True,
        )

        # Create async session factory
        self.async_session = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self._init_db()

    def _init_db(self):
        """Initialize database with migrations"""
        # Initialize migration service
        migrations_dir = Path(__file__).parent.parent.parent / "migrations"
        self.migration_service = MigrationService(
            str(self.db_path), str(migrations_dir)
        )

        # Apply any pending migrations
        if self.migration_service.apply_pending_migrations():
            logger.info("Database migrations applied successfully")
        else:
            logger.error("Failed to apply database migrations")

        # For backward compatibility, also check for schema.sql if no migrations exist
        if not self.migration_service.get_migration_status()["applied_migrations"]:
            schema_path = (
                Path(__file__).parent.parent.parent.parent / "database" / "schema.sql"
            )
            if schema_path.exists():
                logger.info("No migrations found, falling back to schema.sql")
                with open(schema_path, "r") as f:
                    schema = f.read()

                with self.get_connection() as conn:
                    conn.executescript(schema)
                    logger.info("Database schema initialized from schema.sql")
            else:
                logger.warning(
                    f"Schema file not found at {schema_path}, skipping initialization"
                )

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as dicts"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT query and return the last row ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute multiple INSERT/UPDATE queries"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount

    # Utility methods for JSON handling
    def _json_dumps(self, obj: Any) -> str:
        """Convert object to JSON string"""
        return json.dumps(obj) if obj is not None else None

    def _json_loads(self, json_str: str) -> Any:
        """Convert JSON string to object"""
        return json.loads(json_str) if json_str else None

    # Convenience methods for common operations
    def get_by_id(self, table: str, id: int) -> Optional[Dict[str, Any]]:
        """Get a single record by ID"""
        query = f"SELECT * FROM {table} WHERE id = ?"
        results = self.execute_query(query, (id,))
        return results[0] if results else None

    def get_all(
        self, table: str, limit: int = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all records from a table"""
        query = f"SELECT * FROM {table}"
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        return self.execute_query(query)

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert a new record"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = tuple(data.values())

        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return self.execute_insert(query, values)

    def update(self, table: str, id: int, data: Dict[str, Any]) -> int:
        """Update a record by ID"""
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        values = tuple(data.values()) + (id,)

        query = f"UPDATE {table} SET {set_clause} WHERE id = ?"
        return self.execute_update(query, values)

    def delete(self, table: str, id: int) -> int:
        """Delete a record by ID"""
        query = f"DELETE FROM {table} WHERE id = ?"
        return self.execute_update(query, (id,))

    def count(self, table: str, where_clause: str = "", params: tuple = ()) -> int:
        """Count records in a table"""
        query = f"SELECT COUNT(*) as count FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        result = self.execute_query(query, params)
        return result[0]["count"] if result else 0

    # Migration-related methods
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        return self.migration_service.get_migration_status()

    def create_migration(self, migration_id: str, description: str = "") -> Path:
        """Create a new migration file"""
        return self.migration_service.create_migration_file(migration_id, description)

    def apply_pending_migrations(self) -> bool:
        """Apply any pending migrations"""
        return self.migration_service.apply_pending_migrations()

    def rollback_migration(self, migration_id: str) -> bool:
        """Rollback a specific migration"""
        return self.migration_service.rollback_migration(migration_id)


# Global database service instance for dependency injection
# This will be initialized in app startup
_db_service = None


def init_database_service(db_path: str = None):
    """Initialize the global database service instance"""
    global _db_service
    _db_service = DatabaseService(db_path)
    # Update module-level exports for backward compatibility
    _update_exports(_db_service)
    return _db_service


async def get_db() -> AsyncSession:
    """
    Dependency function to get database session for FastAPI routes
    """
    global _db_service
    if _db_service is None:
        raise RuntimeError(
            "Database service not initialized. Call init_database_service() first."
        )

    async with _db_service.async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# Export database components for backward compatibility (used by tests)
DATABASE_URL = None
engine = None
async_session = None


def _update_exports(db_service):
    """Update module-level exports to point to the singleton instance"""
    global DATABASE_URL, engine, async_session
    DATABASE_URL = f"sqlite+aiosqlite:///{db_service.db_path}"
    engine = db_service.engine
    async_session = db_service.async_session


# Export the database service class and functions
__all__ = [
    "DatabaseService",
    "init_database_service",
    "get_db",
    "engine",
    "async_session",
    "DATABASE_URL",
]
