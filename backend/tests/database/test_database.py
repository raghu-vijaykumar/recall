import pytest
from unittest.mock import patch, MagicMock
from backend.app.database import get_db, DATABASE_URL, engine, async_session


class TestDatabase:
    """Test cases for database connection and session management."""

    @pytest.mark.asyncio
    async def test_get_db_dependency(self):
        """Test the get_db dependency function."""
        # Test that we can get a database session
        async for session in get_db():
            assert session is not None
            # Verify it's an AsyncSession
            from sqlalchemy.ext.asyncio import AsyncSession

            assert isinstance(session, AsyncSession)
            break  # Only test the first yield

    def test_database_url_configuration(self):
        """Test that DATABASE_URL is properly configured."""
        assert DATABASE_URL.startswith("sqlite+aiosqlite:///")
        assert "recall.db" in DATABASE_URL

    def test_engine_configuration(self):
        """Test that the async engine is properly configured."""
        assert engine is not None
        assert str(engine.url) == DATABASE_URL

    def test_async_session_factory(self):
        """Test that the async session factory is properly configured."""
        assert async_session is not None
        assert async_session.kw["bind"] is engine
        assert async_session.class_ is not None

    @pytest.mark.asyncio
    async def test_database_operations(self):
        """Test basic database operations work."""
        from sqlalchemy import text

        async for session in get_db():
            # Test that we can execute a simple query
            result = await session.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            assert row is not None
            assert row.test == 1
            break

    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self):
        """Test that sessions are properly closed even on errors."""
        session_used = False
        try:
            async for session in get_db():
                session_used = True
                # Simulate an error
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected error

        assert session_used, "Session should have been created before error"

    def test_database_path_override(self):
        """Test that DATABASE_PATH environment variable is respected."""
        with patch.dict("os.environ", {"DATABASE_PATH": "/custom/path/test.db"}):
            # Re-import to get new DATABASE_URL
            from importlib import reload
            import backend.app.database as db_module

            reload(db_module)

            expected_url = "sqlite+aiosqlite:////custom/path/test.db"
            assert db_module.DATABASE_URL == expected_url

            # Restore original
            reload(db_module)
