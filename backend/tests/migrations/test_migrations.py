"""
Test migration functionality using pytest fixtures
"""

import os
from pathlib import Path
from app.services.database import DatabaseService


def test_migrations():
    """Test the migration system using isolated test database"""
    # DatabaseService will use the DATABASE_PATH environment variable
    # set by the conftest.py fixture, so no need to specify path here
    db_service = DatabaseService()

    # Check migration status
    status = db_service.get_migration_status()
    assert "applied_migrations" in status
    assert "pending_migrations" in status

    # Test creating a new migration
    migration_file = db_service.create_migration(
        "V999__test_migration", "Test migration for verification"
    )
    assert migration_file.exists()

    # Clean up the test migration file
    if migration_file.exists():
        migration_file.unlink()

    # Test applying migrations (should succeed since all are applied)
    success = db_service.apply_pending_migrations()
    assert success is True

    # Check final status
    final_status = db_service.get_migration_status()
    assert len(final_status["applied_migrations"]) >= 1  # At least initial schema

    # Test database operations work
    workspace_id = db_service.insert(
        "workspaces",
        {
            "name": "Test Workspace",
            "description": "Test workspace for migration verification",
            "type": "study",
            "color": "#ff0000",
        },
    )
    assert workspace_id > 0

    # Query the workspace
    workspace = db_service.get_by_id("workspaces", workspace_id)
    assert workspace is not None
    assert workspace["name"] == "Test Workspace"
    assert workspace["description"] == "Test workspace for migration verification"
