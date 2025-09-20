#!/usr/bin/env python3
"""
Test script for migration functionality
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.database import DatabaseService


def test_migrations():
    """Test the migration system"""
    print("Testing migration system...")

    # Use a test database
    test_db_path = "test_recall.db"

    try:
        # Initialize database service
        db_service = DatabaseService(test_db_path)

        # Check migration status
        status = db_service.get_migration_status()
        print(f"Migration Status: {status}")

        # Test creating a new migration
        print("\nCreating a test migration...")
        migration_file = db_service.create_migration(
            "V002__test_migration", "Test migration for verification"
        )
        print(f"Created migration file: {migration_file}")

        # Check if the file was created
        if migration_file.exists():
            print("✓ Migration file created successfully")
        else:
            print("✗ Migration file creation failed")

        # Test applying migrations
        print("\nApplying pending migrations...")
        success = db_service.apply_pending_migrations()
        if success:
            print("✓ Migrations applied successfully")
        else:
            print("✗ Migration application failed")

        # Check final status
        final_status = db_service.get_migration_status()
        print(f"Final Migration Status: {final_status}")

        # Test database operations
        print("\nTesting database operations...")

        # Insert a test workspace
        workspace_id = db_service.insert(
            "workspaces",
            {
                "name": "Test Workspace",
                "description": "Test workspace for migration verification",
                "type": "study",
                "color": "#ff0000",
            },
        )
        print(f"✓ Created test workspace with ID: {workspace_id}")

        # Query the workspace
        workspace = db_service.get_by_id("workspaces", workspace_id)
        if workspace:
            print(f"✓ Retrieved workspace: {workspace['name']}")
        else:
            print("✗ Failed to retrieve workspace")

        print("\n✓ All tests completed successfully!")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

    finally:
        # Clean up test database
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
                print(f"Cleaned up test database: {test_db_path}")
            except PermissionError:
                print(f"Could not clean up test database (in use): {test_db_path}")

    return True


if __name__ == "__main__":
    success = test_migrations()
    sys.exit(0 if success else 1)
