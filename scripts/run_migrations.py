#!/usr/bin/env python3
"""
Script to run database migrations for the Recall application
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.migration_service import MigrationService
from app.services.database import DatabaseService


def main():
    """Run pending migrations"""
    print("Running database migrations...")

    # Get database path from DatabaseService
    db_service = DatabaseService()
    db_path = db_service.db_path

    print(f"Database path: {db_path}")

    # Initialize migration service
    migration_service = MigrationService(str(db_path))

    # Check current status
    status = migration_service.get_migration_status()
    print(f"Applied migrations: {status['applied_migrations']}")
    print(f"Available migrations: {status['available_migrations']}")
    print(f"Pending migrations: {status['pending_migrations']}")

    # Apply pending migrations
    if status["pending_migrations"]:
        print(f"\nApplying {len(status['pending_migrations'])} pending migrations...")
        success = migration_service.apply_pending_migrations()

        if success:
            print("✅ All migrations applied successfully!")
        else:
            print("❌ Failed to apply some migrations")
            return 1
    else:
        print("✅ No pending migrations")

    return 0


if __name__ == "__main__":
    sys.exit(main())
