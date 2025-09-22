"""
Migration service for handling database schema changes
Similar to Liquibase but implemented in Python for SQLite
"""

import os
import sqlite3
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MigrationService:
    def __init__(self, db_path: str, migrations_dir: str = None):
        self.db_path = Path(db_path)
        if migrations_dir:
            self.migrations_dir = Path(migrations_dir)
        elif getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            # Running in a PyInstaller bundle
            self.migrations_dir = Path(sys._MEIPASS) / "migrations"
        else:
            # Running in development
            self.migrations_dir = Path(__file__).parent.parent.parent / "migrations"

        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Migrations directory set to: {self.migrations_dir}")
        self._init_migration_table()

    def _init_migration_table(self):
        """Initialize the migrations tracking table"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_id TEXT NOT NULL UNIQUE,
                    description TEXT,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    checksum TEXT
                )
            """
            )
            conn.commit()

    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration IDs"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT migration_id FROM schema_migrations ORDER BY id")
            return [row[0] for row in cursor.fetchall()]

    def apply_migration(self, migration_file: Path) -> bool:
        """Apply a single migration file"""
        migration_id = migration_file.stem

        # Check if already applied
        applied = self.get_applied_migrations()
        if migration_id in applied:
            logger.info(f"Migration {migration_id} already applied, skipping")
            return True

        try:
            # Read migration content
            with open(migration_file, "r", encoding="utf-8") as f:
                sql_content = f.read()

            # Calculate checksum (simple hash)
            checksum = str(hash(sql_content))

            # Apply migration
            with sqlite3.connect(str(self.db_path)) as conn:
                # Execute the migration SQL
                conn.executescript(sql_content)

                # Record the migration
                conn.execute(
                    """
                    INSERT INTO schema_migrations (migration_id, description, checksum)
                    VALUES (?, ?, ?)
                """,
                    (migration_id, migration_file.name, checksum),
                )

                conn.commit()

            logger.info(f"Successfully applied migration: {migration_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply migration {migration_id}: {e}")
            return False

    def apply_pending_migrations(self) -> bool:
        """Apply all pending migrations in order"""
        migration_files = self._get_migration_files()

        if not migration_files:
            logger.info(f"No migration files found in {self.migrations_dir}")
            return True
        logger.debug(f"Found migration files: {[f.name for f in migration_files]}")

        applied = self.get_applied_migrations()
        pending = [f for f in migration_files if f.stem not in applied]

        if not pending:
            logger.info("No pending migrations")
            return True

        # Sort by filename (assuming V001__, V002__, etc.)
        pending.sort(key=lambda x: x.name)

        success = True
        for migration_file in pending:
            if not self.apply_migration(migration_file):
                success = False
                break

        return success

    def _get_migration_files(self) -> List[Path]:
        """Get all migration files from migrations directory"""
        if not self.migrations_dir.exists():
            return []

        migration_files = []
        for file in self.migrations_dir.iterdir():
            if file.is_file() and file.suffix.lower() == ".sql":
                migration_files.append(file)

        return migration_files

    def create_migration_file(self, migration_id: str, description: str = "") -> Path:
        """Create a new migration file template"""
        filename = f"{migration_id}.sql"
        filepath = self.migrations_dir / filename

        template = f"""-- Migration: {migration_id}
-- Description: {description}
-- Created: {datetime.now().isoformat()}

-- Add your SQL migration statements here
-- Example:
-- ALTER TABLE users ADD COLUMN email TEXT;
-- CREATE INDEX idx_users_email ON users(email);

"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(template)

        logger.info(f"Created migration file: {filepath}")
        return filepath

    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        applied = self.get_applied_migrations()
        available = [f.stem for f in self._get_migration_files()]

        return {
            "applied_migrations": applied,
            "available_migrations": available,
            "pending_migrations": [m for m in available if m not in applied],
        }

    def rollback_migration(self, migration_id: str) -> bool:
        """Rollback a specific migration (if rollback script exists)"""
        rollback_file = self.migrations_dir / f"{migration_id}_rollback.sql"

        if not rollback_file.exists():
            logger.error(f"Rollback file not found: {rollback_file}")
            return False

        try:
            with open(rollback_file, "r", encoding="utf-8") as f:
                sql_content = f.read()

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.executescript(sql_content)
                conn.execute(
                    "DELETE FROM schema_migrations WHERE migration_id = ?",
                    (migration_id,),
                )
                conn.commit()

            logger.info(f"Successfully rolled back migration: {migration_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback migration {migration_id}: {e}")
            return False
