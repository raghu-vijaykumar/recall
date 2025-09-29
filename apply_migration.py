#!/usr/bin/env python3
"""Apply the topics-only architecture migration."""

import sqlite3
import os

# Database path
DB_PATH = "backend/recall.db"


def run_migration():
    """Run the migration to implement topics-only architecture."""

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found. Creating it...")
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        print("Database created.")

    # Read migration file
    migration_file = "backend/migrations/V008__topics_only_architecture.sql"
    with open(migration_file, "r") as f:
        migration_sql = f.read()

    # Execute migration
    print("Applying migration V008__topics_only_architecture...")
    conn = sqlite3.connect(DB_PATH)

    try:
        # Split SQL statements and execute them
        statements = migration_sql.split(";")

        for statement in statements:
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                print(f"Executing: {statement[:50]}...")
                conn.execute(statement)

        conn.commit()
        print("Migration completed successfully!")

    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_migration()
