#!/usr/bin/env python3
"""
Script to check if the new workspace topic discovery tables were created
"""

import sqlite3
from pathlib import Path


def main():
    """Check new tables"""
    db_path = Path.home() / ".recall" / "recall.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return 1

    print(f"Checking database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check for new tables
    tables_to_check = [
        "topic_areas",
        "topic_concept_links",
        "learning_paths",
        "learning_recommendations",
    ]

    print("\nChecking for new tables:")
    for table in tables_to_check:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"✅ {table}: {count} rows")
        except sqlite3.OperationalError:
            print(f"❌ {table}: Table not found")

    # Check schema_migrations table
    try:
        cursor.execute(
            "SELECT migration_id, applied_at FROM schema_migrations ORDER BY id DESC LIMIT 5"
        )
        print("\nRecent migrations:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")
    except sqlite3.OperationalError:
        print("❌ schema_migrations table not found")

    conn.close()
    return 0


if __name__ == "__main__":
    exit(main())
