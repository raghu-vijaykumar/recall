import sqlite3
import os
from pathlib import Path

# Database path configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", str(Path.home() / ".recall" / "recall.db"))

print(f"Database path: {DATABASE_PATH}")

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

# Check if topic-related tables exist
topic_tables = ["topic_areas", "learning_paths", "learning_recommendations"]
for table in topic_tables:
    if table in [t[0] for t in tables]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} records")
    else:
        print(f"{table}: TABLE NOT FOUND")

# Check for concepts and workspaces
cursor.execute("SELECT COUNT(*) FROM concepts")
concept_count = cursor.fetchone()[0]
print(f"concepts: {concept_count} records")

cursor.execute("SELECT COUNT(*) FROM workspaces")
workspace_count = cursor.fetchone()[0]
print(f"workspaces: {workspace_count} records")

if workspace_count > 0:
    cursor.execute("SELECT id, name FROM workspaces")
    workspaces = cursor.fetchall()
    print("Workspaces:")
    for ws in workspaces:
        print(f"  ID: {ws[0]}, Name: {ws[1]}")

conn.close()
