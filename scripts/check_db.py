import sqlite3

conn = sqlite3.connect("backend/database/recall.db")
cursor = conn.cursor()

# Check migrations
cursor.execute("SELECT * FROM schema_migrations")
print("Applied migrations:")
for row in cursor.fetchall():
    print(f"Migration: {row[1]}, Applied: {row[3]}")

# First check table schema
cursor.execute("PRAGMA table_info(workspaces)")
print("\nWorkspaces table schema:")
for row in cursor.fetchall():
    print(f"Column: {row[1]}, Type: {row[2]}")

print("\nWorkspaces data:")
cursor.execute("SELECT * FROM workspaces")
for row in cursor.fetchall():
    print(f"Row: {row}")
conn.close()
