import sqlite3

conn = sqlite3.connect("backend/recall.db")
tables = [
    r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
]
print("Tables:", tables)

# Check if V008 applied
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM schema_migrations WHERE migration_id='V008'")
print("V008 applied:", cursor.fetchone()[0] > 0)

conn.close()
