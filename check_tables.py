import os
import sqlite3

# Check the same database that migrations use
db_path = os.path.expanduser("~/.recall/recall.db")
print(f"Checking database: {db_path}")
conn = sqlite3.connect(db_path)
tables = [
    r[0] for r in conn.execute('SELECT name FROM sqlite_master WHERE type="table"')
]
print("All Tables:", tables)
print()

concept_tables = [t for t in tables if "concept" in t.lower()]
relationship_tables = [t for t in tables if "relationship" in t.lower()]
topic_tables = [t for t in tables if "topic" in t.lower()]

print("Concept-related tables:", concept_tables)
print("Relationship-related tables:", relationship_tables)
print("Topic-related tables:", topic_tables)

conn.close()
