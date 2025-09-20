# Database Migrations

This directory contains database migration files for the Recall Study App. The migration system is similar to Liquibase but implemented in Python for SQLite.

## Overview

The migration system provides:
- Version-controlled database schema changes
- Automatic migration application on startup
- Migration tracking and rollback capabilities
- SQL-based migration files

## Migration Files

Migration files are SQL files with the naming convention: `V{version}__{description}.sql`

Examples:
- `V001__initial_schema.sql` - Initial database schema
- `V002__add_user_table.sql` - Add user management tables
- `V003__update_indexes.sql` - Performance improvements

## File Structure

```
backend/migrations/
├── README.md
├── V001__initial_schema.sql
└── V002__test_migration.sql (example)
```

## Usage

### Creating a New Migration

```python
from app.services.database import DatabaseService

db = DatabaseService()
migration_file = db.create_migration("V003__add_new_feature", "Add new feature tables")
```

### Checking Migration Status

```python
status = db.get_migration_status()
print(status)
# Output: {
#   'applied_migrations': ['V001__initial_schema'],
#   'available_migrations': ['V001__initial_schema', 'V002__add_feature'],
#   'pending_migrations': ['V002__add_feature']
# }
```

### Manual Migration Application

```python
success = db.apply_pending_migrations()
if success:
    print("Migrations applied successfully")
```

### Rollback (if rollback script exists)

```python
success = db.rollback_migration("V002__add_feature")
```

## Migration File Format

Each migration file should contain valid SQL statements:

```sql
-- Migration: V003__add_new_feature
-- Description: Add new feature tables
-- Created: 2025-09-20

-- Add your SQL migration statements here
ALTER TABLE users ADD COLUMN email TEXT UNIQUE;
CREATE INDEX idx_users_email ON users(email);

-- Create new table
CREATE TABLE user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    preference_key TEXT NOT NULL,
    preference_value TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

## Rollback Support

For migrations that need rollback support, create a corresponding rollback file:

- Migration: `V003__add_new_feature.sql`
- Rollback: `V003__add_new_feature_rollback.sql`

Example rollback file:

```sql
-- Rollback for V003__add_new_feature
-- Remove added column and table

DROP TABLE IF EXISTS user_preferences;
CREATE TABLE users_backup AS SELECT id, name, created_at FROM users;
DROP TABLE users;
ALTER TABLE users_backup RENAME TO users;
-- Note: This is a simplified example. In practice, you'd need to recreate all constraints and indexes
```

## Best Practices

1. **Version Numbers**: Use sequential version numbers (V001, V002, etc.)
2. **Descriptive Names**: Use clear, descriptive names for migrations
3. **Idempotent**: Write migrations that can be run multiple times safely
4. **Test First**: Test migrations on a copy of production data
5. **Backup**: Always backup before applying migrations in production
6. **Small Changes**: Keep migrations focused on specific changes
7. **Documentation**: Comment your SQL for clarity

## Automatic Application

Migrations are automatically applied when the DatabaseService is initialized. The system:

1. Scans the migrations directory for SQL files
2. Checks which migrations have already been applied
3. Applies pending migrations in order
4. Records successful migrations in the `schema_migrations` table

## Migration Tracking

Applied migrations are tracked in the `schema_migrations` table:

```sql
CREATE TABLE schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_id TEXT NOT NULL UNIQUE,
    description TEXT,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT
);
```

This ensures migrations are only applied once and provides an audit trail.

## Testing

Run the test script to verify migration functionality:

```bash
cd backend
python -m tests.test_migrations
# or
python tests/test_migrations.py
```

This will:
- Test migration file creation
- Test migration application
- Test database operations
- Verify schema integrity
