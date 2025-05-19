#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
until PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; do
  echo "PostgreSQL is unavailable - sleeping for 2 seconds"
  sleep 2
done
echo "PostgreSQL is up - proceeding with setup"

# Check if database tables already exist FIRST before doing anything else
echo "Checking if database tables already exist..."
TABLES_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')

if [ "$TABLES_COUNT" -gt 0 ]; then
  echo "Database already has $TABLES_COUNT tables, checking for data..."
  
  # Check if there's actual data in a key table (assuming books table exists)
  BOOKS_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM books;" 2>/dev/null || echo "0" | tr -d ' ')
  
  if [ "$BOOKS_COUNT" -gt 0 ]; then
    echo "Found $BOOKS_COUNT books in the database. Database is already populated, skipping migrations and data import."
    SKIP_MIGRATIONS=true
    SKIP_PGDUMP_DOWNLOAD=true
  else
    echo "Tables exist but no data found. Will apply pgdump for data import only."
    SKIP_MIGRATIONS=true
    SKIP_PGDUMP_DOWNLOAD=false
  fi
else
  echo "No tables found, proceeding with migrations first, then data import"
  SKIP_MIGRATIONS=false
  BOOKS_COUNT=0
  SKIP_PGDUMP_DOWNLOAD=false
fi

# First step: Apply migrations if needed
if [ "$SKIP_MIGRATIONS" = false ]; then
  echo "Applying migrations using 0001_init.sql"
  # Apply the initial migration file explicitly
  migration="/root/sql/migrations/0001_init.sql"
  if [ -f "$migration" ]; then
    echo "Applying migration: $migration"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$migration" || { 
      echo "Error: Migration $migration failed!"
      exit 1
    }
    echo "Migration completed successfully"
  else
    echo "Error: Migration file $migration not found!"
    exit 1
  fi
else
  echo "Migrations skipped - database already exists with tables"
fi

# Second step: Download pgdump if needed
if [ "$SKIP_PGDUMP_DOWNLOAD" = false ]; then
  echo "Debug: Environment Variables"
  echo "R2_ENDPOINT_URL: '${R2_ENDPOINT_URL}'"
  echo "R2_BUCKET_NAME: '${R2_BUCKET_NAME}'"
  echo "R2_OBJECT_KEY: '${R2_OBJECT_KEY}'"
  
  # Build the download URL from R2 variables
  if [ -z "$R2_ENDPOINT_URL" ] || [ -z "$R2_BUCKET_NAME" ] || [ -z "$R2_OBJECT_KEY" ]; then
    echo "ERROR: One or more R2 environment variables are not set. Skipping database dump download."
    PGDUMP_IMPORT_SUCCESS=false
  else
    # Make sure endpoint URL has proper format
    # Remove trailing slash if present
    R2_ENDPOINT_URL_CLEANED=${R2_ENDPOINT_URL%/}
    # Add https:// prefix if needed
    if [[ "$R2_ENDPOINT_URL_CLEANED" != http* ]]; then
      R2_ENDPOINT_URL_CLEANED="https://$R2_ENDPOINT_URL_CLEANED"
    fi
    
    # Construct the full URL (R2 format is typically endpoint/bucket/object)
    DUMP_URL="${R2_ENDPOINT_URL_CLEANED}/${R2_BUCKET_NAME}/${R2_OBJECT_KEY}"
    
    echo "Downloading database dump from constructed URL: $DUMP_URL"
    if curl -L "$DUMP_URL" -o /tmp/bookdb_dump.sql; then
      echo "Successfully downloaded database dump"
      PGDUMP_IMPORT_SUCCESS=true
    else
      echo "Failed to download database dump, continuing without import"
      PGDUMP_IMPORT_SUCCESS=false
    fi
  fi
else
  echo "Skipping pgdump download - database already populated"
  PGDUMP_IMPORT_SUCCESS=false
fi

# Third step: Import the database dump if download was successful
if [ "$PGDUMP_IMPORT_SUCCESS" = true ] && [ -f "/tmp/bookdb_dump.sql" ]; then
  echo "Preparing database dump by fixing ownership issues"
  
  # First attempt to create the user if it doesn't exist
  echo "Attempting to create the 'yamirghofran0' user if it doesn't exist"
  PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DO \$\$ 
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='yamirghofran0') THEN
      CREATE ROLE yamirghofran0 WITH LOGIN PASSWORD 'temp_password';
    END IF;
  END \$\$;" || echo "Could not create role, will proceed with substitution"
  
  # Create a modified version of the dump with substituted ownership
  echo "Creating a modified version of the dump file with current database user ownership"
  sed "s/yamirghofran0/$DB_USER/g" /tmp/bookdb_dump.sql > /tmp/bookdb_dump_modified.sql
  
  echo "Importing modified database dump"
  if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/bookdb_dump_modified.sql; then
    echo "Database import completed successfully"
  else
    echo "Warning: Database import had errors. Attempting alternative approach with disable/enable triggers"
    
    # Try importing with triggers disabled
    echo "Disabling triggers before import"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SET session_replication_role = 'replica';"
    
    echo "Importing with triggers disabled"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/bookdb_dump_modified.sql || echo "Import still failed, continuing anyway"
    
    echo "Re-enabling triggers"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SET session_replication_role = 'origin';"
  fi
else
  if [ "$SKIP_PGDUMP_DOWNLOAD" = true ]; then
    echo "Skipping database import - database already populated"
  else
    echo "Skipping database import - no dump file was found or download failed"
  fi
fi

echo "Database setup complete, starting server"
exec ./server
