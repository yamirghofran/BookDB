#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
until PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; do
  echo "PostgreSQL is unavailable - sleeping for 2 seconds"
  sleep 2
done
echo "PostgreSQL is up - proceeding with migrations"

# Debug log environment variables
echo "Debug: Environment Variables"
echo "R2_ENDPOINT_URL: '${R2_ENDPOINT_URL}'"
echo "R2_BUCKET_NAME: '${R2_BUCKET_NAME}'"
echo "R2_OBJECT_KEY: '${R2_OBJECT_KEY}'"

# Configure AWS CLI for Cloudflare R2
mkdir -p ~/.aws
echo "[default]
aws_access_key_id=$R2_ACCESS_KEY_ID
aws_secret_access_key=$R2_SECRET_ACCESS_KEY
" > ~/.aws/credentials

# Check and fix R2 endpoint URL if needed
if [ -z "$R2_ENDPOINT_URL" ]; then
  echo "ERROR: R2_ENDPOINT_URL is not set. Skipping database dump download."
  R2_IMPORT_SUCCESS=false
else
  # Make sure endpoint URL has proper format (starts with http:// or https://)
  if [[ "$R2_ENDPOINT_URL" != http* ]]; then
    echo "Adding https:// prefix to R2_ENDPOINT_URL"
    R2_ENDPOINT_URL="https://$R2_ENDPOINT_URL"
  fi
  
  echo "Downloading database dump from Cloudflare R2 using endpoint: $R2_ENDPOINT_URL"
  if aws s3 --endpoint-url="$R2_ENDPOINT_URL" cp "s3://$R2_BUCKET_NAME/$R2_OBJECT_KEY" /tmp/bookdb_dump.sql; then
    echo "Successfully downloaded database dump"
    R2_IMPORT_SUCCESS=true
  else
    echo "Failed to download database dump, continuing without import"
    R2_IMPORT_SUCCESS=false
  fi
fi

# Check if database tables already exist
echo "Checking if database tables already exist..."
TABLES_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')

if [ "$TABLES_COUNT" -gt 0 ]; then
  echo "Database already has $TABLES_COUNT tables, skipping migrations"
  SKIP_MIGRATIONS=true
else
  echo "No tables found, proceeding with migrations"
  SKIP_MIGRATIONS=false
fi

if [ "$SKIP_MIGRATIONS" = false ]; then
  echo "Applying migrations"
  # Find and apply migration files
  for migration in /root/sql/migrations/*.sql; do
    echo "Applying migration: $migration"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$migration" || echo "Warning: Migration $migration may have had errors"
  done
else
  echo "Migrations skipped - database already exists with tables"
fi

# Import the R2 database dump if download was successful and tables don't exist
if [ "$R2_IMPORT_SUCCESS" = true ] && [ -f "/tmp/bookdb_dump.sql" ] && [ "$SKIP_MIGRATIONS" = false ]; then
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
  
  echo "Importing modified database dump from R2"
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
  echo "Skipping database import - no dump file was found or download failed"
fi

echo "Database setup complete, starting server"
exec ./server
