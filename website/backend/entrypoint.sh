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

echo "Applying migrations"
# Find and apply migration files
for migration in /root/sql/migrations/*.sql; do
  echo "Applying migration: $migration"
  PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$migration" || echo "Warning: Migration $migration may have had errors"
done

# Import the R2 database dump if download was successful
if [ "$R2_IMPORT_SUCCESS" = true ] && [ -f "/tmp/bookdb_dump.sql" ]; then
  echo "Importing database dump from R2"
  if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/bookdb_dump.sql; then
    echo "Database import completed successfully"
  else
    echo "Warning: Database import may have had errors, but continuing"
  fi
else
  echo "Skipping database import - no dump file was found or download failed"
fi

echo "Database setup complete, starting server"
exec ./server
