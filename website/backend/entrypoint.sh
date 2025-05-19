#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
until PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; do
  echo "PostgreSQL is unavailable - sleeping for 2 seconds"
  sleep 2
done
echo "PostgreSQL is up - proceeding with migrations"

# Configure AWS CLI for Cloudflare R2
mkdir -p ~/.aws
echo "[default]
aws_access_key_id=$R2_ACCESS_KEY_ID
aws_secret_access_key=$R2_SECRET_ACCESS_KEY
" > ~/.aws/credentials

echo "Downloading database dump from Cloudflare R2"
aws s3 --endpoint-url=$R2_ENDPOINT_URL cp s3://$R2_BUCKET_NAME/$R2_OBJECT_KEY /tmp/bookdb_dump.sql

echo "Applying migrations"
# Find and apply migration files
for migration in /root/sql/migrations/*.sql; do
  echo "Applying migration: $migration"
  PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$migration"
done

echo "Importing database dump from R2"
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/bookdb_dump.sql

echo "Database setup complete, starting server"
exec ./server
