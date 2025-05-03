#!/bin/bash

# Script to setup PostgreSQL database and import data
set -e

# Load environment variables from .env file
if [ -f ../.env ]; then
  source ../.env
elif [ -f ./.env ]; then
  source ./.env
else
  echo "Error: .env file not found"
  exit 1
fi

# Use environment variables or fallback to defaults
DB_NAME="${POSTGRES_DB:-bookdb}"
DB_USER="${POSTGRES_USER:-bookdbuser}"
DB_PASSWORD="${POSTGRES_PASSWORD:-bookdbpassword}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"

echo "Setting up PostgreSQL database..."

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL is not installed. Installing PostgreSQL..."
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib
    
    # Start PostgreSQL service
    sudo systemctl enable postgresql
    sudo systemctl start postgresql
else
    echo "PostgreSQL is already installed."
fi

# Terminate existing connections and drop database
echo "Terminating all connections to database if it exists..."
sudo -u postgres psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$DB_NAME';" || true

echo "Dropping database if it exists..."
sudo -u postgres psql -c "DROP DATABASE IF EXISTS $DB_NAME;"

# Handle user creation or update
echo "Setting up user $DB_USER..."
sudo -u postgres psql -c "DO \$\$ 
BEGIN 
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = '$DB_USER') THEN
        ALTER USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
    ELSE
        CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
    END IF;
END
\$\$;"

# Create the database with new owner
echo "Creating database $DB_NAME..."
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

echo "Granting privileges to $DB_USER..."
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

# Import data from SQL backup file
echo "Importing data from SQL backup file..."
if [ -f ../db_backups/bookdb_users.sql ]; then
  PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -h $DB_HOST -p $DB_PORT -d $DB_NAME -f ../db_backups/bookdb_users.sql
elif [ -f ./db_backups/bookdb_users.sql ]; then
  PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -h $DB_HOST -p $DB_PORT -d $DB_NAME -f ./db_backups/bookdb_users.sql
else
  echo "Warning: SQL backup file not found"
fi

echo "Database setup completed!"
echo "Database Name: $DB_NAME"
echo "Username: $DB_USER"
echo "Password: $DB_PASSWORD"
echo "Connection string: postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"