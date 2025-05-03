#!/bin/bash

# Script to setup PostgreSQL and restore the database
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get the absolute path to the project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo -e "${YELLOW}Script directory: ${SCRIPT_DIR}${NC}"
echo -e "${YELLOW}Project directory: ${PROJECT_DIR}${NC}"

# Function to check and install required system packages
check_and_install_system_requirements() {
    local packages=("$@")
    local missing_packages=()
    
    for pkg in "${packages[@]}"; do
        if ! command -v "$pkg" &> /dev/null; then
            missing_packages+=("$pkg")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo -e "${YELLOW}Installing required packages: ${missing_packages[*]}${NC}"
        sudo apt update
        sudo apt install -y "${missing_packages[@]}"
    fi
}

# Check for Python, pip and other required packages
check_and_install_system_requirements python3 pip curl

# Load environment variables if .env exists
ENV_FILE="${PROJECT_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Loading environment variables from ${ENV_FILE}${NC}"
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
else
    echo -e "${YELLOW}No .env file found, using default values.${NC}"
fi

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${YELLOW}PostgreSQL is not installed. Installing PostgreSQL...${NC}"
    
    # Install PostgreSQL (works for Ubuntu/Debian)
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib
    
    # Start PostgreSQL service
    sudo systemctl enable postgresql
    sudo systemctl start postgresql
    
    echo -e "${GREEN}PostgreSQL installed successfully!${NC}"
else
    echo -e "${GREEN}PostgreSQL is already installed.${NC}"
fi

# Check if PostgreSQL service is running
if ! pg_isready &> /dev/null; then
    echo -e "${YELLOW}PostgreSQL service is not running. Starting service...${NC}"
    sudo systemctl start postgresql
    sleep 2
    
    # Check again if service started
    if ! pg_isready &> /dev/null; then
        echo -e "${RED}Failed to start PostgreSQL service. Please check your installation.${NC}"
        exit 1
    fi
fi

# Check if PostgreSQL port is in use by another process
PG_PORT=5432
if lsof -i :$PG_PORT | grep -v postgres &> /dev/null; then
    echo -e "${RED}Error: Port $PG_PORT is being used by another process!${NC}"
    lsof -i :$PG_PORT
    echo -e "${RED}Please stop the conflicting process and try again.${NC}"
    exit 1
fi

# Variables for database setup - use environment variables if set, otherwise use defaults
DB_NAME="${POSTGRES_DB:-bookdb}"
DB_USER="${POSTGRES_USER:-bookdbuser}"
DB_PASSWORD="${POSTGRES_PASSWORD:-bookdbpassword}"
DUMP_FILE="${PROJECT_DIR}/db_backups/bookdb_users.sql"

# Create db_backups directory if it doesn't exist
mkdir -p "${PROJECT_DIR}/db_backups"

echo -e "${YELLOW}Database configuration:${NC}"
echo -e "  Database: ${DB_NAME}"
echo -e "  User:     ${DB_USER}"
echo -e "  Password: ${DB_PASSWORD}"
echo -e "${YELLOW}Using dump file: ${DUMP_FILE}${NC}"

# Check if dump file exists, if not try to download it
if [ ! -f "$DUMP_FILE" ]; then
    echo -e "${YELLOW}Dump file not found at ${DUMP_FILE}${NC}"
    echo -e "${YELLOW}Attempting to download dump file...${NC}"
    
    # Install required Python packages for the download script
    echo -e "${YELLOW}Installing required Python packages...${NC}"
    pip install boto3 python-dotenv

    # Make sure the download script is executable
    chmod +x "${SCRIPT_DIR}/download_pgdump.py"
    
    # Run the download script
    "${SCRIPT_DIR}/download_pgdump.py"
    
    # Check if download was successful
    if [ ! -f "$DUMP_FILE" ]; then
        echo -e "${RED}Failed to download dump file using script.${NC}"
        echo -e "${RED}Please download the dump file manually to:${NC}"
        echo -e "${RED}${DUMP_FILE}${NC}"
        exit 1
    fi
fi

# Make a temporary copy with appropriate permissions
TEMP_DUMP="/tmp/bookdb_users_temp.sql"
echo -e "${YELLOW}Copying dump file to temporary location with appropriate permissions...${NC}"
# Create with current user permissions first
cp "$DUMP_FILE" "$TEMP_DUMP" || {
    echo -e "${YELLOW}Permission issue when copying to /tmp, trying with sudo...${NC}"
    sudo cp "$DUMP_FILE" "$TEMP_DUMP"
}

# Set proper permissions
sudo chmod 644 "$TEMP_DUMP"
sudo chown postgres:postgres "$TEMP_DUMP" || {
    echo -e "${YELLOW}Warning: Could not change ownership of the dump file.${NC}"
    echo -e "${YELLOW}Will try to continue anyway...${NC}"
}

# Create database and user
echo -e "${YELLOW}Creating database and user...${NC}"

# Execute PostgreSQL commands as postgres user with error handling
echo "Creating user $DB_USER..."
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" || {
    echo "User may already exist, trying to update password"
    sudo -u postgres psql -c "ALTER USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" || echo "Warning: Could not update user"
}

echo "Dropping database if it exists..."
sudo -u postgres psql -c "DROP DATABASE IF EXISTS $DB_NAME;" || {
    echo -e "${YELLOW}Warning: Could not drop database, it may be in use.${NC}"
    echo "Attempting to terminate connections and retry..."
    sudo -u postgres psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$DB_NAME';"
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS $DB_NAME;" || echo "Warning: Still could not drop database"
}

echo "Creating database $DB_NAME..."
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" || {
    echo -e "${RED}Error: Could not create database.${NC}"
    exit 1
}

echo "Granting privileges to $DB_USER..."
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

echo -e "${GREEN}Database and user created successfully!${NC}"

# Restore the database from temporary dump file
echo -e "${YELLOW}Restoring database from dump file...${NC}"
# Fix common SQL syntax issues
sed -i "s/child's/child\\\\'s/g" "$TEMP_DUMP" 2>/dev/null || true

# Restore the database
sudo -u postgres psql -d $DB_NAME -f "$TEMP_DUMP" || {
    echo -e "${RED}Error: Failed to restore database from dump file.${NC}"
    echo -e "${RED}Please check the dump file and try again.${NC}"
    exit 1
}

# Clean up temp file
sudo rm -f "$TEMP_DUMP"

echo -e "${GREEN}Database restored successfully!${NC}"
echo -e "${GREEN}PostgreSQL database setup completed.${NC}"
echo -e "${YELLOW}Database Name:${NC} $DB_NAME"
echo -e "${YELLOW}Username:${NC} $DB_USER"
echo -e "${YELLOW}Password:${NC} $DB_PASSWORD"

# Create a simple connection test script using absolute paths
cat > "${PROJECT_DIR}/test_db_connection.py" << EOL
#!/usr/bin/env python3

import psycopg2
import sys
import os
from dotenv import load_dotenv

# Try to load .env file if it exists
env_file = "${PROJECT_DIR}/.env"
if os.path.exists(env_file):
    load_dotenv(env_file)

# Get database connection details from environment variables or use defaults
db_name = os.environ.get("POSTGRES_DB", "$DB_NAME")
db_user = os.environ.get("POSTGRES_USER", "$DB_USER")
db_password = os.environ.get("POSTGRES_PASSWORD", "$DB_PASSWORD")
db_host = os.environ.get("POSTGRES_HOST", "localhost")

try:
    # Connect to the database
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host
    )
    
    # Create a cursor
    cur = conn.cursor()
    
    # Execute a test query
    cur.execute("SELECT COUNT(*) FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';")
    
    # Fetch the result
    result = cur.fetchone()
    print(f"Successfully connected to database. Found {result[0]} tables.")
    
    # Close the cursor and connection
    cur.close()
    conn.close()
    
    sys.exit(0)
except Exception as e:
    print(f"Error connecting to database: {e}")
    sys.exit(1)
EOL

# Make the test script executable
chmod +x "${PROJECT_DIR}/test_db_connection.py"

# Install Python requirements
echo -e "${YELLOW}Installing required Python packages for database connection...${NC}"
pip install psycopg2-binary python-dotenv

# Update requirements.txt to include psycopg2 and python-dotenv if not already present
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    if ! grep -q "psycopg2" "${PROJECT_DIR}/requirements.txt"; then
        echo "psycopg2-binary" >> "${PROJECT_DIR}/requirements.txt"
        echo -e "${GREEN}Added psycopg2-binary to requirements.txt${NC}"
    fi
    
    if ! grep -q "python-dotenv" "${PROJECT_DIR}/requirements.txt"; then
        echo "python-dotenv" >> "${PROJECT_DIR}/requirements.txt"
        echo -e "${GREEN}Added python-dotenv to requirements.txt${NC}"
    fi
else
    # Create requirements.txt if it doesn't exist
    echo "psycopg2-binary" > "${PROJECT_DIR}/requirements.txt"
    echo "python-dotenv" >> "${PROJECT_DIR}/requirements.txt"
    echo -e "${GREEN}Created requirements.txt with psycopg2-binary and python-dotenv${NC}"
fi

echo -e "${YELLOW}Testing database connection...${NC}"
python3 "${PROJECT_DIR}/test_db_connection.py"

# Check the result of the connection test
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Connection test successful!${NC}"
else
    echo -e "${RED}Connection test failed. Please check your database configuration.${NC}"
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "You can connect to the database using the following connection string:"
echo -e "${YELLOW}postgresql://$DB_USER:$DB_PASSWORD@localhost/$DB_NAME${NC}"