#!/bin/bash
# This script creates a recovery configuration for PostgreSQL
# to help recover from hard shutdowns.

set -e

# Check if the postgres data directory exists
if [ -d /var/lib/postgresql/data/base ]; then
  echo "PostgreSQL data directory already exists, checking for recovery needs..."
  
  # Check if there are any archive files
  if [ "$(ls -A /var/lib/postgresql/archive 2>/dev/null)" ]; then
    echo "Archive files found, setting up recovery configuration..."
    
    # Create the recovery.signal file to trigger recovery mode
    touch /var/lib/postgresql/data/recovery.signal
    
    # Create or update recovery configuration
    cat > /var/lib/postgresql/data/postgresql.auto.conf << EOF
# Recovery settings added by container initialization
restore_command = 'cp /var/lib/postgresql/archive/%f %p'
recovery_target_timeline = 'latest'
EOF

    echo "Recovery configuration created. PostgreSQL will attempt recovery on startup."
  else
    echo "No archive files found. Standard startup will be attempted."
  fi
else
  echo "Fresh PostgreSQL installation, no recovery needed."
fi

# Continue with regular PostgreSQL startup
exec docker-entrypoint.sh postgres
