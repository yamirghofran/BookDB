#!/bin/bash
set -e

# Enhanced logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local level="$1"
    local message="$2"
    echo "[${timestamp}] [${level}] [POSTGRES] ${message}"
}

# Function to fix ownership and permissions after import
fix_ownership() {
  log "INFO" "Fixing ownership and permissions for database objects..."
  
  # Generate and run SQL to reassign owned objects to current user
  cat > /tmp/fix_ownership.sql << EOF
DO \$\$
DECLARE
  r RECORD;
BEGIN
  -- Reassign database objects
  FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public'
  LOOP
    EXECUTE 'ALTER TABLE public.' || quote_ident(r.tablename) || ' OWNER TO ' || quote_ident('$DB_USER');
  END LOOP;
  
  -- Sequences
  FOR r IN SELECT sequencename FROM pg_sequences WHERE schemaname = 'public'
  LOOP
    EXECUTE 'ALTER SEQUENCE public.' || quote_ident(r.sequencename) || ' OWNER TO ' || quote_ident('$DB_USER');
  END LOOP;
  
  -- Functions
  FOR r IN SELECT proname FROM pg_proc p JOIN pg_namespace n ON p.pronamespace = n.oid WHERE n.nspname = 'public'
  LOOP
    -- Cannot simply alter owner of functions without knowing argument types
    -- This is a simplification, may not work for all functions
    BEGIN
      EXECUTE 'ALTER FUNCTION public.' || quote_ident(r.proname) || '() OWNER TO ' || quote_ident('$DB_USER');
    EXCEPTION WHEN OTHERS THEN
      -- Silently continue on error
    END;
  END LOOP;
  
  -- Fix sequences after data import
  FOR r IN 
    SELECT 
      sequence_name, 
      table_name || '_' || column_name || '_seq' AS expected_sequence
    FROM 
      information_schema.columns
    WHERE 
      column_default LIKE 'nextval%'
  LOOP
    BEGIN
      EXECUTE 'SELECT setval(''' || r.sequence_name || ''', (SELECT COALESCE(MAX(' || 
              substring(r.expected_sequence from 1 for position('_seq' in r.expected_sequence)-1) || '), 1) FROM ' || 
              substring(r.expected_sequence from 1 for position('_' in r.expected_sequence)-1) || '))';
    EXCEPTION WHEN OTHERS THEN
      -- Silently continue on error
    END;
  END LOOP;
END \$\$;
EOF

  PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/fix_ownership.sql
  log "INFO" "Ownership correction completed"
  rm -f /tmp/fix_ownership.sql
}

# Function to set up the database - can be called directly when sourced
setup_database() {
  log "INFO" "Setting up PostgreSQL database..."
  
  # First: Check if the database has tables
  TABLES_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
  log "INFO" "Found $TABLES_COUNT tables in database schema"

  if [ "$TABLES_COUNT" -gt 0 ]; then
    log "INFO" "Database already has $TABLES_COUNT tables, checking for data..."
    
    # Check if there's actual data in a key table (assuming books table exists)
    BOOKS_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM books;" 2>/dev/null || echo "0" | tr -d ' ')
    log "INFO" "Books table contains $BOOKS_COUNT records"
    
    if [ "$BOOKS_COUNT" -gt 0 ]; then
      log "INFO" "Found $BOOKS_COUNT books in the database. Database is already populated, skipping migrations and data import."
      SKIP_MIGRATIONS=true
      SKIP_PGDUMP_DOWNLOAD=true
    else
      log "INFO" "Tables exist but no data found. Will apply pgdump for data import only."
      SKIP_MIGRATIONS=true
      SKIP_PGDUMP_DOWNLOAD=false
    fi
  else
    log "INFO" "No tables found, will download and import PostgreSQL dump"
    SKIP_MIGRATIONS=true # Skip SQL migrations, use pgdump instead
    BOOKS_COUNT=0
    SKIP_PGDUMP_DOWNLOAD=false
  fi
  
  # Third: Download and import data if needed
  if [ "$SKIP_PGDUMP_DOWNLOAD" = false ]; then
    log "INFO" "Downloading database dump from R2..."
    
    # Build the download URL from R2 variables
    if [ -z "$R2_ENDPOINT_URL" ] || [ -z "$R2_BUCKET_NAME" ] || [ -z "$R2_OBJECT_KEY_POSTGRES" ]; then
      log "ERROR" "One or more R2 environment variables are not set. Skipping database dump download."
      log "ERROR" "Required: R2_ENDPOINT_URL, R2_BUCKET_NAME, R2_OBJECT_KEY_POSTGRES"
      PGDUMP_IMPORT_SUCCESS=false
    else
      # Make sure endpoint URL has proper format
      R2_ENDPOINT_URL_CLEANED=${R2_ENDPOINT_URL%/}
      # Add https:// prefix if needed
      if [[ "$R2_ENDPOINT_URL_CLEANED" != http* ]]; then
        R2_ENDPOINT_URL_CLEANED="https://$R2_ENDPOINT_URL_CLEANED"
      fi
      
      # Download the database dump using curl with endpoint and object key directly
      DUMP_URL="${R2_ENDPOINT_URL_CLEANED}/${R2_OBJECT_KEY_POSTGRES}"
      log "INFO" "Downloading database dump from: $DUMP_URL"
      
      if curl -L --retry 3 --retry-delay 2 -f -o /tmp/bookdb_dump.sql "$DUMP_URL"; then
        dump_size=$(du -h /tmp/bookdb_dump.sql | cut -f1)
        log "INFO" "Successfully downloaded database dump (${dump_size})"
        PGDUMP_IMPORT_SUCCESS=true
        
        # Preprocess the SQL dump to replace role references
        log "INFO" "Preprocessing database dump to replace role references..."
        # Create temporary copy for processing
        cp /tmp/bookdb_dump.sql /tmp/bookdb_dump_original.sql
        
        # Replace role references and other potential issues
        sed -i 's/role "yamirghofran0"/role "'$DB_USER'"/g' /tmp/bookdb_dump.sql
        sed -i 's/yamirghofran0/'$DB_USER'/g' /tmp/bookdb_dump.sql
        sed -i '/^SET transaction_timeout/d' /tmp/bookdb_dump.sql  # Remove unrecognized parameter
        
        log "INFO" "Preprocessed database dump. Importing..."
        
        # Import the SQL dump with owner reassignment
        if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -v ON_ERROR_STOP=0 -f /tmp/bookdb_dump.sql; then
          log "INFO" "Database import completed successfully"
          
          # Fix ownership of database objects
          fix_ownership
        else
          log "WARN" "Database import had errors, trying with session_replication_role = 'replica'..."
          # Try importing with triggers disabled
          PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SET session_replication_role = 'replica';"
          PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -v ON_ERROR_STOP=0 -f /tmp/bookdb_dump.sql
          PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SET session_replication_role = 'origin';"
          
          log "INFO" "Checking if tables were created successfully..."
          TABLES_COUNT_AFTER=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
          log "INFO" "Found $TABLES_COUNT_AFTER tables after import"
          
          if [ "$TABLES_COUNT_AFTER" -gt 0 ]; then
            log "INFO" "Tables were created despite errors, continuing..."
            
            # Fix ownership of database objects even after partial success
            fix_ownership
          else
            log "ERROR" "Failed to create tables. Creating schema from 0001_init.sql as fallback..."
            # Apply the initial migration file as fallback
            migration="./sql/migrations/0001_init.sql"
            if [ -f "$migration" ]; then
              PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$migration"
              log "INFO" "Applied fallback migration"
            else
              log "ERROR" "Migration file $migration not found! Database setup may be incomplete."
            fi
          fi
        fi
        
        # Clean up
        rm -f /tmp/bookdb_dump.sql /tmp/bookdb_dump_original.sql
      else
        log "ERROR" "Failed to download database dump, continuing without import"
        PGDUMP_IMPORT_SUCCESS=false
      fi
    fi
  else
    log "INFO" "Skipping database dump download - database already populated"
    PGDUMP_IMPORT_SUCCESS=false
  fi
  
  log "INFO" "Database setup completed successfully"
}

# Main script execution - only runs if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  log "INFO" "Starting PostgreSQL entrypoint script"
  log "INFO" "Environment: APP_ENV=${APP_ENV:-unknown}"
  log "INFO" "Database Connection: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
  
  log "INFO" "Waiting for PostgreSQL to be ready at ${DB_HOST}:${DB_PORT}..."
  start_time=$(date +%s)
  until PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; do
    elapsed=$(($(date +%s) - start_time))
    log "WARN" "PostgreSQL is unavailable (waited ${elapsed}s) - sleeping for 2 seconds"
    sleep 2
    # Fail if we've waited too long (5 minutes)
    if [ $elapsed -gt 300 ]; then
      log "ERROR" "Timed out waiting for PostgreSQL after ${elapsed} seconds"
      exit 1
    fi
  done
  log "INFO" "PostgreSQL is up after $(($(date +%s) - start_time)) seconds - proceeding with setup"
  
  # Call the setup_database function to check tables and set appropriate flags
  setup_database

# The setup_database function has already handled all aspects of the database initialization

# Final status information for logging
BOOKS_COUNT_AFTER=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM books;" 2>/dev/null || echo "0" | tr -d ' ')

log "INFO" "Database setup complete, starting server"
log "INFO" "Runtime summary:"
log "INFO" "- Tables Count: $TABLES_COUNT"
log "INFO" "- Initial Books Count: $BOOKS_COUNT"
log "INFO" "- Final Books Count: ${BOOKS_COUNT_AFTER:-$BOOKS_COUNT}"
log "INFO" "- Migrations Applied: $(if [ "$SKIP_MIGRATIONS" = false ]; then echo "Yes"; else echo "No"; fi)"
log "INFO" "- Data Import Status: $(if [ "$PGDUMP_IMPORT_SUCCESS" = true ]; then echo "Success"; else echo "Skipped/Failed"; fi)"
log "INFO" "- Database Version: $(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT version();" | tr -d ' ' | tr '\n' ' ')"

  # Execute the server only if this script is run directly, not sourced
  exec ./server
fi
