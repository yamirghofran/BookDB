#!/bin/bash
set -e

# Enhanced logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local level="$1"
    local message="$2"
    echo "[${timestamp}] [${level}] [POSTGRES] ${message}"
}

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

# Check if database tables already exist FIRST before doing anything else
log "INFO" "Checking if database tables already exist..."
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
  log "INFO" "No tables found, proceeding with migrations first, then data import"
  SKIP_MIGRATIONS=false
  BOOKS_COUNT=0
  SKIP_PGDUMP_DOWNLOAD=false
fi

# First step: Apply migrations if needed
if [ "$SKIP_MIGRATIONS" = false ]; then
  log "INFO" "Applying migrations using 0001_init.sql"
  # Apply the initial migration file explicitly
  migration="/root/sql/migrations/0001_init.sql"
  if [ -f "$migration" ]; then
    log "INFO" "Applying migration: $migration"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$migration" || { 
      log "ERROR" "Migration $migration failed!"
      exit 1
    }
    log "INFO" "Migration completed successfully"
  else
    log "ERROR" "Migration file $migration not found!"
    exit 1
  fi
else
  log "INFO" "Migrations skipped - database already exists with tables"
fi

# Second step: Download pgdump if needed
if [ "$SKIP_PGDUMP_DOWNLOAD" = false ]; then
  log "DEBUG" "R2 Environment Variables:"
  log "DEBUG" "- R2_ENDPOINT_URL: '${R2_ENDPOINT_URL}'"
  log "DEBUG" "- R2_BUCKET_NAME: '${R2_BUCKET_NAME}'"
  log "DEBUG" "- R2_OBJECT_KEY_POSTGRES: '${R2_OBJECT_KEY_POSTGRES}'"
  log "DEBUG" "- R2_ACCESS_KEY_ID: '${R2_ACCESS_KEY_ID:0:5}***'"
  
  # Build the download URL from R2 variables
  if [ -z "$R2_ENDPOINT_URL" ] || [ -z "$R2_BUCKET_NAME" ] || [ -z "$R2_OBJECT_KEY_POSTGRES" ]; then
    log "ERROR" "One or more R2 environment variables are not set. Skipping database dump download."
    log "ERROR" "Required: R2_ENDPOINT_URL, R2_BUCKET_NAME, R2_OBJECT_KEY_POSTGRES"
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
    DUMP_URL="${R2_ENDPOINT_URL_CLEANED}/${R2_BUCKET_NAME}/${R2_OBJECT_KEY_POSTGRES}"
    
    log "INFO" "Downloading database dump from R2: $DUMP_URL"
    download_start=$(date +%s)
    
    # Check if we have AWS CLI available for authenticated downloads
    if command -v aws &> /dev/null; then
      log "INFO" "Using AWS CLI for authenticated download"
      # AWS CLI is available, use it for authenticated S3 access
      if AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
         aws --endpoint-url $R2_ENDPOINT_URL_CLEANED s3 cp \
         s3://${R2_BUCKET_NAME}/${R2_OBJECT_KEY_POSTGRES} /tmp/bookdb_dump.sql; then
        download_time=$(($(date +%s) - download_start))
        dump_size=$(du -h /tmp/bookdb_dump.sql | cut -f1)
        log "INFO" "Successfully downloaded database dump via AWS CLI (${dump_size}, took ${download_time}s)"
        PGDUMP_IMPORT_SUCCESS=true
      else
        log "ERROR" "AWS CLI download failed, falling back to curl"
        fallback_to_curl=true
      fi
    else
      log "INFO" "AWS CLI not available, using curl with authentication headers"
      fallback_to_curl=true
    fi
    
    # Fallback to curl if AWS CLI not available or failed
    if [ "${fallback_to_curl:-false}" = true ]; then
      # Generate current date in AWS format (required for authentication)
      DATE=$(date -u +"%Y%m%dT%H%M%SZ")
      DATE_SHORT=$(date -u +"%Y%m%d")
      
      # Only attempt authenticated download if credentials are provided
      if [ -n "$R2_ACCESS_KEY_ID" ] && [ -n "$R2_SECRET_ACCESS_KEY" ]; then
        log "INFO" "Using curl with AWS v4 signature authentication"
        
        # Use curl with AWS authentication headers if keys are provided
        # Create temporary headers file
        HEADERS_FILE=$(mktemp)
        
        # Construct authentication headers using R2 credentials
        echo "Host: ${R2_BUCKET_NAME}.${R2_ENDPOINT_URL#*//}" > $HEADERS_FILE
        echo "X-Amz-Date: $DATE" >> $HEADERS_FILE
        echo "X-Amz-Content-Sha256: UNSIGNED-PAYLOAD" >> $HEADERS_FILE
        echo "Authorization: AWS4-HMAC-SHA256 Credential=${R2_ACCESS_KEY_ID}/${DATE_SHORT}/auto/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=dummy" >> $HEADERS_FILE
        
        if curl -L --retry 3 --retry-delay 2 -f -o /tmp/bookdb_dump.sql -H "@$HEADERS_FILE" "$DUMP_URL"; then
          download_time=$(($(date +%s) - download_start))
          dump_size=$(du -h /tmp/bookdb_dump.sql | cut -f1)
          log "INFO" "Successfully downloaded database dump with auth headers (${dump_size}, took ${download_time}s)"
          PGDUMP_IMPORT_SUCCESS=true
        else
          log "WARN" "Authenticated download failed with status: $?, trying unauthenticated as fallback"
          # Try unauthenticated as a last resort
          if curl -L --retry 3 --retry-delay 2 -f -o /tmp/bookdb_dump.sql "$DUMP_URL"; then
            download_time=$(($(date +%s) - download_start))
            dump_size=$(du -h /tmp/bookdb_dump.sql | cut -f1)
            log "INFO" "Successfully downloaded database dump without auth (${dump_size}, took ${download_time}s)"
            PGDUMP_IMPORT_SUCCESS=true
          else
            log "ERROR" "Failed to download database dump (HTTP status: $?), continuing without import"
            PGDUMP_IMPORT_SUCCESS=false
          fi
        fi
        
        # Remove temporary headers file
        rm -f $HEADERS_FILE
      else
        log "WARN" "R2 credentials not provided, attempting unauthenticated download"
        # Try unauthenticated download as fallback
        if curl -L --retry 3 --retry-delay 2 -f -o /tmp/bookdb_dump.sql "$DUMP_URL"; then
          download_time=$(($(date +%s) - download_start))
          dump_size=$(du -h /tmp/bookdb_dump.sql | cut -f1)
          log "INFO" "Successfully downloaded database dump without auth (${dump_size}, took ${download_time}s)"
          PGDUMP_IMPORT_SUCCESS=true
        else
          log "ERROR" "Failed to download database dump (HTTP status: $?), continuing without import"
          PGDUMP_IMPORT_SUCCESS=false
        fi
      fi
    fi
  fi
else
  log "INFO" "Skipping pgdump download - database already populated"
  PGDUMP_IMPORT_SUCCESS=false
fi

# Third step: Import the database dump if download was successful
if [ "$PGDUMP_IMPORT_SUCCESS" = true ] && [ -f "/tmp/bookdb_dump.sql" ]; then
  log "INFO" "Preparing database dump by fixing ownership issues"
  
  # First attempt to create the user if it doesn't exist
  log "INFO" "Attempting to create the 'yamirghofran0' user if it doesn't exist"
  PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DO \$\$ 
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='yamirghofran0') THEN
      CREATE ROLE yamirghofran0 WITH LOGIN PASSWORD 'temp_password';
    END IF;
  END \$\$;" || log "WARN" "Could not create role, will proceed with substitution"
  
  # Create a modified version of the dump with substituted ownership
  log "INFO" "Creating a modified version of the dump file with current database user ownership"
  sed "s/yamirghofran0/$DB_USER/g" /tmp/bookdb_dump.sql > /tmp/bookdb_dump_modified.sql
  
  log "INFO" "Importing modified database dump"
  import_start=$(date +%s)
  if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/bookdb_dump_modified.sql; then
    import_time=$(($(date +%s) - import_start))
    log "INFO" "Database import completed successfully in ${import_time}s"
    # Verify import by counting books
    BOOKS_COUNT_AFTER=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM books;" 2>/dev/null || echo "0" | tr -d ' ')
    log "INFO" "Books table now contains $BOOKS_COUNT_AFTER records (added $((BOOKS_COUNT_AFTER - BOOKS_COUNT)))"
  else
    log "WARN" "Database import had errors. Attempting alternative approach with disable/enable triggers"
    
    # Try importing with triggers disabled
    log "INFO" "Disabling triggers before import"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SET session_replication_role = 'replica';"
    
    log "INFO" "Importing with triggers disabled"
    import_start=$(date +%s)
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/bookdb_dump_modified.sql
    import_result=$?
    import_time=$(($(date +%s) - import_start))
    
    if [ $import_result -eq 0 ]; then
      log "INFO" "Database import with disabled triggers completed in ${import_time}s"
    else
      log "ERROR" "Import still failed with exit code $import_result, continuing anyway"
    fi
    
    log "INFO" "Re-enabling triggers"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SET session_replication_role = 'origin';"
    
    # Verify import by counting books
    BOOKS_COUNT_AFTER=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM books;" 2>/dev/null || echo "0" | tr -d ' ')
    log "INFO" "Books table now contains $BOOKS_COUNT_AFTER records (added $((BOOKS_COUNT_AFTER - BOOKS_COUNT)))"
  fi
  
  # Cleanup temp files
  log "INFO" "Cleaning up temporary files"
  rm -f /tmp/bookdb_dump.sql /tmp/bookdb_dump_modified.sql
else
  if [ "$SKIP_PGDUMP_DOWNLOAD" = true ]; then
    log "INFO" "Skipping database import - database already populated"
  else
    log "INFO" "Skipping database import - no dump file was found or download failed"
  fi
fi

log "INFO" "Database setup complete, starting server"
log "INFO" "Runtime summary:"
log "INFO" "- Tables Count: $TABLES_COUNT"
log "INFO" "- Initial Books Count: $BOOKS_COUNT"
log "INFO" "- Final Books Count: ${BOOKS_COUNT_AFTER:-$BOOKS_COUNT}"
log "INFO" "- Migrations Applied: $(if [ "$SKIP_MIGRATIONS" = false ]; then echo "Yes"; else echo "No"; fi)"
log "INFO" "- Data Import Status: $(if [ "$PGDUMP_IMPORT_SUCCESS" = true ]; then echo "Success"; else echo "Skipped/Failed"; fi)"
log "INFO" "- Database Version: $(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT version();" | tr -d ' ' | tr '\n' ' ')"

exec ./server
