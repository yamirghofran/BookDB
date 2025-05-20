#!/bin/bash
set -e

# Enhanced logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local level="$1"
    local message="$2"
    echo "[${timestamp}] [${level}] [ENTRYPOINT] ${message}"
}

log "INFO" "Starting BookDB backend entrypoint script"
log "INFO" "Environment: APP_ENV=${APP_ENV:-unknown}"
log "INFO" "Service type: SERVICE_TYPE=${SERVICE_TYPE:-unspecified}"

# Log important environment variables
log "INFO" "DB Host: ${DB_HOST:-not set}"
log "INFO" "Qdrant Host: ${QDRANT_HOST:-not set}"
log "INFO" "R2 Endpoint Available: $(if [ -n "${R2_ENDPOINT_URL}" ]; then echo "Yes"; else echo "No"; fi)"

# Determine which entrypoint script to run based on the service type
if [ "${SERVICE_TYPE}" = "postgres" ]; then
  log "INFO" "Starting PostgreSQL entrypoint script..."
  exec /bin/bash ./postgres_entrypoint.sh
elif [ "${SERVICE_TYPE}" = "qdrant" ]; then
  log "INFO" "Starting Qdrant entrypoint script..."
  exec /bin/bash ./qdrant_entrypoint.sh
else
  log "INFO" "SERVICE_TYPE not set or unrecognized. Starting with auto-configuration..."
  
  # ===== Check PostgreSQL =====
  log "INFO" "Waiting for PostgreSQL to be ready at ${DB_HOST}:${DB_PORT}..."
  start_time=$(date +%s)
  until PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; do
    elapsed=$(($(date +%s) - start_time))
    log "WARN" "PostgreSQL is unavailable (waited ${elapsed}s) - sleeping for 2 seconds"
    sleep 2
    # Fail if we've waited too long (5 minutes)
    if [ $elapsed -gt 60 ]; then
      log "ERROR" "Timed out waiting for PostgreSQL after ${elapsed} seconds"
      exit 1
    fi
  done
  log "INFO" "PostgreSQL is up after $(($(date +%s) - start_time)) seconds"
  
  # Check if database tables already exist
  log "INFO" "Checking if database tables already exist..."
  TABLES_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
  log "INFO" "Found $TABLES_COUNT tables in database schema"
  
  if [ "$TABLES_COUNT" -gt 0 ]; then
    log "INFO" "Database already has $TABLES_COUNT tables, checking for data..."
    
    # Check if there's actual data in a key table (assuming books table exists)
    BOOKS_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM books;" 2>/dev/null || echo "0" | tr -d ' ')
    log "INFO" "Books table contains $BOOKS_COUNT records"
    
    if [ "$BOOKS_COUNT" -gt 0 ]; then
      log "INFO" "Found $BOOKS_COUNT books in the database. Database is already populated, skipping SQL setup."
      SKIP_SQL_SETUP=true
    else
      log "INFO" "Tables exist but no data found. Will set up database data."
      SKIP_SQL_SETUP=false
    fi
  else
    log "INFO" "No tables found. Will download and import PostgreSQL dump."
    SKIP_SQL_SETUP=false
  fi
  
  # Set up database if needed
  if [ "$SKIP_SQL_SETUP" = false ]; then
    log "INFO" "Setting up PostgreSQL database..."
    
    # Source the functions from PostgreSQL entrypoint script using relative path
    source ./postgres_entrypoint.sh
    
    # Set up the database
    setup_database
    
    log "INFO" "PostgreSQL database setup completed"
  fi
  
  # ===== Check Qdrant =====
  # Always use HTTP port (6333) for health checks, regardless of what QDRANT_PORT is set to
  log "INFO" "Waiting for Qdrant to be ready at http://${QDRANT_HOST}:6333..."
  start_time=$(date +%s)
  QDRANT_AVAILABLE=true
  until curl -s "http://${QDRANT_HOST}:6333/healthz" | grep -q "{\s*\"status\"\s*:\s*\"ok\"\s*}" > /dev/null 2>&1; do
    elapsed=$(($(date +%s) - start_time))
    log "WARN" "Qdrant is unavailable (waited ${elapsed}s) - sleeping for 2 seconds"
    sleep 2
    # If we've waited too long (1 minute), continue without Qdrant setup
    if [ $elapsed -gt 60 ]; then
      log "ERROR" "Timed out waiting for Qdrant after ${elapsed} seconds, continuing without Qdrant setup"
      QDRANT_AVAILABLE=false
      SKIP_QDRANT_SETUP=true
      break
    fi
  done
  
  if [ "$QDRANT_AVAILABLE" = true ]; then
    log "INFO" "Qdrant is up after $(($(date +%s) - start_time)) seconds"
  fi
  
  # Only check collections if Qdrant is available
  if [ "$QDRANT_AVAILABLE" = true ]; then
    # Check if collections already exist
    log "INFO" "Checking if collections already exist in Qdrant..."
    collection_response=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/list")
    log "DEBUG" "Collection list response: $collection_response"
    
    COLLECTIONS_EXIST=$(echo "$collection_response" | grep -q "sbert_books\|gmf_users\|gmf_books" && echo "true" || echo "false")
    
    if [ "$COLLECTIONS_EXIST" = "true" ]; then
      log "INFO" "Qdrant collections already exist, checking for data..."
      
      # Get detailed info for logging
      collections_found=$(echo "$collection_response" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')
      log "INFO" "Found collections: $collections_found"
      
      # Check if there's actual data in the collections
      sbert_response=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/sbert_books")
      SBERT_COUNT=$(echo "$sbert_response" | grep -o '"vectors_count":[0-9]*' | cut -d':' -f2)
      
      if [ -z "$SBERT_COUNT" ]; then
        log "WARN" "Could not determine vector count for sbert_books collection"
        SBERT_COUNT=0
      fi
      
      log "INFO" "SBERT books vectors count: $SBERT_COUNT"
      
      if [ "$SBERT_COUNT" -gt 0 ]; then
        log "INFO" "Found $SBERT_COUNT vectors in Qdrant collections. Embeddings already loaded, skipping setup."
        SKIP_QDRANT_SETUP=true
      else
        log "INFO" "Collections exist but no data found. Will set up Qdrant data."
        SKIP_QDRANT_SETUP=false
      fi
    else
      log "INFO" "No collections found. Will set up Qdrant collections and data."
      SKIP_QDRANT_SETUP=false
    fi
  else
    log "INFO" "Qdrant is unavailable. Skipping Qdrant setup."
    SKIP_QDRANT_SETUP=true
  fi
  
  # Set up Qdrant if needed and available
  if [ "$SKIP_QDRANT_SETUP" = false ] && [ "$QDRANT_AVAILABLE" = true ]; then
    log "INFO" "Setting up Qdrant embeddings..."
    
    # Source the functions from Qdrant entrypoint script using relative path
    source ./qdrant_entrypoint.sh
    
    # Set up Qdrant
    setup_qdrant
    
    log "INFO" "Qdrant setup completed"
  else
    log "INFO" "Skipping Qdrant setup - ${QDRANT_AVAILABLE:+Qdrant is $([ "$QDRANT_AVAILABLE" = true ] && echo "available" || echo "unavailable"), }setup is not needed or was skipped"
  fi
  
  log "INFO" "Database auto-configuration completed. Starting server..."
  log "INFO" "Running server on port: ${PORT:-8080}"
  exec ./server
fi
