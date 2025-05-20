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
  log "INFO" "Checking PostgreSQL connectivity at ${DB_HOST}:${DB_PORT}..."
  # Make sure all variables are properly defined before using them
  if [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ] || [ -z "$DB_USER" ] || [ -z "$DB_NAME" ]; then
    log "ERROR" "Missing required database connection parameters"
    log "ERROR" "DB_HOST=${DB_HOST:-not set}, DB_PORT=${DB_PORT:-not set}, DB_USER=${DB_USER:-not set}, DB_NAME=${DB_NAME:-not set}"
    log "WARN" "Continuing without PostgreSQL connection"
    POSTGRES_AVAILABLE=false
  else
    # Try to connect to PostgreSQL with timeout (5 attempts)
    POSTGRES_AVAILABLE=false
    for i in {1..5}; do
      if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" > /dev/null 2>&1; then
        log "INFO" "PostgreSQL is available"
        POSTGRES_AVAILABLE=true
        break
      fi
      log "WARN" "PostgreSQL is unavailable - attempt ${i}/5"
      sleep 2
    done
    
    if [ "$POSTGRES_AVAILABLE" = false ]; then
      log "WARN" "Could not connect to PostgreSQL after 5 attempts, continuing without database connection"
    fi
  fi
  
  # ===== Check Qdrant =====
  log "INFO" "Checking Qdrant connectivity at http://${QDRANT_HOST}:6333..."
  # Always use HTTP port (6333) for health checks
  QDRANT_AVAILABLE=false
  
  # Simple check for connectivity with a timeout
  for i in {1..5}; do
    RESPONSE=$(curl -s --max-time 5 "http://${QDRANT_HOST}:6333/healthz")
    if echo "$RESPONSE" | grep -q "healthz check passed"; then
      log "INFO" "Qdrant is available"
      QDRANT_AVAILABLE=true
      break
    fi
    log "WARN" "Qdrant is unavailable - attempt ${i}/5"
    sleep 2
  done
  
  if [ "$QDRANT_AVAILABLE" = false ]; then
    log "WARN" "Could not connect to Qdrant after 5 attempts, continuing without Qdrant connection"
  fi
  
  # ===== Setup services =====
  # Setup PostgreSQL if available
  if [ "$POSTGRES_AVAILABLE" = true ]; then
    log "INFO" "Setting up PostgreSQL database..."
    source ./postgres_entrypoint.sh
    setup_database
    log "INFO" "PostgreSQL database setup completed"
  fi
  
  # Setup Qdrant if available
  if [ "$QDRANT_AVAILABLE" = true ]; then
    log "INFO" "Setting up Qdrant embeddings..."
    source ./qdrant_entrypoint.sh
    setup_qdrant
    log "INFO" "Qdrant setup completed"
  fi
  
  log "INFO" "Service auto-configuration completed. Starting server..."
  log "INFO" "Running server on port: ${PORT:-8080}"
  exec ./server
fi
