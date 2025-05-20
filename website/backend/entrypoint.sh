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
  exec /bin/bash /root/postgres_entrypoint.sh
elif [ "${SERVICE_TYPE}" = "qdrant" ]; then
  log "INFO" "Starting Qdrant entrypoint script..."
  exec /bin/bash /root/qdrant_entrypoint.sh
else
  log "INFO" "SERVICE_TYPE not set or unrecognized. Starting server directly..."
  log "INFO" "Running server on port: ${PORT:-8080}"
  exec ./server
fi
