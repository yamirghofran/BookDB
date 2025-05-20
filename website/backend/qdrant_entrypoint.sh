#!/bin/bash
set -e

# Enhanced logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local level="$1"
    local message="$2"
    echo "[${timestamp}] [${level}] [QDRANT] ${message}"
}

# Function to set up Qdrant - can be called directly when sourced
setup_qdrant() {
  log "INFO" "Setting up Qdrant embeddings..."
  
  # Wait for Qdrant to be ready
  log "INFO" "Waiting for Qdrant to be ready at http://${QDRANT_HOST}:${QDRANT_PORT:-6333}..."
  start_time=$(date +%s)
  QDRANT_AVAILABLE=true
  until response=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT:-6333}/healthz") && echo "$response" | grep -q "healthz check passed"; do
    log "DEBUG" "Health check response: '${response}'"
    elapsed=$(($(date +%s) - start_time))
    log "WARN" "Qdrant is unavailable (waited ${elapsed}s) - sleeping for 2 seconds"
    sleep 2
    if [ $elapsed -gt 300 ]; then
      log "ERROR" "Timed out waiting for Qdrant after ${elapsed} seconds"
      log "INFO" "Continuing without Qdrant setup..."
      QDRANT_AVAILABLE=false
      break
    fi
  done
  
  if [ "$QDRANT_AVAILABLE" = false ]; then
    log "WARN" "Qdrant is not available. Skipping embedding upload."
    return 1
  fi
  
  log "INFO" "Qdrant is up after $(($(date +%s) - start_time)) seconds - proceeding with embedding upload"
  
  # Check for existing collections
  log "INFO" "Checking for existing Qdrant collections..."
  collection_response=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT:-6333}/collections")
  log "DEBUG" "Collection list response: $collection_response"

  COLLECTIONS_EXIST=$(echo "$collection_response" | grep -q "sbert_books\|gmf_users\|gmf_book_embeddings" && echo "true" || echo "false")

  if [ "$COLLECTIONS_EXIST" = "true" ]; then
    collections_found=$(echo "$collection_response" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')
    log "INFO" "Found existing collections: $collections_found"
    
    sbert_response=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT:-6333}/collections/sbert_books")
    SBERT_COUNT=$(echo "$sbert_response" | grep -o '"vectors_count":[0-9]*' | cut -d':' -f2)
    
    if [ -z "$SBERT_COUNT" ]; then
      log "WARN" "Could not determine vector count for sbert_books collection"
      SBERT_COUNT=0
    fi
    
    log "INFO" "Current SBERT books vectors count: $SBERT_COUNT"
    
    log "INFO" "Deleting existing collections to ensure fresh data..."
    for collection in sbert_books gmf_users gmf_book_embeddings; do
      delete_response=$(curl -s -X DELETE "http://${QDRANT_HOST}:${QDRANT_PORT:-6333}/collections/${collection}")
      log "INFO" "Deleted collection ${collection}: $delete_response"
    done
  else
    log "INFO" "No collections found, will create new collections"
  fi
  
  # Always download and import embeddings
  log "INFO" "Always downloading and populating embeddings from R2..."
  
  # Download embedding files
  log "INFO" "Downloading embedding files..."
  mkdir -p /tmp/embeddings
  
  # Source URL for embeddings
  if [ -n "$R2_ENDPOINT_URL" ] && [ -n "$R2_BUCKET_NAME" ] && [ -n "$R2_OBJECT_KEY_QDRANT" ]; then
    # Download from R2
    log "INFO" "Downloading embeddings from R2..."
    R2_ENDPOINT_URL_CLEANED=${R2_ENDPOINT_URL%/}
    if [[ "$R2_ENDPOINT_URL_CLEANED" != http* ]]; then
      R2_ENDPOINT_URL_CLEANED="https://$R2_ENDPOINT_URL_CLEANED"
    fi
    
    # Download embedding files
    EMBEDDING_SUCCESS=true
    for file in "SBERT_embeddings.parquet" "gmf_user_embeddings.parquet" "gmf_book_embeddings.parquet"; do
      URL="${R2_ENDPOINT_URL_CLEANED}/${R2_OBJECT_KEY_QDRANT}${file}"
      log "INFO" "Downloading $file from $URL..."
      if ! curl -L --retry 3 --retry-delay 2 -f -o "/tmp/embeddings/${file}" "$URL"; then
        log "ERROR" "Failed to download $file"
        EMBEDDING_SUCCESS=false
      else
        log "INFO" "Successfully downloaded $file"
      fi
    done
  
  # Upload embeddings to Qdrant
  if [ "$EMBEDDING_SUCCESS" = true ]; then
    log "INFO" "Uploading embeddings to Qdrant..."
    
    # Install Python dependencies first
    log "INFO" "Installing Python dependencies..."
    pip install pandas numpy qdrant-client dask[dataframe] pyarrow --break-system-packages || {
      log "WARN" "Failed to install with system packages, trying with virtualenv..."
      python3 -m venv /tmp/venv
      . /tmp/venv/bin/activate
      pip install pandas numpy qdrant-client dask[dataframe] pyarrow
    }
    
    # Now run the Python script to upload embeddings
    log "INFO" "Running Python script to upload embeddings..."
    if python /root/scripts/upload_embeddings.py \
      --qdrant-host ${QDRANT_HOST} \
      --qdrant-port ${QDRANT_PORT:-6333} \
      --qdrant-grpc-port ${QDRANT_GRPC_PORT:-6334} \
      --use-grpc \
      --embeddings-dir /tmp/embeddings; then
      
      log "INFO" "Successfully uploaded embeddings to Qdrant"
    else
      log "WARN" "Error during upload with gRPC, retrying with HTTP..."
      if python /root/scripts/upload_embeddings.py \
        --qdrant-host ${QDRANT_HOST} \
        --qdrant-port ${QDRANT_PORT:-6333} \
        --embeddings-dir /tmp/embeddings \
        --use-grpc=false; then
        
        log "INFO" "Successfully uploaded embeddings to Qdrant using HTTP"
      else
        log "ERROR" "Failed to upload embeddings to Qdrant"
      fi
    fi
  fi
    
    # Cleanup
    rm -rf /tmp/embeddings
    
    # Verify the upload
    log "INFO" "Verifying collections after upload..."
    for collection in "sbert_books" "gmf_users" "gmf_book_embeddings"; do
      collection_info=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT:-6333}/collections/${collection}")
      if echo "$collection_info" | grep -q '"status":"ok"'; then
        vector_count=$(echo "$collection_info" | grep -o '"vectors_count":[0-9]*' | cut -d':' -f2)
        log "INFO" "Collection ${collection} contains ${vector_count} vectors"
      else
        log "ERROR" "Collection ${collection} not found after upload"
      fi
    done
  else
    log "ERROR" "Skipping embedding upload due to download failures"
  fi
  
  log "INFO" "Qdrant setup completed successfully"
}

# Main script execution - only runs if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  log "INFO" "Starting Qdrant embedding uploader..."
  log "INFO" "Environment: APP_ENV=${APP_ENV:-unknown}"
  log "INFO" "Qdrant connection: ${QDRANT_HOST}:${QDRANT_PORT} (HTTP) / ${QDRANT_HOST}:${QDRANT_GRPC_PORT:-6334} (gRPC)"
  log "INFO" "R2 Endpoint: ${R2_ENDPOINT_URL:-not set}"
  
  setup_qdrant
fi
