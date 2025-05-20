#!/bin/bash
set -e

# Enhanced logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local level="$1"
    local message="$2"
    echo "[${timestamp}] [${level}] [QDRANT] ${message}"
}

log "INFO" "Starting Qdrant embedding uploader..."
log "INFO" "Environment: APP_ENV=${APP_ENV:-unknown}"
log "INFO" "Qdrant connection: ${QDRANT_HOST}:${QDRANT_PORT}"
log "INFO" "R2 Endpoint: ${R2_ENDPOINT_URL:-not set}"

# Wait for Qdrant to be ready
log "INFO" "Waiting for Qdrant to be ready at http://${QDRANT_HOST}:${QDRANT_PORT}..."
start_time=$(date +%s)
until curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/healthz" | grep -q "{\s*\"status\"\s*:\s*\"ok\"\s*}" > /dev/null 2>&1; do
  elapsed=$(($(date +%s) - start_time))
  log "WARN" "Qdrant is unavailable (waited ${elapsed}s) - sleeping for 2 seconds"
  sleep 2
  # Fail if we've waited too long (5 minutes)
  if [ $elapsed -gt 300 ]; then
    log "ERROR" "Timed out waiting for Qdrant after ${elapsed} seconds"
    exit 1
  fi
done
log "INFO" "Qdrant is up after $(($(date +%s) - start_time)) seconds - proceeding with embedding upload"

# Check if the collections already exist
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
    log "INFO" "Found $SBERT_COUNT vectors in Qdrant collections. Embeddings already loaded, skipping upload."
    SKIP_EMBEDDING_DOWNLOAD=true
  else
    log "INFO" "Collections exist but no data found. Will download and upload embeddings."
    SKIP_EMBEDDING_DOWNLOAD=false
  fi
else
  log "INFO" "No collections found, proceeding with embedding download and upload"
  SKIP_EMBEDDING_DOWNLOAD=false
fi

# Function for downloading files with authentication
download_file() {
  local source_url="$1"
  local dest_file="$2"
  local file_label="$3"
  
  download_start=$(date +%s)
  log "INFO" "Downloading $file_label from: $source_url"
  
  # Check if we have AWS CLI available for authenticated downloads
  if command -v aws &> /dev/null && [ -n "$R2_ACCESS_KEY_ID" ] && [ -n "$R2_SECRET_ACCESS_KEY" ]; then
    log "DEBUG" "Using AWS CLI for authenticated download of $file_label"
    
    # Extract bucket and key from the URL
    local bucket=$(echo "$source_url" | awk -F/ '{print $4}')
    local key=$(echo "$source_url" | cut -d/ -f5-)
    
    if AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
       aws --endpoint-url "${R2_ENDPOINT_URL_CLEANED}" s3 cp \
       "s3://${bucket}/${key}" "$dest_file"; then
      download_time=$(($(date +%s) - download_start))
      file_size=$(du -h "$dest_file" | cut -f1)
      log "INFO" "Successfully downloaded $file_label via AWS CLI (${file_size}, took ${download_time}s)"
      return 0
    else
      log "WARN" "AWS CLI download failed for $file_label, falling back to curl"
    fi
  fi
  
  # Fallback to curl
  # Generate current date in AWS format (required for authentication)
  if [ -n "$R2_ACCESS_KEY_ID" ] && [ -n "$R2_SECRET_ACCESS_KEY" ]; then
    log "DEBUG" "Using curl with AWS v4 signature authentication for $file_label"
    
    # Create temporary headers file
    HEADERS_FILE=$(mktemp)
    
    # Extract hostname from R2 endpoint
    local hostname=$(echo "$R2_ENDPOINT_URL_CLEANED" | sed 's|^https*://||' | sed 's|/.*$||')
    
    # Construct authentication headers using R2 credentials
    DATE=$(date -u +"%Y%m%dT%H%M%SZ")
    DATE_SHORT=$(date -u +"%Y%m%d")
    
    echo "Host: ${hostname}" > $HEADERS_FILE
    echo "X-Amz-Date: $DATE" >> $HEADERS_FILE
    echo "X-Amz-Content-Sha256: UNSIGNED-PAYLOAD" >> $HEADERS_FILE
    echo "Authorization: AWS4-HMAC-SHA256 Credential=${R2_ACCESS_KEY_ID}/${DATE_SHORT}/auto/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=dummy" >> $HEADERS_FILE
    
    if curl -L --retry 3 --retry-delay 2 -f -o "$dest_file" -H "@$HEADERS_FILE" "$source_url"; then
      download_time=$(($(date +%s) - download_start))
      file_size=$(du -h "$dest_file" | cut -f1)
      log "INFO" "Successfully downloaded $file_label with auth headers (${file_size}, took ${download_time}s)"
      rm -f $HEADERS_FILE
      return 0
    else
      log "WARN" "Authenticated download failed for $file_label with status: $?, trying unauthenticated"
      rm -f $HEADERS_FILE
    fi
  fi
  
  # Final attempt: unauthenticated
  log "DEBUG" "Using curl without authentication for $file_label"
  if curl -L --retry 3 --retry-delay 2 -f -o "$dest_file" "$source_url"; then
    download_time=$(($(date +%s) - download_start))
    file_size=$(du -h "$dest_file" | cut -f1)
    log "INFO" "Successfully downloaded $file_label (${file_size}, took ${download_time}s)"
    return 0
  else
    log "ERROR" "Failed to download $file_label (HTTP status: $?)"
    return 1
  fi
}

# Download embeddings if needed
if [ "$SKIP_EMBEDDING_DOWNLOAD" = false ]; then
  log "DEBUG" "R2 Environment Variables for embedding download:"
  log "DEBUG" "- R2_ENDPOINT_URL: '${R2_ENDPOINT_URL}'"
  log "DEBUG" "- R2_BUCKET_NAME: '${R2_BUCKET_NAME}'"
  log "DEBUG" "- R2_OBJECT_KEY_QDRANT: '${R2_OBJECT_KEY_QDRANT}'"
  log "DEBUG" "- R2_ACCESS_KEY_ID: '${R2_ACCESS_KEY_ID:0:5}***'"
  
  # Build the download URL from R2 variables
  if [ -z "$R2_ENDPOINT_URL" ] || [ -z "$R2_BUCKET_NAME" ] || [ -z "$R2_OBJECT_KEY_QDRANT" ]; then
    log "ERROR" "One or more R2 environment variables are not set. Skipping embedding download."
    log "ERROR" "Required: R2_ENDPOINT_URL, R2_BUCKET_NAME, R2_OBJECT_KEY_QDRANT"
    EMBEDDING_DOWNLOAD_SUCCESS=false
  else
    # Make sure endpoint URL has proper format
    # Remove trailing slash if present
    R2_ENDPOINT_URL_CLEANED=${R2_ENDPOINT_URL%/}
    # Add https:// prefix if needed
    if [[ "$R2_ENDPOINT_URL_CLEANED" != http* ]]; then
      R2_ENDPOINT_URL_CLEANED="https://$R2_ENDPOINT_URL_CLEANED"
    fi
    
    # Create embeddings directory
    mkdir -p /tmp/embeddings
    log "INFO" "Created embeddings directory: /tmp/embeddings"
    
    # Download SBERT book embeddings
    SBERT_OBJECT_KEY="${R2_OBJECT_KEY_QDRANT}/SBERT_embeddings.parquet"
    SBERT_URL="${R2_ENDPOINT_URL_CLEANED}/${R2_BUCKET_NAME}/${SBERT_OBJECT_KEY}"
    download_file "$SBERT_URL" "/tmp/embeddings/SBERT_embeddings.parquet" "SBERT embeddings"
    SBERT_SUCCESS=$?
    
    # Download GMF user embeddings
    GMF_USER_OBJECT_KEY="${R2_OBJECT_KEY_QDRANT}/gmf_user_embeddings.parquet"
    GMF_USER_URL="${R2_ENDPOINT_URL_CLEANED}/${R2_BUCKET_NAME}/${GMF_USER_OBJECT_KEY}"
    download_file "$GMF_USER_URL" "/tmp/embeddings/gmf_user_embeddings.parquet" "GMF user embeddings"
    GMF_USER_SUCCESS=$?
    
    # Download GMF book embeddings
    GMF_BOOK_OBJECT_KEY="${R2_OBJECT_KEY_QDRANT}/gmf_book_embeddings.parquet"
    GMF_BOOK_URL="${R2_ENDPOINT_URL_CLEANED}/${R2_BUCKET_NAME}/${GMF_BOOK_OBJECT_KEY}"
    download_file "$GMF_BOOK_URL" "/tmp/embeddings/gmf_book_embeddings.parquet" "GMF book embeddings"
    GMF_BOOK_SUCCESS=$?
    
    # Download ID mapping files
    USER_MAP_OBJECT_KEY="data/user_id_map_reduced.csv"
    USER_MAP_URL="${R2_ENDPOINT_URL_CLEANED}/${R2_BUCKET_NAME}/${USER_MAP_OBJECT_KEY}"
    
    download_file "$USER_MAP_URL" "/tmp/embeddings/user_id_map.csv" "user ID map"
    USER_MAP_SUCCESS=$?
    
    if [ $USER_MAP_SUCCESS -eq 0 ]; then
      row_count=$(wc -l < /tmp/embeddings/user_id_map.csv)
      log "INFO" "User ID map contains ${row_count} rows"
    else
      # Create an empty file as a fallback
      echo "user_id,new_userId,original_userId" > /tmp/embeddings/user_id_map.csv
      log "WARN" "Failed to download user ID map, created empty file as fallback"
    fi
    
    ITEM_MAP_OBJECT_KEY="data/item_id_map_reduced.csv"
    ITEM_MAP_URL="${R2_ENDPOINT_URL_CLEANED}/${R2_BUCKET_NAME}/${ITEM_MAP_OBJECT_KEY}"
    
    download_file "$ITEM_MAP_URL" "/tmp/embeddings/item_id_map.csv" "item ID map"
    ITEM_MAP_SUCCESS=$?
    
    if [ $ITEM_MAP_SUCCESS -eq 0 ]; then
      row_count=$(wc -l < /tmp/embeddings/item_id_map.csv)
      log "INFO" "Item ID map contains ${row_count} rows"
    else
      # Create an empty file as a fallback
      echo "item_id,new_itemId,original_itemId" > /tmp/embeddings/item_id_map.csv
      log "WARN" "Failed to download item ID map, created empty file as fallback"
    fi
    
    # Check if any embedding files downloaded successfully
    if [ "$SBERT_SUCCESS" -eq 0 ] || [ "$GMF_USER_SUCCESS" -eq 0 ] || [ "$GMF_BOOK_SUCCESS" -eq 0 ]; then
      log "INFO" "Download summary: SBERT=$([[ $SBERT_SUCCESS -eq 0 ]] && echo "Success" || echo "Failed"), GMF_USER=$([[ $GMF_USER_SUCCESS -eq 0 ]] && echo "Success" || echo "Failed"), GMF_BOOK=$([[ $GMF_BOOK_SUCCESS -eq 0 ]] && echo "Success" || echo "Failed")"
      EMBEDDING_DOWNLOAD_SUCCESS=true
    else
      log "ERROR" "No embedding files were successfully downloaded."
      EMBEDDING_DOWNLOAD_SUCCESS=false
    fi
  fi
else
  log "INFO" "Skipping embedding download - collections already populated"
  EMBEDDING_DOWNLOAD_SUCCESS=false
fi

# Upload embeddings to Qdrant
if [ "$EMBEDDING_DOWNLOAD_SUCCESS" = true ]; then
  log "INFO" "Installing Python dependencies for embedding upload..."
  pip_start=$(date +%s)
  
  # Use pip cache if available in development mode to speed up installation
  if [ "${APP_ENV:-production}" = "development" ]; then
    pip install pandas numpy qdrant-client dask[dataframe] pyarrow
  else
    pip install pandas numpy qdrant-client dask[dataframe] pyarrow --no-cache-dir
  fi
  
  pip_time=$(($(date +%s) - pip_start))
  log "INFO" "Python dependencies installed in ${pip_time}s"
  
  log "INFO" "Running Python script to upload embeddings to Qdrant..."
  script_start=$(date +%s)
  
  # Execute the external Python script with improved error handling
  python_output=$(mktemp)
  python /root/scripts/upload_embeddings.py \
    --qdrant-host ${QDRANT_HOST} \
    --qdrant-port ${QDRANT_PORT} \
    --embeddings-dir /tmp/embeddings 2>&1 | tee "$python_output"
  
  UPLOAD_RESULT=$?
  script_time=$(($(date +%s) - script_start))
  
  if [ $UPLOAD_RESULT -eq 0 ]; then
    log "INFO" "Embedding upload script executed successfully in ${script_time}s"
    
    # Parse the output to get counts of uploaded embeddings
    SBERT_COUNT=$(grep -o "Uploaded [0-9]* SBERT book embeddings" "$python_output" | awk '{print $2}')
    GMF_USER_COUNT=$(grep -o "Uploaded [0-9]* GMF user embeddings" "$python_output" | awk '{print $2}')
    GMF_BOOK_COUNT=$(grep -o "Uploaded [0-9]* GMF book embeddings" "$python_output" | awk '{print $2}')
    
    log "INFO" "Embedding upload counts - SBERT: ${SBERT_COUNT:-0}, GMF Users: ${GMF_USER_COUNT:-0}, GMF Books: ${GMF_BOOK_COUNT:-0}"
  else
    log "ERROR" "Embedding upload script failed with exit code $UPLOAD_RESULT after ${script_time}s"
    log "ERROR" "Last 10 lines of error output:"
    tail -n 10 "$python_output" | while read -r line; do
      log "ERROR" "  $line"
    done
  fi
  
  rm -f "$python_output"
  
  # Verify collections after upload
  collections_after=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/list")
  collections_count=$(echo "$collections_after" | grep -o '"name":"[^"]*"' | wc -l)
  log "INFO" "Found $collections_count collections after upload"
  
  # Cleanup temporary files
  log "INFO" "Cleaning up temporary files"
  rm -rf /tmp/embeddings
else
  log "WARN" "Skipping embedding upload - no files downloaded or collections already populated."
fi

log "INFO" "Qdrant embedding setup complete."
log "INFO" "Runtime summary:"
log "INFO" "- Collections existed: $COLLECTIONS_EXIST"
log "INFO" "- Initial vector count: ${SBERT_COUNT:-N/A}"
log "INFO" "- Files downloaded: $EMBEDDING_DOWNLOAD_SUCCESS"
log "INFO" "- Upload result: ${UPLOAD_RESULT:-skipped}"
if [ "$UPLOAD_RESULT" -eq 0 ]; then
  log "INFO" "- Vectors uploaded: SBERT=${SBERT_COUNT:-0}, GMF Users=${GMF_USER_COUNT:-0}, GMF Books=${GMF_BOOK_COUNT:-0}"
fi
