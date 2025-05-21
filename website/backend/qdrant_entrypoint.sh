#!/bin/bash
set -e

# Enhanced logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local level="$1"
    local message="$2"
    echo "[${timestamp}] [${level}] [QDRANT] ${message}"
}

# Function to download from Cloudflare R2
download_from_r2() {
    local object_key="$1"
    local output_file="$2"
    local endpoint="$R2_ENDPOINT"
    local timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    local date_stamp=$(date -u +%Y%m%d)

    log "INFO" "Downloading ${object_key} from R2..."

    # Create canonical request headers
    local host=$(echo "$endpoint" | sed 's|https://||')
    local headers="host:${host}
x-amz-content-sha256:UNSIGNED-PAYLOAD
x-amz-date:${timestamp}"

    # Create string to sign
    local string_to_sign="AWS4-HMAC-SHA256
${timestamp}
${date_stamp}/auto/s3/aws4_request
$(echo -en "GET\n/${object_key}\n\n${headers}\n\nUNSIGNED-PAYLOAD" | openssl dgst -sha256 -hex | sed 's/^.* //')"

    # Calculate signature
    local k_date=$(echo -n "${date_stamp}" | openssl sha256 -hmac "AWS4${AWS_SECRET_ACCESS_KEY}" -hex | sed 's/^.* //')
    local k_region=$(echo -n "auto" | openssl sha256 -hmac "${k_date}" -hex | sed 's/^.* //')
    local k_service=$(echo -n "s3" | openssl sha256 -hmac "${k_region}" -hex | sed 's/^.* //')
    local k_signing=$(echo -n "aws4_request" | openssl sha256 -hmac "${k_service}" -hex | sed 's/^.* //')
    local signature=$(echo -n "${string_to_sign}" | openssl sha256 -hmac "${k_signing}" -hex | sed 's/^.* //')

    # Make the request
    curl -s -o "${output_file}" \
        -H "Host: ${host}" \
        -H "X-Amz-Date: ${timestamp}" \
        -H "X-Amz-Content-Sha256: UNSIGNED-PAYLOAD" \
        -H "Authorization: AWS4-HMAC-SHA256 Credential=${AWS_ACCESS_KEY_ID}/${date_stamp}/auto/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=${signature}" \
        "${endpoint}/${object_key}"

    if [ $? -eq 0 ] && [ -s "${output_file}" ]; then
        log "INFO" "Successfully downloaded ${object_key}"
        return 0
    else
        log "ERROR" "Failed to download ${object_key}"
        return 1
    fi
}

# Download ID mapping files before starting Qdrant
download_id_maps() {
    log "INFO" "Starting download of ID mapping files..."
    
    mkdir -p /data/id_maps
    
    # Download item (book) ID mapping
    download_from_r2 "data/item_id_map_reduced.csv" "/data/id_maps/item_id_map.csv"
    if [ $? -ne 0 ]; then
        log "ERROR" "Failed to download item ID mapping"
        return 1
    fi
    
    # Download user ID mapping
    download_from_r2 "data/user_id_map_reduced.csv" "/data/id_maps/user_id_map.csv"
    if [ $? -ne 0 ]; then
        log "ERROR" "Failed to download user ID mapping"
        return 1
    fi
    
    log "INFO" "Successfully downloaded all ID mapping files"
    return 0
}

# Function to set up Qdrant - can be called directly when sourced
setup_qdrant() {
    log "INFO" "Setting up Qdrant embeddings..."
    
    # First download the ID mapping files
    download_id_maps
    if [ $? -ne 0 ]; then
        log "WARN" "Failed to download ID mappings, but continuing with Qdrant setup..."
    fi
    
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
          --use-grpc true \
          --embeddings-dir /tmp/embeddings; then
          
          log "INFO" "Successfully uploaded embeddings to Qdrant"
        else
          log "WARN" "Error during upload with gRPC, retrying with HTTP..."
          if python /root/scripts/upload_embeddings.py \
            --qdrant-host ${QDRANT_HOST} \
            --qdrant-port ${QDRANT_PORT:-6333} \
            --embeddings-dir /tmp/embeddings \
            --use-grpc false; then
            
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
        
        # Check all available collections first
        all_collections=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT:-6333}/collections")
        log "DEBUG" "All available collections: $all_collections"
        
        for collection in "sbert_books" "gmf_users" "gmf_book_embeddings" "connection_test"; do
            log "DEBUG" "Checking collection: ${collection}"
            collection_info=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT:-6333}/collections/${collection}")
            log "DEBUG" "Collection response: $collection_info"
            
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
