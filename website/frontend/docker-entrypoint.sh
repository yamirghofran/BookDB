#!/bin/sh

# Check if required environment variables are set
if [ -z "$BACKEND_SERVICE" ]; then
  echo "ERROR: BACKEND_SERVICE environment variable must be set"
  exit 1
fi

if [ -z "$QDRANT_SERVICE" ]; then
  echo "ERROR: QDRANT_SERVICE environment variable must be set"
  exit 1
fi

# Set default values only for optional environment variables
: "${VIRTUAL_HOST:=localhost}"
: "${VIRTUAL_PORT:=80}"
: "${APP_ENV:=production}"

echo "Starting frontend with following configuration:"
echo "BACKEND_SERVICE: ${BACKEND_SERVICE}"
echo "QDRANT_SERVICE: ${QDRANT_SERVICE}"
echo "VIRTUAL_HOST: ${VIRTUAL_HOST}"
echo "VIRTUAL_PORT: ${VIRTUAL_PORT}"
echo "APP_ENV: ${APP_ENV}"
echo "Note: SSL is handled by Cloudflare Tunnel"

# Replace variables in nginx.conf with environment values
envsubst '${BACKEND_SERVICE} ${QDRANT_SERVICE} ${VIRTUAL_HOST} ${VIRTUAL_PORT} ${APP_ENV}' < /etc/nginx/nginx.template > /etc/nginx/nginx.conf

# Start nginx
exec nginx -g 'daemon off;'
