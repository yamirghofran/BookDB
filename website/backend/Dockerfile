FROM golang:1.24-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git

# Set working directory
WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o server ./cmd/server

# Use a small alpine image for the final container
FROM alpine:latest

# Install ca-certificates for HTTPS requests and tools for database operations
RUN apk --no-cache add ca-certificates postgresql-client wget curl jq bash aws-cli

WORKDIR /root/

# Copy binary from builder stage
COPY --from=builder /app/server .
COPY --from=builder /app/sql /root/sql

# Create startup script to wait for PostgreSQL and apply migrations
RUN echo -e '#!/bin/bash\n\
set -e\n\
\n\
echo "Waiting for PostgreSQL to be ready..."\n\
until PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; do\n\
  echo "PostgreSQL is unavailable - sleeping for 2 seconds"\n\
  sleep 2\n\
done\n\
echo "PostgreSQL is up - proceeding with migrations"\n\
\n\
# Configure AWS CLI for Cloudflare R2\n\
mkdir -p ~/.aws\n\
echo "[default]\n\
aws_access_key_id=$R2_ACCESS_KEY_ID\n\
aws_secret_access_key=$R2_SECRET_ACCESS_KEY\n\
" > ~/.aws/credentials\n\
\n\
echo "Downloading database dump from Cloudflare R2"\n\
aws s3 --endpoint-url=$R2_ENDPOINT_URL cp s3://$R2_BUCKET_NAME/$R2_OBJECT_KEY /tmp/bookdb_dump.sql\n\
\n\
echo "Applying migrations"\n\
# Find and apply migration files\n\
for migration in /root/sql/migrations/*.sql; do\n\
  echo "Applying migration: $migration"\n\
  PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$migration"\n\
done\n\
\n\
echo "Importing database dump from R2"\n\
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/bookdb_dump.sql\n\
\n\
echo "Database setup complete, starting server"\n\
exec ./server\n\
' > /startup.sh && chmod +x /startup.sh\n\
\n\
# Health check 
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD wget -q --spider http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run the startup script instead of directly running the binary
CMD ["/bin/bash", "/startup.sh"]
