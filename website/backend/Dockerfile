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
RUN apk --no-cache add ca-certificates postgresql-client wget curl jq bash python3 py3-pip

WORKDIR /root/

# Copy binary from builder stage
COPY --from=builder /app/server .
COPY --from=builder /app/sql /root/sql

# Create scripts directory and copy necessary files
RUN mkdir -p /root/scripts

# Copy the entrypoint scripts and Python utilities
COPY entrypoint.sh /root/entrypoint.sh
COPY postgres_entrypoint.sh /root/postgres_entrypoint.sh
COPY qdrant_entrypoint.sh /root/qdrant_entrypoint.sh
COPY scripts/upload_embeddings.py /root/scripts/upload_embeddings.py
RUN chmod +x /root/entrypoint.sh /root/postgres_entrypoint.sh /root/qdrant_entrypoint.sh /root/scripts/upload_embeddings.py


# Expose port
EXPOSE 8080

# Run the entrypoint script instead of directly running the binary
CMD ["/bin/bash", "/root/entrypoint.sh"]
