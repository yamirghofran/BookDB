FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm ci

# Copy all frontend source code
COPY . .

# Build the application
RUN npm run build

# Use nginx to serve the built application
FROM nginx:alpine

# Install envsubst for environment variable substitution
RUN apk add --no-cache bash gettext

# Copy the built app from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration as template
COPY nginx.conf /etc/nginx/nginx.template

# Copy entrypoint script
COPY docker-entrypoint.sh /

# Set proper permissions
RUN chmod -R 755 /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD curl -f http://localhost/health || exit 1

# Expose HTTP port
EXPOSE 80

# Set default service names
ENV BACKEND_SERVICE=backend QDRANT_SERVICE=qdrant

# Make the entrypoint script executable
RUN chmod +x /docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
