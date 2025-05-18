FROM nginx:alpine

# Copy nginx configuration
COPY ./nginx/nginx.conf /etc/nginx/nginx.conf

# Copy dashboard files
COPY ./nginx/html/ /usr/share/nginx/html/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD curl -f http://localhost/health || exit 1

# Expose HTTP port
EXPOSE 80
