# BookDB Qdrant Server (Legacy Documentation)

> **NOTE**: This documentation is for the previous Node.js API architecture. The current implementation uses Nginx as a direct proxy to Qdrant. See `/nginx/README.md` for the current architecture.

The original architecture contained:
- Qdrant vector database
- Node.js (NPM) API proxy server
- Connection to existing Cloudflare Tunnel for bookdb.shadowlabs.cc

## Current Architecture

The current architecture has been simplified to:
- Qdrant vector database
- Nginx reverse proxy (replacing the Node.js API)
- Connection to existing Cloudflare Tunnel for bookdb.shadowlabs.cc

## Prerequisites

- Docker and Docker Compose installed
- Existing Cloudflare Tunnel and "cloudflared" Docker network
- Portainer for deployment (optional)

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bookdb.git
   cd bookdb
   ```

2. Environment Configuration:
   - Update the `.env` file in the root directory with your settings
   - Make sure to set `CLOUDFLARE_TUNNEL_TOKEN` to your actual token

2. Ensure the Cloudflare Tunnel is set up:
   - The tunnel should already be running and configured
   - Docker network "cloudflared" should exist

3. Build and start the containers:
   ```
   docker-compose up -d
   ```

4. Verify that the services are running:
   ```
   docker-compose ps
   ```

## Accessing the Services

- Qdrant API (container name): http://qdrant:6333 (from inside Docker network)
- Node.js API (container name): http://bookdb-api:3000 (from inside Docker network)
- Public URL: https://bookdb.shadowlabs.cc

## Testing the Setup

1. Test the API from within Docker network:
   ```
   docker exec -it bookdb-api curl http://localhost:3000/health
   ```

2. Test Qdrant through the API:
   ```
   docker exec -it bookdb-api curl http://localhost:3000/collections
   ```

3. Test the public URL:
   ```
   curl https://bookdb.shadowlabs.cc/health
   ```

## Deployment with Portainer

1. In Portainer, go to "Stacks" and click "Add stack"
2. Upload the `docker-compose.yaml` file
3. Set the environment variables or upload the `.env` file
4. Deploy the stack

## Customization

- Modify `api/server.js` to add custom endpoints or authentication
- Adjust volumes in `docker-compose.yaml` for persistent data storage
- Update Cloudflare Tunnel configuration as needed

## Troubleshooting

- Check container logs: `docker-compose logs`
- Verify Cloudflare Tunnel status: `cloudflared tunnel info <tunnel-name>`
- Ensure firewall allows required ports
