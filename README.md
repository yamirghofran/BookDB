# BookDB Qdrant Server

This repository contains a Docker Compose setup for running:
- Qdrant vector database
- Node.js (NPM) API proxy server
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

2. **For local testing:** 
   - See `LOCAL_TESTING.md` for instructions to test on Docker Desktop
   - Run `docker-compose -f docker-compose.local.yaml up -d` to start local environment

3. Ensure the Cloudflare Tunnel is set up:
   - The tunnel should already be running and configured
   - Docker network "cloudflared" should exist

4. Build and start the containers:
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
