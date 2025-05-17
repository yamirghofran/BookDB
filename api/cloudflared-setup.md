# Instructions for setting up Cloudflare Tunnel

To set up your Cloudflare Tunnel for the BookDB service, follow these steps:

1. Install the `cloudflared` CLI tool on your machine:
   https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/

2. Log in to your Cloudflare account:
   ```
   cloudflared tunnel login
   ```

3. Create a new tunnel:
   ```
   cloudflared tunnel create bookdb
   ```

4. This will generate a tunnel token. Add this token to your root .env file:
   ```
   # In the root .env file
   CLOUDFLARE_TUNNEL_TOKEN=your_generated_token
   ```

5. Create DNS records for your domain:
   ```
   cloudflared tunnel route dns bookdb bookdb.shadowlabs.cc
   ```

6. Create a config file for your tunnel:
   ```
   vim ~/.cloudflared/config.yml
   ```

7. Add the following configuration:
   ```yaml
   tunnel: <YOUR_TUNNEL_ID>
   credentials-file: /root/.cloudflared/<TUNNEL_ID>.json
   
   ingress:
     - hostname: bookdb.shadowlabs.cc
       service: http://nginx:80
     - service: http_status:404
   ```

Note: Replace <YOUR_TUNNEL_ID> with the ID of your tunnel.

For more details, see the Cloudflare documentation:
https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
