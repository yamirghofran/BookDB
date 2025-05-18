Basic QDrant Database behind Nginx Reverse Proxy
[]: # 
[]: # ## Nginx Configuration
[]: # 
[]: # - The Nginx configuration file is located at `nginx/nginx.conf`
[]: # - It handles incoming requests and forwards them to the appropriate service
[]: # - Make sure to update the configuration if you change any service names or ports
[]: # 
[]: # ## Troubleshooting
[]: # 
[]: # - Check logs for any errors:
[]: #    ```
[]: #    docker-compose logs nginx
[]: #    docker-compose logs bookdb-api
[]: #    docker-compose logs qdrant
[]: #    ```