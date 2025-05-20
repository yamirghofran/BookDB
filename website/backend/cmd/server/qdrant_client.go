package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// createQdrantClient creates a Qdrant client using gRPC communication
func createQdrantClient(host string, port int) (*qdrant.Client, error) {
	// Check if we should use the gRPC port instead of HTTP port
	grpcPort := port
	grpcPortEnv := os.Getenv("QDRANT_GRPC_PORT")
	if grpcPortEnv != "" {
		if p, err := strconv.Atoi(grpcPortEnv); err == nil {
			grpcPort = p
		}
	} else {
		// Default gRPC port is typically HTTP port + 1 (e.g., 6334 if HTTP is 6333)
		grpcPort = port + 1
	}
	
	// Configure gRPC options for better reliability
	grpcOptions := []grpc.DialOption{
		// Use insecure credentials for internal container communication
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		// Configure keepalive settings
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second, // Send pings every 10 seconds if there is no activity
			Timeout:             3 * time.Second,  // Wait 3 seconds for ping ACK before considering connection dead
			PermitWithoutStream: true,             // Allow pings even without active streams
		}),
	}
	
	// Check for API key
	apiKey := os.Getenv("QDRANT_API_KEY")
	
	// Add block option to grpcOptions for connection timeout
	grpcOptions = append(grpcOptions, grpc.WithBlock())

	// Create the client
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:        host,
		Port:        grpcPort, // Use gRPC port
		APIKey:      apiKey,
		GrpcOptions: grpcOptions,
	})
	
	if err != nil {
		return nil, fmt.Errorf("failed to create Qdrant client: %v", err)
	}

	// Test the connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	// Try a simple healthcheck operation
	if _, err := client.HealthCheck(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect to Qdrant: %v", err)
	}
	
	return client, nil
}
