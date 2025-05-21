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
		fmt.Printf("Using configured gRPC port: %d\n", grpcPort)
	} else {
		// Default gRPC port is typically HTTP port + 1 (e.g., 6334 if HTTP is 6333)
		grpcPort = port + 1
		fmt.Printf("Using default gRPC port: %d\n", grpcPort)
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
	fmt.Printf("Connecting to Qdrant at %s:%d with API key: %v\n", host, grpcPort, apiKey != "")
	
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
		return nil, fmt.Errorf("failed to create Qdrant client: %v (host=%s, port=%d)", err, host, grpcPort)
	}

	// Test the connection with context
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	// First do a basic health check
	if _, err := client.HealthCheck(ctx); err != nil {
		return nil, fmt.Errorf("failed Qdrant health check: %v (host=%s, port=%d)", err, host, grpcPort)
	}
	
	// Now attempt to get info about the collections to verify full connectivity
	collections, err := client.ListCollections(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list Qdrant collections: %v (host=%s, port=%d)", err, host, grpcPort)
	}

	// Try to get a point from a collection if any exist
	if len(collections) > 0 {
		// Try with a known collection that should exist
		testCollections := []string{"sbert_books", "gmf_book_embeddings", "gmf_users"}
		var pointRetrieved bool

		for _, collection := range testCollections {
			// Check if this collection exists
			for _, existingColl := range collections {
				if existingColl == collection {
					// Try to get a point from this collection
					_, err = client.Get(ctx, &qdrant.GetPoints{
						CollectionName: collection,
						Ids:           []*qdrant.PointId{qdrant.NewIDNum(1)}, // Try to get point with ID 1
						WithVectors:   qdrant.NewWithVectors(true),
					})
					if err == nil {
						pointRetrieved = true
						fmt.Printf("Successfully verified point retrieval from collection %s\n", collection)
						break
					}
					fmt.Printf("Note: Could not fetch test point from collection %s: %v\n", collection, err)
				}
			}
			if pointRetrieved {
				break
			}
		}

		if !pointRetrieved {
			fmt.Printf("Warning: Could not verify point retrieval from any known collection, but server is responding\n")
		}
	} else {
		fmt.Printf("Warning: No collections found in Qdrant, but server is responding\n")
	}
	
	fmt.Printf("Successfully connected to Qdrant at %s:%d with %d collections\n", host, grpcPort, len(collections))
	return client, nil
}
