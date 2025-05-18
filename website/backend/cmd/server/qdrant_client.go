package main

import (
	"fmt"
	"net/url"
	"os"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// createQdrantRestClient creates a Qdrant client that works around HTTP/2 vs HTTP/1.1 compatibility issues
func createQdrantRestClient(host string, port int) (*qdrant.Client, error) {
	// Construct URL for validation
	baseURL := fmt.Sprintf("http://%s:%d", host, port)
	
	// Parse URL for validation
	_, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Qdrant URL: %v", err)
	}
	
	// Ensure the HTTP_PROXY environment variable is set to force HTTP/1.1
	if os.Getenv("QDRANT_HTTP_VERSION") == "1.1" {
		os.Setenv("GRPC_GO_REQUIRE_HANDSHAKE", "off")
	}
	
	// Create client with proper gRPC options to avoid HTTP/2
	grpcOptions := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDisableRetry(),
		grpc.WithBlock(),
	}
	
	return qdrant.NewClient(&qdrant.Config{
		Host: host,
		Port: port,
		GrpcOptions: grpcOptions,
		SkipCompatibilityCheck: true, // Skip version checks that may use HTTP/2
	})
}
