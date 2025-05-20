package main

import (
	"context"
	"log"
	"os"
	"strconv"
	"time"

	// Replace pgx with pgxpool for connection pooling
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"github.com/qdrant/go-client/qdrant"

	"github.com/yamirghofran/BookDB/internal/api"
	"github.com/yamirghofran/BookDB/internal/db"
)

func main() {
	log.Println("Attempting to load .env file...")
	err := godotenv.Load()
	if err != nil {
		// Only treat "file not found" as a warning, other errors are fatal
		if !os.IsNotExist(err) {
			log.Fatalf("FATAL: Error loading .env file: %v", err)
		} else {
			log.Println("Warning: .env file not found. Relying on system environment variables.")
		}
	} else {
		log.Println(".env file loaded successfully.")
	}

	// Construct PostgreSQL connection string from environment variables
	pgHost := os.Getenv("DB_HOST")
	pgPort := os.Getenv("DB_PORT")
	pgUser := os.Getenv("DB_USER")
	pgPass := os.Getenv("DB_PASSWORD")
	pgDB := os.Getenv("DB_NAME")
	
	// Fallback to legacy env var if the new ones aren't set
	postgresqlURL := os.Getenv("POSTGRESQL_URL")
	if postgresqlURL == "" && (pgHost != "" && pgUser != "" && pgDB != "") {
		postgresqlURL = "postgres://" + pgUser
		if pgPass != "" {
			postgresqlURL += ":" + pgPass
		}
		postgresqlURL += "@" + pgHost
		if pgPort != "" {
			postgresqlURL += ":" + pgPort
		}
		postgresqlURL += "/" + pgDB
	}
	
	if postgresqlURL == "" {
		log.Fatalf("FATAL: No database connection details provided")
	}
		log.Printf("Creating PostgreSQL connection pool...")
	config, err := pgxpool.ParseConfig(postgresqlURL)
	if err != nil {
		log.Fatalf("FATAL: Invalid database connection string: %v", err)
	}
	
	// Set pool configuration - adjust these values based on your workload
	config.MaxConns = 10 // Set maximum number of connections
	
	// Create the connection pool
	dbPool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		log.Fatalf("FATAL: Error creating database connection pool: %v", err)
	}
	defer dbPool.Close()
	
	// Verify connection
	if err := dbPool.Ping(context.Background()); err != nil {
		log.Fatalf("FATAL: Could not ping database: %v", err)
	}
	log.Printf("Successfully connected to PostgreSQL database")

	// Get Qdrant configuration from environment
	qdrantHost := os.Getenv("QDRANT_HOST")
	if qdrantHost == "" {
		qdrantHost = "bookdb-qdrant" // Default for Docker networking
	}
	
	qdrantPort := 6333 // Default Qdrant port
	if os.Getenv("QDRANT_PORT") != "" {
		// Convert port string to int
		if portVal, err := strconv.Atoi(os.Getenv("QDRANT_PORT")); err == nil {
			qdrantPort = portVal
		}
	}
	log.Printf("Connecting to Qdrant at %s:%d", qdrantHost, qdrantPort)
	
	// Wait for Qdrant to be ready
	log.Printf("Waiting for Qdrant to be ready...")
	retries := 0
	maxRetries := 15
	connected := false
	var qdrantClient *qdrant.Client
	
	for retries < maxRetries {
		qdrantClient, err = createQdrantClient(qdrantHost, qdrantPort)
		if err != nil {
			log.Printf("Attempt %d: Error creating Qdrant client: %v", retries+1, err)
			retries++
			time.Sleep(2 * time.Second)
			continue
		}
		
		// Connection and health check are handled inside createQdrantClient now
		connected = true
		break
	}
	
	if !connected {
		log.Fatalf("FATAL: Failed to connect to Qdrant after %d attempts", maxRetries)
	}
	
	log.Printf("Successfully connected to Qdrant")
	defer qdrantClient.Close()
	// Create the DB queries instance using the connection pool
	queries := db.New(dbPool)

	// Create and start the API server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	server := api.NewServer(queries, qdrantClient)
	log.Printf("Starting server on :%s", port)
	if err := server.Run(":" + port); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}
