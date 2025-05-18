package main

import (
	"context"
	"log"
	"os"
	"strconv"

	// Keep if you need it for other things, but UUID point ID is used below
	"github.com/jackc/pgx/v5"
	"github.com/joho/godotenv"

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
	
	log.Printf("Connecting to PostgreSQL database...")
	conn, err := pgx.Connect(context.Background(), postgresqlURL)
	if err != nil {
		log.Fatalf("FATAL: Error connecting to database: %v", err)
	}
	defer conn.Close(context.Background())

	// Get Qdrant configuration from environment
	qdrantHost := os.Getenv("QDRANT_HOST")
	if qdrantHost == "" {
		qdrantHost = "qdrant" // Default for Docker networking
	}
	
	qdrantPort := 6333 // Default Qdrant port
	if os.Getenv("QDRANT_PORT") != "" {
		// Convert port string to int
		if portVal, err := strconv.Atoi(os.Getenv("QDRANT_PORT")); err == nil {
			qdrantPort = portVal
		}
	}
	
	log.Printf("Connecting to Qdrant at %s:%d", qdrantHost, qdrantPort)
	qdrantClient, err := createQdrantRestClient(qdrantHost, qdrantPort)
	if err != nil {
		log.Fatalf("FATAL: Error creating Qdrant client: %v", err)
	}
	defer qdrantClient.Close()

	// Assuming db.New returns your sqlc generated queries struct
	queries := db.New(conn)

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
