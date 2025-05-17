package main

import (
	"context"
	"log"
	"os"

	// Keep if you need it for other things, but UUID point ID is used below
	"github.com/jackc/pgx/v5"
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

	conn, err := pgx.Connect(context.Background(), os.Getenv("POSTGRESQL_URL"))
	if err != nil {
		log.Fatalf("FATAL: Error connecting to database: %v", err)
	}
	defer conn.Close(context.Background())

	qdrantClient, err := qdrant.NewClient(&qdrant.Config{
		Host: "localhost",
		Port: 6334,
	})
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
