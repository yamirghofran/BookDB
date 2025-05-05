package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/jackc/pgx/v5"
	"github.com/joho/godotenv"

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

	conn, err := pgx.Connect(context.Background(), os.Getenv("DATABASE_URL"))
	if err != nil {
		log.Fatalf("FATAL: Error connecting to database: %v", err)
	}
	defer conn.Close(context.Background())

	queries := db.New(conn)

	books, err := queries.ListBooks(context.Background(), db.ListBooksParams{
		Limit:  10,
		Offset: 0,
	})
	if err != nil {
		log.Fatalf("Error listing books: %v", err)
	}
	for _, book := range books {
		data, _ := json.MarshalIndent(book, "", "  ")
		fmt.Println(string(data))
	}

}
