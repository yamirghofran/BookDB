package main

import (
	"context"
	"log"
	"os"

	// Keep if you need it for other things, but UUID point ID is used below
	"github.com/jackc/pgx/v5"
	"github.com/joho/godotenv"
	"google.golang.org/protobuf/proto"

	"github.com/qdrant/go-client/qdrant"
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

	// --- Get a Book from PostgreSQL ---
	books, err := queries.ListBooks(context.Background(), db.ListBooksParams{
		Limit:  5,
		Offset: 0,
	})
	if err != nil {
		log.Fatalf("Error listing books from DB: %v", err)
	}
	if len(books) == 0 {
		log.Fatalf("No books found in PostgreSQL database to use as search target.")
	}

	targetBookID := int64(1) // Or get this dynamically

	filter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			{
				ConditionOneOf: &qdrant.Condition_Field{
					Field: &qdrant.FieldCondition{
						Key: "book_id", // The key of the field in your payload
						Match: &qdrant.Match{
							MatchValue: &qdrant.Match_Integer{
								Integer: targetBookID,
							},
						},
					},
				},
			},
		},
	}

	bookPoint, err := qdrantClient.Scroll(context.Background(), &qdrant.ScrollPoints{
		CollectionName: "sbert_embeddings",
		Filter:         filter,
		Limit:          proto.Uint32(1),
		WithVectors:    qdrant.NewWithVectorsEnable(true),
	})

	if err != nil {
		log.Fatalf("Error querying Qdrant: %v", err)
	}
	if len(bookPoint) > 0 {
		log.Println("Found points in Qdrant:")
		for _, point := range bookPoint { // Iterate over bookPoint directly
			log.Printf("  Point ID: %s", point.Id) // Access point properties
			log.Printf("  Payload: %+v", point.Payload)
			// log.Printf("  Vector: %v", point.Vectors) // Uncomment if you need the vector
		}
	} else {
		log.Println("No points found in Qdrant matching the filter.")
	}

}
