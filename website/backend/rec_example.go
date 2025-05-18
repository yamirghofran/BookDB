package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/qdrant/go-client/qdrant"
)

// initDatabase initializes and verifies the database connection
func initDatabase(ctx context.Context) (*pgxpool.Pool, error) {
	dbpool, err := pgxpool.New(ctx, os.Getenv("DATABASE_URL"))
	if err != nil {
		return nil, fmt.Errorf("unable to create connection pool: %v", err)
	}

	// Verify connection
	err = dbpool.Ping(ctx)
	if err != nil {
		dbpool.Close()
		return nil, fmt.Errorf("unable to ping database: %v", err)
	}

	return dbpool, nil
}

// initQdrant initializes the Qdrant client
func initQdrant() (*qdrant.Client, error) {
	qdrantClient, err := qdrant.NewClient(&qdrant.Config{
		Host: "localhost",
		Port: 6334,
	})
	if err != nil {
		return nil, fmt.Errorf("unable to create Qdrant client: %v", err)
	}
	return qdrantClient, nil
}

// getEmbeddingByPointID retrieves the embedding for a specific point ID
func getEmbeddingByPointID(ctx context.Context, qdrantClient *qdrant.Client, pointID int64, collectionName string) ([]float32, error) {
	points, err := qdrantClient.Get(ctx, &qdrant.GetPoints{
		CollectionName: collectionName,
		Ids: []*qdrant.PointId{
			qdrant.NewIDNum(uint64(pointID)),
		},
		WithVectors: qdrant.NewWithVectors(true),
	})

	if err != nil {
		return nil, fmt.Errorf("unable to get embedding for point %d: %v", pointID, err)
	}

	if len(points) == 0 {
		return nil, fmt.Errorf("no embedding found for point %d", pointID)
	}

	return points[0].Vectors.GetVector().Data, nil
}

// getPointByID retrieves a point by its ID
func getPointByID(ctx context.Context, qdrantClient *qdrant.Client, collection string, id interface{}) ([]*qdrant.RetrievedPoint, error) {
	var pointID *qdrant.PointId

	switch v := id.(type) {
	case int64:
		pointID = qdrant.NewIDNum(uint64(v))
	case string:
		pointID = qdrant.NewID(v)
	default:
		return nil, fmt.Errorf("unsupported ID type")
	}

	points, err := qdrantClient.Get(ctx, &qdrant.GetPoints{
		CollectionName: collection,
		Ids:            []*qdrant.PointId{pointID},
		WithVectors:    qdrant.NewWithVectors(true),
	})

	if err != nil {
		return nil, fmt.Errorf("unable to get points: %v", err)
	}

	return points, nil
}

// querySimilarByID finds points similar to the given ID
func querySimilarByID(ctx context.Context, qdrantClient *qdrant.Client, collection string, id int64, limit uint64) ([]*qdrant.ScoredPoint, error) {
	similarPoints, err := qdrantClient.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQueryID(qdrant.NewIDNum(uint64(id))),
		Limit:          &limit,
	})

	if err != nil {
		return nil, fmt.Errorf("unable to query Qdrant: %v", err)
	}

	return similarPoints, nil
}

// querySimilarByVector finds points similar to the given vector
func querySimilarByVector(ctx context.Context, qdrantClient *qdrant.Client, collection string, vector []float32, limit uint64) ([]*qdrant.ScoredPoint, error) {
	similarPoints, err := qdrantClient.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQueryDense(vector),
		Limit:          &limit,
	})

	if err != nil {
		return nil, fmt.Errorf("unable to query Qdrant: %v", err)
	}

	return similarPoints, nil
}

// getAverageEmbeddingsAndFindSimilar gets embeddings for multiple points, averages them, and finds similar points
func getAverageEmbeddingsAndFindSimilar(ctx context.Context, qdrantClient *qdrant.Client, pointIDs []int64, collectionName string, limit uint64) ([]*qdrant.ScoredPoint, error) {
	// Get embeddings for each point ID
	var allEmbeddings [][]float32

	for _, pointID := range pointIDs {
		// Get the embedding for this point
		embedding, err := getEmbeddingByPointID(ctx, qdrantClient, pointID, collectionName)
		if err != nil {
			fmt.Printf("Warning: %v\n", err)
			continue // Skip if no embedding found for this point
		}

		allEmbeddings = append(allEmbeddings, embedding)
	}

	if len(allEmbeddings) == 0 {
		return nil, fmt.Errorf("no embeddings found for the provided point IDs")
	}

	// Calculate the average embedding
	embeddingDim := len(allEmbeddings[0])
	avgEmbedding := make([]float32, embeddingDim)

	for _, embedding := range allEmbeddings {
		for i, val := range embedding {
			avgEmbedding[i] += val
		}
	}

	// Divide by the number of embeddings to get the average
	for i := range avgEmbedding {
		avgEmbedding[i] /= float32(len(allEmbeddings))
	}

	// Query for similar points using the average embedding
	return querySimilarByVector(ctx, qdrantClient, collectionName, avgEmbedding, limit)
}

func main() {
	ctx := context.Background()

	// Initialize database
	dbpool, err := initDatabase(ctx)
	if err != nil {
		log.Fatalf("%v\n", err)
	}
	defer dbpool.Close()
	fmt.Println("Successfully connected to database")

	// Initialize Qdrant
	qdrantClient, err := initQdrant()
	if err != nil {
		log.Fatalf("%v\n", err)
	}
	defer qdrantClient.Close()
	fmt.Println("Successfully connected to Qdrant")

	// Example 1: Get embedding by point ID
	pointID := int64(8698)
	_, err = getEmbeddingByPointID(ctx, qdrantClient, pointID, "gmf_book_embeddings")
	if err != nil {
		log.Printf("Error getting embedding: %v\n", err)
	} else {
		fmt.Printf("Successfully retrieved embedding for point %d\n", pointID)
	}

	// Example 2: Get point by ID
	_, err = getPointByID(ctx, qdrantClient, "sbert_embeddings", int64(1))
	if err != nil {
		log.Printf("Error getting point: %v\n", err)
	} else {
		fmt.Println("Successfully retrieved point")
	}

	// Example 3: Query similar by ID
	_, err = querySimilarByID(ctx, qdrantClient, "sbert_embeddings", 13, 5)
	if err != nil {
		log.Printf("Error querying similar points: %v\n", err)
	} else {
		fmt.Println("Successfully queried similar points")
	}

	// Example 4: Get user embedding and find similar books
	userPoints, err := getPointByID(ctx, qdrantClient, "gmf_user_embeddings", "00009ab2ed8cbfceda5a59da40966321")
	if err != nil {
		log.Printf("Error getting user points: %v\n", err)
	} else {
		userVector := userPoints[0].Vectors.GetVector().Data
		_, err := querySimilarByVector(ctx, qdrantClient, "gmf_book_embeddings", userVector, 5)
		if err != nil {
			log.Printf("Error finding similar books: %v\n", err)
		} else {
			fmt.Println("Successfully found similar books for user")
		}
	}

	// Example 5: Average multiple embeddings and find similar points
	pointIDs := []int64{18176747, 18050143, 69571}
	similarPoints, err := getAverageEmbeddingsAndFindSimilar(ctx, qdrantClient, pointIDs, "gmf_book_embeddings", 10)
	if err != nil {
		log.Printf("Error finding similar points to average: %v\n", err)
	} else {
		fmt.Println("Points similar to the average of provided points:")
		for i, point := range similarPoints {
			fmt.Printf("%d. ID: %v, Score: %f\n", i+1, point.Id, point.Score)
		}
	}
}
