package handlers

import (
	"context"
	"fmt"
	"net/http"
	"strconv" // Added for string to int64 conversion

	"github.com/gin-gonic/gin"
	"github.com/google/uuid" // Added for UUID parsing
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/qdrant/go-client/qdrant"
	"github.com/yamirghofran/BookDB/internal/db" // Added for DB queries
)

type RecommendationService struct {
	QdrantClient qdrant.Client
	DB           *db.Queries // Added DB to RecommendationService if it were to be used standalone
}

// Helper function to get embedding by Goodreads ID from Qdrant
func getEmbeddingByGoodreadsID(ctx context.Context, qdrantClient *qdrant.Client, goodreadsID int64, collectionName string) ([]float32, error) {
	points, err := qdrantClient.Get(ctx, &qdrant.GetPoints{
		CollectionName: collectionName,
		Ids: []*qdrant.PointId{
			qdrant.NewIDNum(uint64(goodreadsID)),
		},
		WithVectors: qdrant.NewWithVectors(true),
	})

	if err != nil {
		return nil, fmt.Errorf("qdrant client.Get: unable to get embedding for Goodreads ID %d: %v", goodreadsID, err)
	}

	if len(points) == 0 {
		return nil, fmt.Errorf("no embedding found for Goodreads ID %d in collection %s", goodreadsID, collectionName)
	}

	if points[0].Vectors == nil || points[0].Vectors.GetVector() == nil {
		return nil, fmt.Errorf("embedding vector is nil for Goodreads ID %d in collection %s", goodreadsID, collectionName)
	}

	return points[0].Vectors.GetVector().Data, nil
}

// Helper function to get embedding by User ID (string) from Qdrant
func getEmbeddingByUserID(ctx context.Context, qdrantClient *qdrant.Client, userID string, collectionName string) ([]float32, error) {
	points, err := qdrantClient.Get(ctx, &qdrant.GetPoints{
		CollectionName: collectionName,
		Ids: []*qdrant.PointId{
			qdrant.NewID(userID), // Use NewID for string IDs
		},
		WithVectors: qdrant.NewWithVectors(true),
	})

	if err != nil {
		return nil, fmt.Errorf("qdrant client.Get: unable to get embedding for User ID %s: %v", userID, err)
	}
	if len(points) == 0 {
		return nil, fmt.Errorf("no embedding found for User ID %s in collection %s", userID, collectionName)
	}
	if points[0].Vectors == nil || points[0].Vectors.GetVector() == nil {
		return nil, fmt.Errorf("embedding vector is nil for User ID %s in collection %s", userID, collectionName)
	}
	return points[0].Vectors.GetVector().Data, nil
}

// Helper function to query similar points by vector, excluding certain Goodreads IDs (int64)
func querySimilarByVectorWithExclusion(
	ctx context.Context,
	qdrantClient *qdrant.Client,
	collectionName string,
	vector []float32,
	limit uint64,
	excludeIDs []int64,
) ([]*qdrant.ScoredPoint, error) {
	mustNotConditions := make([]*qdrant.Condition, 0, len(excludeIDs))
	for _, id := range excludeIDs {
		// Assuming 'goodreads_id' is a payload field we can filter on.
		// If not, this filter won't work as intended for ID exclusion.
		// The qdrant.NewHasIDCondition was not found, so trying with a field condition.
		// This assumes the ID itself is stored as a filterable payload field.
		// Based on rec_example.go, NewMatchInt seems to be the correct function.
		// This assumes 'goodreads_id' is a payload field for filtering.
		condition := qdrant.NewMatchInt("goodreads_id", id)
		mustNotConditions = append(mustNotConditions, condition)
	}

	filter := &qdrant.Filter{
		MustNot: mustNotConditions,
	}

	queryPoints := &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query:          qdrant.NewQueryDense(vector),
		Limit:          &limit,
		Filter:         filter,
		WithPayload:    qdrant.NewWithPayload(true), // Request payload to get GoodreadsID if stored there
	}

	similarPoints, err := qdrantClient.Query(ctx, queryPoints)
	if err != nil {
		return nil, fmt.Errorf("qdrant client.Query: unable to query Qdrant: %v", err)
	}
	return similarPoints, nil
}

// Helper function to query similar users by vector, excluding a single UserID (string)
func querySimilarUsersByVector(
	ctx context.Context,
	qdrantClient *qdrant.Client,
	collectionName string,
	vector []float32,
	limit uint64,
	excludeUserID string, // String ID for users
) ([]*qdrant.ScoredPoint, error) {
	var mustNotConditions []*qdrant.Condition
	if excludeUserID != "" {
		// Qdrant string IDs are typically not filterable via NewMatchInt.
		// We need to filter on a payload field if the ID itself is stored there,
		// or rely on the client to filter post-fetch if direct ID exclusion isn't easy.
		// For now, assuming Qdrant point ID for users is the string UUID.
		// The primary way to exclude specific point IDs is by NOT including them in a Get operation,
		// or if the search operation itself has an `exclude: [PointId]` field.
		// The current QueryPoints does not seem to have a direct `excludeIds` field.
		// We will filter the results after fetching if excludeUserID matches.
		// Alternatively, if user UUIDs are also stored as a filterable payload field (e.g., "user_uuid_payload"),
		// we could use qdrant.NewMatchText("user_uuid_payload", excludeUserID).
		// For simplicity, we'll fetch N+1 and filter in code if needed, or fetch N and hope the excluded one isn't top.
		// A robust way is to ensure the point ID itself is filterable or use a payload field.
		// Let's assume for now the `id` field in Qdrant for users is the string UUID and we can use a MatchText condition
		// if we also store this UUID as a payload field named "user_id_payload" or similar.
		// If not, this filter might not be effective.
		// For "gmf_user_embeddings", the ID is a string like "00009ab2...".
		// We'll assume there's a payload field "id_str" that stores this string ID for filtering.
		// This is a common pattern if direct ID filtering in search is tricky.
		// If "id_str" is not a payload field, this filter will not work.
		condition := qdrant.NewMatchText("id_str", excludeUserID) // Corrected to NewMatchText
		mustNotConditions = append(mustNotConditions, condition)
	}

	filter := &qdrant.Filter{}
	if len(mustNotConditions) > 0 {
		filter.MustNot = mustNotConditions
	}

	queryPoints := &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query:          qdrant.NewQueryDense(vector),
		Limit:          &limit,
		Filter:         filter,
		WithPayload:    qdrant.NewWithPayload(true), // Request payload to get user UUID if stored there
	}

	similarPoints, err := qdrantClient.Query(ctx, queryPoints)
	if err != nil {
		return nil, fmt.Errorf("qdrant client.Query (users): unable to query Qdrant: %v", err)
	}
	return similarPoints, nil
}

// getAverageEmbeddingsAndFindSimilarWithExclusion gets embeddings for multiple Goodreads IDs,
// averages them, and finds similar points, excluding the original IDs.
func getAverageEmbeddingsAndFindSimilarWithExclusion(
	ctx context.Context,
	qdrantClient *qdrant.Client,
	goodreadsIDs []int64,
	collectionName string,
	limit uint64,
) ([]*qdrant.ScoredPoint, error) {
	if len(goodreadsIDs) == 0 {
		return []*qdrant.ScoredPoint{}, nil // Return empty if no IDs provided
	}

	var allEmbeddings [][]float32
	for _, gid := range goodreadsIDs {
		embedding, err := getEmbeddingByGoodreadsID(ctx, qdrantClient, gid, collectionName)
		if err != nil {
			fmt.Printf("Warning: could not get embedding for Goodreads ID %d: %v\n", gid, err)
			continue // Skip if no embedding found for this point
		}
		allEmbeddings = append(allEmbeddings, embedding)
	}

	if len(allEmbeddings) == 0 {
		return nil, fmt.Errorf("no embeddings found for the provided Goodreads IDs")
	}

	embeddingDim := len(allEmbeddings[0])
	avgEmbedding := make([]float32, embeddingDim)

	for _, embedding := range allEmbeddings {
		if len(embedding) != embeddingDim {
			return nil, fmt.Errorf("inconsistent embedding dimensions")
		}
		for i, val := range embedding {
			avgEmbedding[i] += val
		}
	}

	for i := range avgEmbedding {
		avgEmbedding[i] /= float32(len(allEmbeddings))
	}

	return querySimilarByVectorWithExclusion(ctx, qdrantClient, collectionName, avgEmbedding, limit, goodreadsIDs)
}

// GetAnonymousRecommendations returns book recommendations based on provided book UUIDs
func (h *Handler) GetAnonymousRecommendations(c *gin.Context) {
	var req struct {
		LikedBookIds []string `json:"likedBookIds" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload: " + err.Error()})
		return
	}

	if len(req.LikedBookIds) == 0 {
		c.JSON(http.StatusOK, []BookResponse{}) // Return empty if no liked books
		return
	}

	ctx := c.Request.Context()
	inputGoodreadsIDs := make([]int64, 0, len(req.LikedBookIds))
	inputBookUUIDs := make([]pgtype.UUID, 0, len(req.LikedBookIds))

	for _, idStr := range req.LikedBookIds {
		bookUUID, err := uuid.Parse(idStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Invalid book UUID format: %s", idStr)})
			return
		}
		pgUUID := pgtype.UUID{Bytes: bookUUID, Valid: true}
		inputBookUUIDs = append(inputBookUUIDs, pgUUID)

		// Fetch GoodreadsID for each book UUID
		// Assuming GetBookByID returns a struct with GoodreadsID field.
		// We need to fetch the book to get its GoodreadsID.
		// A batch fetch might be more efficient if you have many LikedBookIds.
		// For simplicity, fetching one by one here.
		book, err := h.DB.GetBookByID(ctx, pgUUID) // GetBookByID returns db.GetBookByIDRow
		if err != nil {
			fmt.Printf("Warning: Could not fetch book details for UUID %s: %v\n", idStr, err)
			// Decide if this should be a hard error or just skip this book
			continue
		}
		inputGoodreadsIDs = append(inputGoodreadsIDs, book.GoodreadsID)
	}

	if len(inputGoodreadsIDs) == 0 {
		c.JSON(http.StatusOK, []BookResponse{}) // Return empty if no valid Goodreads IDs found
		return
	}

	recommendationLimit := uint64(10) // Number of recommendations to return
	qdrantCollection := "gmf_book_embeddings"

	similarPoints, err := getAverageEmbeddingsAndFindSimilarWithExclusion(ctx, h.QdrantClient, inputGoodreadsIDs, qdrantCollection, recommendationLimit)
	if err != nil {
		fmt.Printf("Error getting average embeddings and finding similar: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to calculate recommendations"})
		return
	}

	if len(similarPoints) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	recommendedGoodreadsIDs := make([]int64, 0, len(similarPoints))
	for _, point := range similarPoints {
		if point.Id != nil {
			// Assuming point.Id.GetNum() gives the Goodreads ID (uint64)
			recommendedGoodreadsIDs = append(recommendedGoodreadsIDs, int64(point.Id.GetNum()))
		} else if point.Payload != nil {
			// Fallback: if GoodreadsID is in payload (e.g. as "goodreads_id" or "book_id")
			// This depends on how Qdrant points are structured.
			// For rec_example.go, ID itself is the Goodreads ID.
			// Example: if payload has "goodreads_id": val,
			// if goidVal, ok := point.Payload["goodreads_id"]; ok {
			// if goid, ok := goidVal.GetIntegerValue(); ok {
			// recommendedGoodreadsIDs = append(recommendedGoodreadsIDs, goid)
			// }
			// }
		}
	}

	// Filter out any of the original inputGoodreadsIDs from the recommendations
	finalRecommendedGoodreadsIDs := make([]int64, 0, len(recommendedGoodreadsIDs))
	inputIDSet := make(map[int64]struct{})
	for _, id := range inputGoodreadsIDs {
		inputIDSet[id] = struct{}{}
	}
	for _, recID := range recommendedGoodreadsIDs {
		if _, exists := inputIDSet[recID]; !exists {
			finalRecommendedGoodreadsIDs = append(finalRecommendedGoodreadsIDs, recID)
		}
	}
	recommendedGoodreadsIDs = finalRecommendedGoodreadsIDs // Update with filtered list

	if len(recommendedGoodreadsIDs) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	// Fetch full book details for the recommended Goodreads IDs
	dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, recommendedGoodreadsIDs)
	if err != nil {
		fmt.Printf("Error fetching recommended book details from DB: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
		return
	}

	responseBooks := make([]BookResponse, 0, len(dbBooks))
	seenTitles := make(map[string]struct{}) // To track unique titles

	for _, bookRow := range dbBooks {
		if _, exists := seenTitles[bookRow.Title]; exists {
			continue // Skip if title already processed
		}
		seenTitles[bookRow.Title] = struct{}{}

		var apiAvgRating *float64
		if bookRow.AverageRating.Valid {
			float8Val, convErr := bookRow.AverageRating.Float64Value()
			if convErr == nil && float8Val.Valid {
				apiAvgRating = &float8Val.Float64
			} else if convErr != nil {
				fmt.Printf("Error converting AverageRating for book ID %v: %v\n", bookRow.ID, convErr)
			}
		}

		responseBooks = append(responseBooks, BookResponse{
			ID:              bookRow.ID,
			GoodreadsID:     bookRow.GoodreadsID,
			GoodreadsUrl:    bookRow.GoodreadsUrl,
			Title:           bookRow.Title,
			Description:     bookRow.Description,
			PublicationYear: bookRow.PublicationYear,
			CoverImageUrl:   bookRow.CoverImageUrl,
			AverageRating:   apiAvgRating,
			RatingsCount:    bookRow.RatingsCount,
			Authors:         processStringArrayInterface(bookRow.Authors), // from basic.go
			Genres:          processStringArrayInterface(bookRow.Genres),  // from basic.go
			CreatedAt:       bookRow.CreatedAt,
			UpdatedAt:       bookRow.UpdatedAt,
			// Reviews and ReviewsTotalCount are not typically part of a recommendation list item
		})
	}

	c.JSON(http.StatusOK, responseBooks)
}

// GetContentBasedRecommendations returns content-based recommendations for a single book (by its UUID)
func (h *Handler) GetContentBasedRecommendations(c *gin.Context) { // Changed receiver to *Handler
	bookUUIDStr := c.Param("id") // Path param from /recommendations/books/:id/similar
	ctx := c.Request.Context()

	bookUUID, err := uuid.Parse(bookUUIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book UUID format"})
		return
	}
	pgBookUUID := pgtype.UUID{Bytes: bookUUID, Valid: true}

	// 1. Get the Goodreads ID of the input book
	sourceBook, err := h.DB.GetBookByID(ctx, pgBookUUID) // db.GetBookByIDRow
	if err != nil {
		fmt.Printf("Error fetching source book %s for similar: %v\n", bookUUIDStr, err)
		c.JSON(http.StatusNotFound, gin.H{"error": "Source book not found"})
		return
	}
	sourceGoodreadsID := sourceBook.GoodreadsID

	// 2. Get the embedding for the source book's Goodreads ID
	// Using "gmf_book_embeddings" as per rec_example for averaging, implies it's a good general purpose embedding.
	// Or use "sbert_embeddings" if that's preferred for content similarity. Let's use "gmf_book_embeddings" for consistency.
	qdrantCollection := "gmf_book_embeddings" // Or "sbert_embeddings"
	sourceEmbedding, err := getEmbeddingByGoodreadsID(ctx, h.QdrantClient, sourceGoodreadsID, qdrantCollection)
	if err != nil {
		fmt.Printf("Error getting embedding for source book GoodreadsID %d: %v\n", sourceGoodreadsID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not retrieve embedding for source book"})
		return
	}

	// 3. Query Qdrant for similar books, excluding the source book itself
	recommendationLimit := uint64(10)
	similarPoints, err := querySimilarByVectorWithExclusion(ctx, h.QdrantClient, qdrantCollection, sourceEmbedding, recommendationLimit, []int64{sourceGoodreadsID})
	if err != nil {
		fmt.Printf("Error querying Qdrant for similar books to %d: %v\n", sourceGoodreadsID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to find similar books"})
		return
	}

	if len(similarPoints) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	recommendedGoodreadsIDs := make([]int64, 0, len(similarPoints))
	for _, point := range similarPoints {
		if point.Id != nil {
			recommendedGoodreadsIDs = append(recommendedGoodreadsIDs, int64(point.Id.GetNum()))
		}
	}

	// Filter out the sourceGoodreadsID from the recommendations, if present
	finalRecommendedGoodreadsIDsCB := make([]int64, 0, len(recommendedGoodreadsIDs))
	for _, recID := range recommendedGoodreadsIDs {
		if recID != sourceGoodreadsID {
			finalRecommendedGoodreadsIDsCB = append(finalRecommendedGoodreadsIDsCB, recID)
		}
	}
	recommendedGoodreadsIDs = finalRecommendedGoodreadsIDsCB // Update with filtered list

	if len(recommendedGoodreadsIDs) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	// 4. Fetch full book details for the recommended Goodreads IDs
	dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, recommendedGoodreadsIDs)
	if err != nil {
		fmt.Printf("Error fetching recommended book details from DB: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
		return
	}

	responseBooks := make([]BookResponse, 0, len(dbBooks))
	seenTitles := make(map[string]struct{}) // To track unique titles

	for _, bookRow := range dbBooks {
		if _, exists := seenTitles[bookRow.Title]; exists {
			continue // Skip if title already processed
		}
		seenTitles[bookRow.Title] = struct{}{}

		var apiAvgRating *float64
		if bookRow.AverageRating.Valid {
			float8Val, convErr := bookRow.AverageRating.Float64Value()
			if convErr == nil && float8Val.Valid {
				apiAvgRating = &float8Val.Float64
			} else if convErr != nil {
				fmt.Printf("Error converting AverageRating for book ID %v: %v\n", bookRow.ID, convErr)
			}
		}
		responseBooks = append(responseBooks, BookResponse{
			ID:              bookRow.ID,
			GoodreadsID:     bookRow.GoodreadsID,
			GoodreadsUrl:    bookRow.GoodreadsUrl,
			Title:           bookRow.Title,
			Description:     bookRow.Description,
			PublicationYear: bookRow.PublicationYear,
			CoverImageUrl:   bookRow.CoverImageUrl,
			AverageRating:   apiAvgRating,
			RatingsCount:    bookRow.RatingsCount,
			Authors:         processStringArrayInterface(bookRow.Authors),
			Genres:          processStringArrayInterface(bookRow.Genres),
			CreatedAt:       bookRow.CreatedAt,
			UpdatedAt:       bookRow.UpdatedAt,
		})
	}
	c.JSON(http.StatusOK, responseBooks)
}

// GetCollaborativeRecommendations returns collaborative filtering recommendations for a user
// (Existing function - can be kept as is or adapted)
func (rs *RecommendationService) GetCollaborativeRecommendations(c *gin.Context) {
	userID := c.Param("userId") // Assuming this is the string representation of the user's Qdrant ID

	userPoints, err := rs.QdrantClient.Get(context.Background(), &qdrant.GetPoints{
		CollectionName: "gmf_user_embeddings", // User embeddings collection
		Ids: []*qdrant.PointId{
			qdrant.NewID(userID), // User IDs in Qdrant are strings
		},
		WithVectors: qdrant.NewWithVectors(true),
	})

	if err != nil {
		fmt.Printf("Error retrieving user embedding for %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve user embedding"})
		return
	}

	if len(userPoints) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found or no embedding available"})
		return
	}
	if userPoints[0].Vectors == nil || userPoints[0].Vectors.GetVector() == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "User embedding vector is nil"})
		return
	}
	userVector := userPoints[0].Vectors.GetVector().Data

	limit := uint64(10)
	// Find similar books based on user embedding from gmf_book_embeddings
	similarBookPoints, err := rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "gmf_book_embeddings", // Book embeddings for collaborative filtering
		Query:          qdrant.NewQueryDense(userVector),
		Limit:          &limit,
		// Potentially add a filter here if user has already interacted with some books
	})

	if err != nil {
		fmt.Printf("Error querying Qdrant for collaborative recs: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve book recommendations"})
		return
	}

	// Similar to GetContentBasedRecommendations, map these points to BookResponse
	// For brevity, returning raw points.
	c.JSON(http.StatusOK, similarBookPoints) // Placeholder: ideally return []BookResponse
}

// GetSimilarUsers finds users with similar taste to a given user.
func (h *Handler) GetSimilarUsers(c *gin.Context) {
	userID := c.Param("id") // This is the string UUID from the path
	ctx := c.Request.Context()
	qdrantUserCollection := "gmf_user_embeddings"
	recommendationLimit := uint64(10)

	// 1. Get the source user's embedding
	sourceUserEmbedding, err := getEmbeddingByUserID(ctx, h.QdrantClient, userID, qdrantUserCollection)
	if err != nil {
		fmt.Printf("Error getting embedding for source user ID %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not retrieve embedding for source user"})
		return
	}

	// 2. Query Qdrant for similar user vectors, excluding the source user itself.
	// Assumes user points in Qdrant have a payload field "id_str" storing the string UUID for filtering.
	similarUserPoints, err := querySimilarUsersByVector(ctx, h.QdrantClient, qdrantUserCollection, sourceUserEmbedding, recommendationLimit, userID)
	if err != nil {
		fmt.Printf("Error querying Qdrant for similar users to %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to find similar users"})
		return
	}

	if len(similarUserPoints) == 0 {
		c.JSON(http.StatusOK, []db.User{}) // Return empty list of users
		return
	}

	// 3. Extract similar user UUIDs from Qdrant points
	similarUserUUIDStrings := make([]string, 0, len(similarUserPoints))
	for _, point := range similarUserPoints {
		if point.Id != nil {
			// Qdrant user IDs are strings (UUIDs). Access via GetUuid()
			userQdrantID := point.Id.GetUuid()
			if userQdrantID != userID { // Ensure the source user is not included
				similarUserUUIDStrings = append(similarUserUUIDStrings, userQdrantID)
			}
		}
		// Add fallback for payload if needed
	}

	if len(similarUserUUIDStrings) == 0 {
		c.JSON(http.StatusOK, []db.User{})
		return
	}

	// 4. Fetch full user details for these UUIDs from PostgreSQL
	responseUsers := make([]db.User, 0, len(similarUserUUIDStrings))
	for _, uuidStr := range similarUserUUIDStrings {
		pgUUID, err := uuid.Parse(uuidStr)
		if err != nil {
			fmt.Printf("Warning: Could not parse stored Qdrant user ID %s as UUID: %v\n", uuidStr, err)
			continue
		}
		dbUserRow, err := h.DB.GetUserByID(ctx, pgtype.UUID{Bytes: pgUUID, Valid: true})
		if err != nil {
			// Log error but continue, so if one user fetch fails, others might still succeed.
			fmt.Printf("Warning: Could not fetch user details for UUID %s from DB: %v\n", uuidStr, err)
			continue
		}
		responseUsers = append(responseUsers, db.User{
			ID:        dbUserRow.ID,
			Name:      dbUserRow.Name,
			Email:     dbUserRow.Email, // Consider if email should be exposed here
			CreatedAt: dbUserRow.CreatedAt,
			UpdatedAt: dbUserRow.UpdatedAt,
		})
	}
	c.JSON(http.StatusOK, responseUsers)
}

// GetBookRecommendationsForUser recommends books for a given user using a hybrid approach.
func (h *Handler) GetBookRecommendationsForUser(c *gin.Context) {
	userID := c.Param("id") // This is the string UUID from the path
	ctx := c.Request.Context()

	qdrantUserCollection := "gmf_user_embeddings"
	qdrantGmfBookCollection := "gmf_book_embeddings"
	qdrantSbertBookCollection := "sbert_embeddings" // As per new requirement

	limitPerSource := uint64(5) // Target 5 recommendations from each source
	// Fetch more initially to account for filtering and overlap
	initialQueryLimit := limitPerSource * 2

	// --- Fetch user's current library (Goodreads IDs) ---
	pgUserID, err := uuid.Parse(userID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid user ID format for DB query"})
		return
	}
	userLibraryDBRows, err := h.DB.GetUserLibraryDetails(ctx, pgtype.UUID{Bytes: pgUserID, Valid: true})
	if err != nil {
		// Log error but proceed; recommendations might be less personalized if library is missing
		fmt.Printf("Warning: Error fetching library for user %s: %v. Proceeding without full library exclusion for GMF part.\n", userID, err)
	}
	libraryGoodreadsIDs := make([]int64, 0, len(userLibraryDBRows))
	libraryIDSet := make(map[int64]struct{})
	for _, libBookRow := range userLibraryDBRows {
		libraryGoodreadsIDs = append(libraryGoodreadsIDs, libBookRow.GoodreadsID)
		libraryIDSet[libBookRow.GoodreadsID] = struct{}{}
	}

	var gmfRecsGoodreadsIDs []int64
	var sbertRecsGoodreadsIDs []int64
	// finalRecommendedGoodreadsIDsMap := make(map[int64]struct{}) // Removed unused variable

	// --- Part 1: GMF-based Recommendations (Collaborative Style) ---
	userEmbedding, err := getEmbeddingByUserID(ctx, h.QdrantClient, userID, qdrantUserCollection)
	if err != nil {
		fmt.Printf("Error getting GMF user embedding for user ID %s: %v\n", userID, err)
		// Don't fail outright, SBERT part might still work
	} else {
		gmfSimilarBookPoints, err := querySimilarByVectorWithExclusion(ctx, h.QdrantClient, qdrantGmfBookCollection, userEmbedding, initialQueryLimit, libraryGoodreadsIDs)
		if err != nil {
			fmt.Printf("Error querying GMF book recommendations for user %s: %v\n", userID, err)
		} else {
			for _, point := range gmfSimilarBookPoints {
				if point.Id != nil {
					gid := int64(point.Id.GetNum())
					if _, existsInLibrary := libraryIDSet[gid]; !existsInLibrary {
						gmfRecsGoodreadsIDs = append(gmfRecsGoodreadsIDs, gid)
					}
				}
			}
		}
	}

	// --- Part 2: SBERT-based Recommendations (Content from Library Average) ---
	if len(libraryGoodreadsIDs) > 0 {
		sbertSimilarBookPoints, err := getAverageEmbeddingsAndFindSimilarWithExclusion(ctx, h.QdrantClient, libraryGoodreadsIDs, qdrantSbertBookCollection, initialQueryLimit)
		if err != nil {
			fmt.Printf("Error getting SBERT recommendations based on library average for user %s: %v\n", userID, err)
		} else {
			for _, point := range sbertSimilarBookPoints {
				if point.Id != nil {
					gid := int64(point.Id.GetNum())
					// getAverageEmbeddingsAndFindSimilarWithExclusion already excludes libraryGoodreadsIDs via its call to querySimilarByVectorWithExclusion
					sbertRecsGoodreadsIDs = append(sbertRecsGoodreadsIDs, gid)
				}
			}
		}
	}

	// --- Combine and select top N from each, ensuring uniqueness and desired order ---
	// SBERT recommendations first, then GMF.

	var orderedFinalGoodreadsIDs []int64
	// tempFinalRecommendedGoodreadsIDsMap was the variable name from the previous attempt,
	// let's use tempCombinedRecsMap for clarity as it's for combining.
	tempCombinedRecsMap := make(map[int64]struct{}) // Used to track uniqueness while building the ordered list

	// Add SBERT recommendations first
	sbertCollectedCount := 0
	for _, gid := range sbertRecsGoodreadsIDs {
		if sbertCollectedCount >= int(limitPerSource) { // Max 5 from SBERT
			break
		}
		// Ensure it's not in the library AND not already added to our combined list
		if _, existsInLibrary := libraryIDSet[gid]; !existsInLibrary {
			if _, alreadyAdded := tempCombinedRecsMap[gid]; !alreadyAdded {
				orderedFinalGoodreadsIDs = append(orderedFinalGoodreadsIDs, gid)
				tempCombinedRecsMap[gid] = struct{}{}
				sbertCollectedCount++
			}
		}
	}

	// Then add GMF recommendations
	gmfCollectedCount := 0
	for _, gid := range gmfRecsGoodreadsIDs {
		if gmfCollectedCount >= int(limitPerSource) { // Max 5 from GMF
			break
		}
		// Ensure it's not in the library AND not already added to our combined list
		if _, existsInLibrary := libraryIDSet[gid]; !existsInLibrary {
			if _, alreadyAdded := tempCombinedRecsMap[gid]; !alreadyAdded {
				orderedFinalGoodreadsIDs = append(orderedFinalGoodreadsIDs, gid)
				tempCombinedRecsMap[gid] = struct{}{}
				gmfCollectedCount++
			}
		}
	}

	if len(orderedFinalGoodreadsIDs) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	// --- Fetch full book details ---
	dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, orderedFinalGoodreadsIDs)
	if err != nil {
		fmt.Printf("Error fetching combined recommended book details from DB for user %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
		return
	}

	responseBooks := make([]BookResponse, 0, len(dbBooks))
	seenTitles := make(map[string]struct{}) // To track unique titles

	for _, bookRow := range dbBooks {
		if _, exists := seenTitles[bookRow.Title]; exists {
			continue // Skip if title already processed
		}
		seenTitles[bookRow.Title] = struct{}{}

		var apiAvgRating *float64
		if bookRow.AverageRating.Valid {
			float8Val, convErr := bookRow.AverageRating.Float64Value()
			if convErr == nil && float8Val.Valid {
				apiAvgRating = &float8Val.Float64
			}
		}
		responseBooks = append(responseBooks, BookResponse{
			ID:              bookRow.ID,
			GoodreadsID:     bookRow.GoodreadsID,
			GoodreadsUrl:    bookRow.GoodreadsUrl,
			Title:           bookRow.Title,
			Description:     bookRow.Description,
			PublicationYear: bookRow.PublicationYear,
			CoverImageUrl:   bookRow.CoverImageUrl,
			AverageRating:   apiAvgRating,
			RatingsCount:    bookRow.RatingsCount,
			Authors:         processStringArrayInterface(bookRow.Authors),
			Genres:          processStringArrayInterface(bookRow.Genres),
			CreatedAt:       bookRow.CreatedAt,
			UpdatedAt:       bookRow.UpdatedAt,
		})
	}
	c.JSON(http.StatusOK, responseBooks)
}

// GetHybridRecommendations combines both content-based and collaborative filtering recommendations
// (Existing function - can be reviewed or adapted)
func (rs *RecommendationService) GetHybridRecommendations(c *gin.Context) {
	// This function would typically involve:
	// 1. Getting content-based recommendations (e.g., for a specific book).
	// 2. Getting collaborative filtering recommendations (e.g., for the user).
	// 3. Combining and re-ranking these lists.
	// The current implementation fetches embeddings and recs but doesn't show a clear combination strategy.
	// For a true hybrid, you'd fetch two lists of BookResponse and then merge/re-rank.

	// For now, let's assume the logic needs more refinement for a proper hybrid response.
	// The existing code fetches raw Qdrant points.
	bookIDStr := c.Param("bookId") // e.g., Goodreads ID for content-based anchor
	userID := c.Param("userId")    // User ID for collaborative part

	// ... (logic to get contentBasedRecs as []*qdrant.ScoredPoint) ...
	// ... (logic to get collaborativeRecs as []*qdrant.ScoredPoint) ...

	// Example: Fetching content-based (assuming bookIDStr is Goodreads ID)
	bookGID, err := strconv.ParseInt(bookIDStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID for content part"})
		return
	}
	contentLimit := uint64(5)
	contentBasedRecs, _ := rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "sbert_embeddings", // Or gmf_book_embeddings
		Query:          qdrant.NewQueryID(qdrant.NewIDNum(uint64(bookGID))),
		Limit:          &contentLimit,
	})

	// Example: Fetching collaborative
	userPoints, err := rs.QdrantClient.Get(context.Background(), &qdrant.GetPoints{
		CollectionName: "gmf_user_embeddings",
		Ids:            []*qdrant.PointId{qdrant.NewID(userID)},
		WithVectors:    qdrant.NewWithVectors(true),
	})
	var collaborativeRecs []*qdrant.ScoredPoint
	if err == nil && len(userPoints) > 0 && userPoints[0].Vectors != nil && userPoints[0].Vectors.GetVector() != nil {
		collabLimit := uint64(5)
		collaborativeRecs, _ = rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
			CollectionName: "gmf_book_embeddings",
			Query:          qdrant.NewQueryDense(userPoints[0].Vectors.GetVector().Data),
			Limit:          &collabLimit,
		})
	}

	// Actual combination logic is non-trivial (re-ranking, ensuring diversity, etc.)
	// For now, just returning them separately as the original code did.
	response := gin.H{
		"content_based_raw": contentBasedRecs,  // Ideally map to []BookResponse
		"collaborative_raw": collaborativeRecs, // Ideally map to []BookResponse
	}

	c.JSON(http.StatusOK, response)
}
