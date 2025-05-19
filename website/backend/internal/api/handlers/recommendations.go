package handlers

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv" // Added for string to int64 conversion

	"github.com/gin-gonic/gin"
	"github.com/google/uuid" // Added for UUID parsing
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/qdrant/go-client/qdrant"
	"github.com/yamirghofran/BookDB/internal/db" // Added for DB queries
)

// RecommendationService struct and existing helper functions (getEmbeddingByGoodreadsID, etc.) remain here...
// ... (lines 16-206 from original file)

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
		WithPayload:    qdrant.NewWithPayload(true),
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
		condition := qdrant.NewMatchText("id_str", excludeUserID)
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
		WithPayload:    qdrant.NewWithPayload(true),
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
		return []*qdrant.ScoredPoint{}, nil
	}

	var allEmbeddings [][]float32
	for _, gid := range goodreadsIDs {
		embedding, err := getEmbeddingByGoodreadsID(ctx, qdrantClient, gid, collectionName)
		if err != nil {
			fmt.Printf("Warning: could not get embedding for Goodreads ID %d: %v\n", gid, err)
			continue
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

// --- RERANKER LOGIC START ---

const (
	defaultLambdaParam                 = 0.7
	defaultPubYearBoostFactor          = 0.01
	defaultPubYearBoostBaseYear        = 2000
	rerankerInitialCandidateMultiplier = 3 // Fetch N*X candidates initially for reranking to get N
)

// RerankerBookInfo holds processed metadata and scores for a book during reranking.
type RerankerBookInfo struct {
	GoodreadsID     int64
	PublicationYear int32
	Genres          []string
	Authors         []string
	Representation  map[string]struct{} // Combined set of unique genre and author strings
	InitialScore    float32             // Score from Qdrant or other base model
	AdjustedScore   float32             // Score after heuristic adjustments (e.g., pub year boost)
}

// newRerankerBookInfo creates a RerankerBookInfo object from database row and initial score.
func newRerankerBookInfo(bookRow db.GetBooksByGoodreadsIDsRow, initialScore float32) RerankerBookInfo {
	genres := processStringArrayInterface(bookRow.Genres)
	authors := processStringArrayInterface(bookRow.Authors)

	representation := make(map[string]struct{})
	for _, g := range genres {
		if g != "" {
			representation[g] = struct{}{}
		}
	}
	for _, a := range authors {
		if a != "" {
			representation[a] = struct{}{}
		}
	}

	var pubYear int32
	if bookRow.PublicationYear.Valid {
		// pgtype.Int8 maps to int64 in Go, publication year should fit in int32
		pubYear = int32(bookRow.PublicationYear.Int64)
	}

	return RerankerBookInfo{
		GoodreadsID:     bookRow.GoodreadsID,
		PublicationYear: pubYear,
		Genres:          genres,
		Authors:         authors,
		Representation:  representation,
		InitialScore:    initialScore,
		AdjustedScore:   initialScore, // Will be adjusted later
	}
}

// jaccardSimilarity calculates the Jaccard similarity between two sets of strings.
func jaccardSimilarity(setA, setB map[string]struct{}) float64 {
	intersectionSize := 0
	unionSet := make(map[string]struct{})

	for k := range setA {
		unionSet[k] = struct{}{}
		if _, exists := setB[k]; exists {
			intersectionSize++
		}
	}
	for k := range setB {
		unionSet[k] = struct{}{}
	}

	unionSize := len(unionSet)

	if unionSize == 0 { // Both sets were empty
		return 0.0 // Or 1.0 if convention prefers, Python example implies 0 for problematic cases
	}
	return float64(intersectionSize) / float64(unionSize)
}

// mmrDiversify reranks candidates using Maximal Marginal Relevance.
func mmrDiversify(
	candidatesToRank []RerankerBookInfo, // Candidates with all necessary info including adjusted scores and representations
	k int, // Target number of books to pick
	lambdaParam float64, // Trade-off between relevance (score) vs. diversity
) []int64 {
	if len(candidatesToRank) == 0 || k <= 0 {
		return []int64{}
	}

	// Ensure k is not greater than the number of candidates
	if k > len(candidatesToRank) {
		k = len(candidatesToRank)
	}

	selectedIDs := make([]int64, 0, k)
	selectedRepresentations := make([]map[string]struct{}, 0, k) // Store representations of selected items

	// Create a pool of candidates by their GoodreadsID for easy lookup and removal
	pool := make(map[int64]RerankerBookInfo)
	for _, c := range candidatesToRank {
		pool[c.GoodreadsID] = c
	}

	// Pick the highest-scored book first
	var firstCandidate RerankerBookInfo
	firstCandidateFound := false
	for _, candidate := range candidatesToRank {
		if !firstCandidateFound || candidate.AdjustedScore > firstCandidate.AdjustedScore {
			firstCandidate = candidate
			firstCandidateFound = true
		}
	}

	if !firstCandidateFound { // Should not happen if candidatesToRank is not empty
		return []int64{}
	}

	selectedIDs = append(selectedIDs, firstCandidate.GoodreadsID)
	selectedRepresentations = append(selectedRepresentations, firstCandidate.Representation)
	delete(pool, firstCandidate.GoodreadsID)

	// MMR loop
	for len(selectedIDs) < k && len(pool) > 0 {
		bestNextID := int64(-1)
		maxMMRScore := -1.0 * math.MaxFloat64

		for _, candidateInPool := range pool {
			relevance := float64(candidateInPool.AdjustedScore)
			maxSimToSelected := 0.0

			if len(selectedRepresentations) > 0 { // Only calculate similarity if items have been selected
				currentSimToSelected := 0.0
				for _, selRep := range selectedRepresentations {
					sim := jaccardSimilarity(candidateInPool.Representation, selRep)
					if sim > currentSimToSelected {
						currentSimToSelected = sim
					}
				}
				maxSimToSelected = currentSimToSelected
			}

			mmrScore := lambdaParam*relevance - (1-lambdaParam)*maxSimToSelected

			if mmrScore > maxMMRScore {
				maxMMRScore = mmrScore
				bestNextID = candidateInPool.GoodreadsID
			}
		}

		if bestNextID != -1 {
			selectedCandidateInfo := pool[bestNextID]
			selectedIDs = append(selectedIDs, bestNextID)
			selectedRepresentations = append(selectedRepresentations, selectedCandidateInfo.Representation)
			delete(pool, bestNextID)
		} else {
			break // No suitable candidate found
		}
	}
	return selectedIDs
}

// InitialCandidate holds basic info for a candidate before full reranking metadata is fetched.
type InitialCandidate struct {
	GoodreadsID int64
	Score       float32
}

// applyRerankingLogic filters, adjusts scores, and diversifies candidates.
func applyRerankingLogic(
	ctx context.Context,
	dbQueries *db.Queries,
	initialRawCandidates []InitialCandidate,
	userHistoryGoodreadsIDs map[int64]struct{}, // Books to exclude (e.g., user's library or input books)
	finalK int, // Desired number of recommendations
	lambdaParam float64,
	pubYearBoostFactor float64,
	pubYearBoostBaseYear int,
) ([]int64, error) {
	if len(initialRawCandidates) == 0 || finalK <= 0 {
		return []int64{}, nil
	}

	// 1. Filter out user history and prepare for metadata fetching
	candidateIDsToFetchMeta := make([]int64, 0, len(initialRawCandidates))
	scoresMap := make(map[int64]float32) // Store initial scores by GoodreadsID

	for _, ic := range initialRawCandidates {
		if _, existsInHistory := userHistoryGoodreadsIDs[ic.GoodreadsID]; !existsInHistory {
			candidateIDsToFetchMeta = append(candidateIDsToFetchMeta, ic.GoodreadsID)
			scoresMap[ic.GoodreadsID] = ic.Score
		}
	}

	if len(candidateIDsToFetchMeta) == 0 {
		return []int64{}, nil
	}

	// 2. Fetch book metadata for filtered candidates
	// Ensure unique IDs are passed to GetBooksByGoodreadsIDs if there's any chance of duplicates earlier
	uniqueCandidateIDs := make([]int64, 0, len(candidateIDsToFetchMeta))
	seenForMetaFetch := make(map[int64]struct{})
	for _, id := range candidateIDsToFetchMeta {
		if _, seen := seenForMetaFetch[id]; !seen {
			uniqueCandidateIDs = append(uniqueCandidateIDs, id)
			seenForMetaFetch[id] = struct{}{}
		}
	}

	if len(uniqueCandidateIDs) == 0 {
		return []int64{}, nil
	}

	dbBookRows, err := dbQueries.GetBooksByGoodreadsIDs(ctx, uniqueCandidateIDs)
	if err != nil {
		return nil, fmt.Errorf("applyRerankingLogic: failed to get book metadata: %w", err)
	}

	// 3. Create RerankerBookInfo, apply score adjustments (e.g., pub year boost)
	rerankerCandidates := make([]RerankerBookInfo, 0, len(dbBookRows))
	for _, bookRow := range dbBookRows {
		initialScore, ok := scoresMap[bookRow.GoodreadsID]
		if !ok {
			// This book was in dbBookRows but not in our initial scoresMap after filtering.
			// This might happen if a book from userHistory was also a candidate.
			// Or if GetBooksByGoodreadsIDs returned a book not in uniqueCandidateIDs (should not happen).
			// For safety, skip.
			fmt.Printf("Warning: applyRerankingLogic - book %d metadata fetched but no initial score found after filtering. Skipping.\n", bookRow.GoodreadsID)
			continue
		}

		info := newRerankerBookInfo(bookRow, initialScore)

		// Apply publication year boost
		if info.PublicationYear > 0 { // Ensure valid publication year
			boost := float32(pubYearBoostFactor * float64(int(info.PublicationYear)-pubYearBoostBaseYear))
			info.AdjustedScore += boost
		}
		rerankerCandidates = append(rerankerCandidates, info)
	}

	if len(rerankerCandidates) == 0 {
		return []int64{}, nil
	}

	// Sort candidates by adjusted score descending before passing to MMR,
	// as MMR's first pick relies on the highest score.
	// While MMR itself finds the max, pre-sorting helps if the first pick logic is strict.
	// The current mmrDiversify finds its own max, so this sort is for conceptual alignment / potential future use.
	sort.SliceStable(rerankerCandidates, func(i, j int) bool {
		return rerankerCandidates[i].AdjustedScore > rerankerCandidates[j].AdjustedScore
	})

	// 4. Diversify via MMR
	rerankedIDs := mmrDiversify(rerankerCandidates, finalK, lambdaParam)

	return rerankedIDs, nil
}

// --- RERANKER LOGIC END ---

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
	// inputBookUUIDs := make([]pgtype.UUID, 0, len(req.LikedBookIds)) // Not directly used after fetching GoodreadsIDs

	userHistoryGoodreadsIDs := make(map[int64]struct{}) // For reranker exclusion

	for _, idStr := range req.LikedBookIds {
		bookUUID, err := uuid.Parse(idStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Invalid book UUID format: %s", idStr)})
			return
		}
		pgUUID := pgtype.UUID{Bytes: bookUUID, Valid: true}
		// inputBookUUIDs = append(inputBookUUIDs, pgUUID) // Keep if needed elsewhere

		book, err := h.DB.GetBookByID(ctx, pgUUID)
		if err != nil {
			fmt.Printf("Warning: Could not fetch book details for UUID %s: %v\n", idStr, err)
			continue
		}
		inputGoodreadsIDs = append(inputGoodreadsIDs, book.GoodreadsID)
		userHistoryGoodreadsIDs[book.GoodreadsID] = struct{}{} // Add to history for exclusion
	}

	if len(inputGoodreadsIDs) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	recommendationTargetCount := 10 // Final number of recommendations to return
	// Fetch more candidates for reranking
	initialCandidateQueryLimit := uint64(recommendationTargetCount * rerankerInitialCandidateMultiplier)
	qdrantCollection := "gmf_book_embeddings"

	// Get average embedding of liked books (this part remains the same)
	// Note: getAverageEmbeddingsAndFindSimilarWithExclusion already excludes inputGoodreadsIDs from Qdrant query
	// However, the reranker needs the raw candidates *before* Qdrant's exclusion if we want to manage exclusion consistently.
	// For simplicity with existing helpers, we'll let getAverageEmbeddingsAndFindSimilarWithExclusion do its exclusion,
	// and the reranker will primarily handle diversity and recency boosts.
	// The userHistoryGoodreadsIDs passed to reranker will ensure liked books are not re-recommended if they somehow pass Qdrant filter.

	var avgEmbedding []float32
	var err error
	// Simplified averaging logic for clarity, assuming getAverageEmbeddingsAndFindSimilarWithExclusion internals
	// This part is to get the avgEmbedding to query with.
	if len(inputGoodreadsIDs) > 0 {
		var allEmbeddings [][]float32
		for _, gid := range inputGoodreadsIDs {
			embedding, embErr := getEmbeddingByGoodreadsID(ctx, h.QdrantClient, gid, qdrantCollection)
			if embErr != nil {
				fmt.Printf("Warning: could not get embedding for Goodreads ID %d for averaging: %v\n", gid, embErr)
				continue
			}
			allEmbeddings = append(allEmbeddings, embedding)
		}
		if len(allEmbeddings) > 0 {
			embeddingDim := len(allEmbeddings[0])
			avgEmbedding = make([]float32, embeddingDim)
			for _, embedding := range allEmbeddings {
				for i, val := range embedding {
					avgEmbedding[i] += val
				}
			}
			for i := range avgEmbedding {
				avgEmbedding[i] /= float32(len(allEmbeddings))
			}
		} else {
			fmt.Println("Warning: No embeddings found for liked books to calculate average.")
			c.JSON(http.StatusOK, []BookResponse{})
			return
		}
	} else {
		c.JSON(http.StatusOK, []BookResponse{}) // Should have been caught earlier
		return
	}

	// Query Qdrant for initial candidates, excluding liked books (as per querySimilarByVectorWithExclusion)
	similarPoints, err := querySimilarByVectorWithExclusion(ctx, h.QdrantClient, qdrantCollection, avgEmbedding, initialCandidateQueryLimit, inputGoodreadsIDs)
	if err != nil {
		fmt.Printf("Error finding similar points for anonymous recs: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to calculate recommendations"})
		return
	}

	if len(similarPoints) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	initialCandidatesForReranker := make([]InitialCandidate, 0, len(similarPoints))
	for _, point := range similarPoints {
		if point.Id != nil {
			gid := int64(point.Id.GetNum())
			// Double check exclusion, though querySimilarByVectorWithExclusion should handle it
			if _, existsInLiked := userHistoryGoodreadsIDs[gid]; !existsInLiked {
				initialCandidatesForReranker = append(initialCandidatesForReranker, InitialCandidate{
					GoodreadsID: gid,
					Score:       point.Score,
				})
			}
		}
	}

	if len(initialCandidatesForReranker) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	// Apply reranking logic
	rerankedGoodreadsIDs, err := applyRerankingLogic(
		ctx,
		h.DB,
		initialCandidatesForReranker,
		userHistoryGoodreadsIDs, // Liked books are the history here
		recommendationTargetCount,
		defaultLambdaParam,
		defaultPubYearBoostFactor,
		defaultPubYearBoostBaseYear,
	)
	if err != nil {
		fmt.Printf("Error applying reranking logic for anonymous recs: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to rerank recommendations"})
		return
	}

	if len(rerankedGoodreadsIDs) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	// Fetch full book details for the reranked Goodreads IDs
	dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, rerankedGoodreadsIDs)
	if err != nil {
		fmt.Printf("Error fetching reranked recommended book details from DB: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
		return
	}

	// Order dbBooks according to rerankedGoodreadsIDs
	orderedDbBooks := make([]db.GetBooksByGoodreadsIDsRow, 0, len(dbBooks))
	bookMap := make(map[int64]db.GetBooksByGoodreadsIDsRow)
	for _, b := range dbBooks {
		bookMap[b.GoodreadsID] = b
	}
	for _, id := range rerankedGoodreadsIDs {
		if book, ok := bookMap[id]; ok {
			orderedDbBooks = append(orderedDbBooks, book)
		}
	}

	responseBooks := make([]BookResponse, 0, len(orderedDbBooks))
	seenTitles := make(map[string]struct{})

	for _, bookRow := range orderedDbBooks { // Use orderedDbBooks
		if _, exists := seenTitles[bookRow.Title]; exists {
			continue
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
	qdrantCollection := "gmf_book_embeddings" // Or "sbert_embeddings"
	sourceEmbedding, err := getEmbeddingByGoodreadsID(ctx, h.QdrantClient, sourceGoodreadsID, qdrantCollection)
	if err != nil {
		fmt.Printf("Error getting embedding for source book GoodreadsID %d: %v\n", sourceGoodreadsID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not retrieve embedding for source book"})
		return
	}

	// 3. Query Qdrant for similar books, excluding the source book itself
	recommendationTargetCount := 10
	initialCandidateQueryLimit := uint64(recommendationTargetCount * rerankerInitialCandidateMultiplier)

	userHistoryGoodreadsIDs := map[int64]struct{}{sourceGoodreadsID: {}} // Exclude the source book

	similarPoints, err := querySimilarByVectorWithExclusion(ctx, h.QdrantClient, qdrantCollection, sourceEmbedding, initialCandidateQueryLimit, []int64{sourceGoodreadsID})
	if err != nil {
		fmt.Printf("Error querying Qdrant for similar books to %d: %v\n", sourceGoodreadsID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to find similar books"})
		return
	}

	if len(similarPoints) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	initialCandidatesForReranker := make([]InitialCandidate, 0, len(similarPoints))
	for _, point := range similarPoints {
		if point.Id != nil {
			gid := int64(point.Id.GetNum())
			if gid != sourceGoodreadsID { // Ensure source book is not included
				initialCandidatesForReranker = append(initialCandidatesForReranker, InitialCandidate{
					GoodreadsID: gid,
					Score:       point.Score,
				})
			}
		}
	}

	if len(initialCandidatesForReranker) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	rerankedGoodreadsIDs, err := applyRerankingLogic(
		ctx,
		h.DB,
		initialCandidatesForReranker,
		userHistoryGoodreadsIDs, // Source book is the history
		recommendationTargetCount,
		defaultLambdaParam,
		defaultPubYearBoostFactor,
		defaultPubYearBoostBaseYear,
	)
	if err != nil {
		fmt.Printf("Error applying reranking logic for content-based recs: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to rerank recommendations"})
		return
	}

	if len(rerankedGoodreadsIDs) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	// 4. Fetch full book details for the reranked Goodreads IDs
	dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, rerankedGoodreadsIDs)
	if err != nil {
		fmt.Printf("Error fetching recommended book details from DB: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
		return
	}

	// Order dbBooks according to rerankedGoodreadsIDs
	orderedDbBooks := make([]db.GetBooksByGoodreadsIDsRow, 0, len(dbBooks))
	bookMap := make(map[int64]db.GetBooksByGoodreadsIDsRow)
	for _, b := range dbBooks {
		bookMap[b.GoodreadsID] = b
	}
	for _, id := range rerankedGoodreadsIDs {
		if book, ok := bookMap[id]; ok {
			orderedDbBooks = append(orderedDbBooks, book)
		}
	}

	responseBooks := make([]BookResponse, 0, len(orderedDbBooks))
	seenTitles := make(map[string]struct{})

	for _, bookRow := range orderedDbBooks {
		if _, exists := seenTitles[bookRow.Title]; exists {
			continue
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
// For now, this function is NOT MODIFIED to use the new reranker.
// It would require similar changes to GetBookRecommendationsForUser if it were to be fully fledged.
func (rs *RecommendationService) GetCollaborativeRecommendations(c *gin.Context) {
	userID := c.Param("userId")

	userPoints, err := rs.QdrantClient.Get(context.Background(), &qdrant.GetPoints{
		CollectionName: "gmf_user_embeddings",
		Ids: []*qdrant.PointId{
			qdrant.NewID(userID),
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
	similarBookPoints, err := rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "gmf_book_embeddings",
		Query:          qdrant.NewQueryDense(userVector),
		Limit:          &limit,
	})

	if err != nil {
		fmt.Printf("Error querying Qdrant for collaborative recs: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve book recommendations"})
		return
	}
	c.JSON(http.StatusOK, similarBookPoints)
}

// GetSimilarUsers finds users with similar taste to a given user.
// (Existing function - NOT MODIFIED)
func (h *Handler) GetSimilarUsers(c *gin.Context) {
	userID := c.Param("id")
	ctx := c.Request.Context()
	qdrantUserCollection := "gmf_user_embeddings"
	recommendationLimit := uint64(10)

	sourceUserEmbedding, err := getEmbeddingByUserID(ctx, h.QdrantClient, userID, qdrantUserCollection)
	if err != nil {
		fmt.Printf("Error getting embedding for source user ID %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not retrieve embedding for source user"})
		return
	}
	similarUserPoints, err := querySimilarUsersByVector(ctx, h.QdrantClient, qdrantUserCollection, sourceUserEmbedding, recommendationLimit, userID)
	if err != nil {
		fmt.Printf("Error querying Qdrant for similar users to %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to find similar users"})
		return
	}

	if len(similarUserPoints) == 0 {
		c.JSON(http.StatusOK, []db.User{})
		return
	}
	similarUserUUIDStrings := make([]string, 0, len(similarUserPoints))
	for _, point := range similarUserPoints {
		if point.Id != nil {
			userQdrantID := point.Id.GetUuid()
			if userQdrantID != userID {
				similarUserUUIDStrings = append(similarUserUUIDStrings, userQdrantID)
			}
		}
	}

	if len(similarUserUUIDStrings) == 0 {
		c.JSON(http.StatusOK, []db.User{})
		return
	}
	responseUsers := make([]db.User, 0, len(similarUserUUIDStrings))
	for _, uuidStr := range similarUserUUIDStrings {
		pgUUID, err := uuid.Parse(uuidStr)
		if err != nil {
			fmt.Printf("Warning: Could not parse stored Qdrant user ID %s as UUID: %v\n", uuidStr, err)
			continue
		}
		dbUserRow, err := h.DB.GetUserByID(ctx, pgtype.UUID{Bytes: pgUUID, Valid: true})
		if err != nil {
			fmt.Printf("Warning: Could not fetch user details for UUID %s from DB: %v\n", uuidStr, err)
			continue
		}
		responseUsers = append(responseUsers, db.User{
			ID:        dbUserRow.ID,
			Name:      dbUserRow.Name,
			Email:     dbUserRow.Email,
			CreatedAt: dbUserRow.CreatedAt,
			UpdatedAt: dbUserRow.UpdatedAt,
		})
	}
	c.JSON(http.StatusOK, responseUsers)
}

// GetBookRecommendationsForUser recommends books for a given user using a hybrid approach.
// MODIFIED to use the new reranker logic.
func (h *Handler) GetBookRecommendationsForUser(c *gin.Context) {
	userID := c.Param("id")
	ctx := c.Request.Context()

	qdrantUserCollection := "gmf_user_embeddings"
	qdrantGmfBookCollection := "gmf_book_embeddings"
	qdrantSbertBookCollection := "sbert_embeddings"

	recommendationTargetCount := 10 // Final total recommendations
	// For hybrid, let's aim for roughly half from each source before reranking, then rerank combined pool
	limitPerSourceBeforeRerank := recommendationTargetCount * rerankerInitialCandidateMultiplier / 2
	if limitPerSourceBeforeRerank < recommendationTargetCount { // ensure enough candidates if target is small
		limitPerSourceBeforeRerank = recommendationTargetCount
	}

	pgUserID, err := uuid.Parse(userID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid user ID format for DB query"})
		return
	}
	userLibraryDBRows, err := h.DB.GetUserLibraryDetails(ctx, pgtype.UUID{Bytes: pgUserID, Valid: true})
	if err != nil {
		fmt.Printf("Warning: Error fetching library for user %s: %v. Proceeding without full library exclusion for GMF part.\n", userID, err)
	}
	libraryGoodreadsIDs := make([]int64, 0, len(userLibraryDBRows))
	userHistoryGoodreadsIDs := make(map[int64]struct{}) // For reranker
	for _, libBookRow := range userLibraryDBRows {
		libraryGoodreadsIDs = append(libraryGoodreadsIDs, libBookRow.GoodreadsID)
		userHistoryGoodreadsIDs[libBookRow.GoodreadsID] = struct{}{}
	}

	allInitialCandidates := make([]InitialCandidate, 0)
	seenCandidateGIDs := make(map[int64]struct{}) // To avoid duplicate candidates from different sources

	// --- Part 1: GMF-based Recommendations (Collaborative Style) ---
	userEmbedding, err := getEmbeddingByUserID(ctx, h.QdrantClient, userID, qdrantUserCollection)
	if err != nil {
		fmt.Printf("Error getting GMF user embedding for user ID %s: %v\n", userID, err)
	} else {
		gmfSimilarBookPoints, err := querySimilarByVectorWithExclusion(ctx, h.QdrantClient, qdrantGmfBookCollection, userEmbedding, uint64(limitPerSourceBeforeRerank), libraryGoodreadsIDs)
		if err != nil {
			fmt.Printf("Error querying GMF book recommendations for user %s: %v\n", userID, err)
		} else {
			for _, point := range gmfSimilarBookPoints {
				if point.Id != nil {
					gid := int64(point.Id.GetNum())
					if _, existsInLibrary := userHistoryGoodreadsIDs[gid]; !existsInLibrary {
						if _, seen := seenCandidateGIDs[gid]; !seen {
							allInitialCandidates = append(allInitialCandidates, InitialCandidate{GoodreadsID: gid, Score: point.Score})
							seenCandidateGIDs[gid] = struct{}{}
						}
					}
				}
			}
		}
	}

	// --- Part 2: SBERT-based Recommendations (Content from Library Average) ---
	if len(libraryGoodreadsIDs) > 0 {
		// getAverageEmbeddingsAndFindSimilarWithExclusion already excludes libraryGoodreadsIDs
		sbertSimilarBookPoints, err := getAverageEmbeddingsAndFindSimilarWithExclusion(ctx, h.QdrantClient, libraryGoodreadsIDs, qdrantSbertBookCollection, uint64(limitPerSourceBeforeRerank))
		if err != nil {
			fmt.Printf("Error getting SBERT recommendations based on library average for user %s: %v\n", userID, err)
		} else {
			for _, point := range sbertSimilarBookPoints {
				if point.Id != nil {
					gid := int64(point.Id.GetNum())
					// querySimilarByVectorWithExclusion (called by getAverage...) already handles library exclusion
					if _, existsInLibrary := userHistoryGoodreadsIDs[gid]; !existsInLibrary { // Double check
						if _, seen := seenCandidateGIDs[gid]; !seen {
							allInitialCandidates = append(allInitialCandidates, InitialCandidate{GoodreadsID: gid, Score: point.Score})
							seenCandidateGIDs[gid] = struct{}{}
						}
					}
				}
			}
		}
	}

	if len(allInitialCandidates) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	// --- Apply Reranking Logic to the combined list ---
	rerankedGoodreadsIDs, err := applyRerankingLogic(
		ctx,
		h.DB,
		allInitialCandidates,
		userHistoryGoodreadsIDs,   // User's library
		recommendationTargetCount, // Final desired count
		defaultLambdaParam,
		defaultPubYearBoostFactor,
		defaultPubYearBoostBaseYear,
	)
	if err != nil {
		fmt.Printf("Error applying reranking logic for user %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to rerank recommendations"})
		return
	}

	if len(rerankedGoodreadsIDs) == 0 {
		c.JSON(http.StatusOK, []BookResponse{})
		return
	}

	// --- Fetch full book details ---
	dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, rerankedGoodreadsIDs)
	if err != nil {
		fmt.Printf("Error fetching combined recommended book details from DB for user %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
		return
	}

	// Order dbBooks according to rerankedGoodreadsIDs
	orderedDbBooks := make([]db.GetBooksByGoodreadsIDsRow, 0, len(dbBooks))
	bookMap := make(map[int64]db.GetBooksByGoodreadsIDsRow)
	for _, b := range dbBooks {
		bookMap[b.GoodreadsID] = b
	}
	for _, id := range rerankedGoodreadsIDs {
		if book, ok := bookMap[id]; ok {
			orderedDbBooks = append(orderedDbBooks, book)
		}
	}

	responseBooks := make([]BookResponse, 0, len(orderedDbBooks))
	seenTitles := make(map[string]struct{})

	for _, bookRow := range orderedDbBooks {
		if _, exists := seenTitles[bookRow.Title]; exists {
			continue
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
// (Existing function - NOT MODIFIED, placeholder)
func (rs *RecommendationService) GetHybridRecommendations(c *gin.Context) {
	bookIDStr := c.Param("bookId")
	userID := c.Param("userId")

	bookGID, err := strconv.ParseInt(bookIDStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID for content part"})
		return
	}
	contentLimit := uint64(5)
	contentBasedRecs, _ := rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "sbert_embeddings",
		Query:          qdrant.NewQueryID(qdrant.NewIDNum(uint64(bookGID))),
		Limit:          &contentLimit,
	})

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
	response := gin.H{
		"content_based_raw": contentBasedRecs,
		"collaborative_raw": collaborativeRecs,
	}
	c.JSON(http.StatusOK, response)
}
