package handlers

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/qdrant/go-client/qdrant"
	"github.com/yamirghofran/BookDB/internal/db"
)

// Types
type RecommendationService struct {
	QdrantClient qdrant.Client
	DB           *db.Queries
}

type RerankerBookInfo struct {
	GoodreadsID     int64
	PublicationYear int32
	Genres          []string
	Authors         []string
	Representation  map[string]struct{}
	InitialScore    float32
	AdjustedScore   float32
}

type InitialCandidate struct {
	GoodreadsID int64
	Score       float32
}

// Constants
const (
	defaultLambdaParam                 = 0.7
	defaultPubYearBoostFactor          = 0.01
	defaultPubYearBoostBaseYear        = 2000
	rerankerInitialCandidateMultiplier = 3
)

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
			qdrant.NewID(userID),
		},
		WithVectors: qdrant.NewWithVectors(true),
	})

	if err != nil {
		if err != nil {
			return nil, fmt.Errorf("qdrant client.Get: unable to get embedding for User ID %s: %v", userID, err)
		}
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
func (s *RecommendationService) querySimilarByVectorWithExclusion(ctx context.Context, vector []float32, excludeIDs []int64, limit uint64, collectionName string) ([]*qdrant.ScoredPoint, error) {
    if len(vector) == 0 {
        return nil, fmt.Errorf("empty vector provided for similarity search")
    }

    // Create a slice of point IDs to exclude
    pointIds := make([]*qdrant.PointId, len(excludeIDs))
    for i, id := range excludeIDs {
        pointIds[i] = qdrant.NewIDNum(uint64(id))
    }

    // Create the filter with the must_not condition for the excluded IDs
    var filter *qdrant.Filter
    if len(excludeIDs) > 0 {
        filter = &qdrant.Filter{
            MustNot: []*qdrant.Condition{
                {
                    ConditionOneOf: &qdrant.Condition_HasId{
                        HasId: &qdrant.HasIdCondition{
                            HasId: pointIds,
                        },
                    },
                },
            },
        }
    }

    // Prepare the query request with the dense vector
    queryRequest := &qdrant.QueryPoints{
        CollectionName: collectionName,
        Query:         qdrant.NewQueryDense(vector),
        Limit:         &limit,
        Filter:        filter,
        WithPayload:   qdrant.NewWithPayload(true),
        WithVectors:   qdrant.NewWithVectors(false), // We only need the IDs, not the vectors
    }

    // Execute the query
    result, err := s.QdrantClient.Query(ctx, queryRequest)
    if err != nil {
        return nil, fmt.Errorf("failed to search similar vectors: %w", err)
    }

    return result, nil
}

// Helper function to query similar users by vector, excluding a single UserID (string)
func querySimilarUsersByVector(
	ctx context.Context,
	qdrantClient *qdrant.Client,
	collectionName string,
	vector []float32,
	limit uint64,
	excludeUserID string,
) ([]*qdrant.ScoredPoint, error) {
	if len(vector) == 0 {
		return nil, fmt.Errorf("empty vector provided for user similarity search")
	}

	start := time.Now()
	fmt.Printf("Starting user similarity search in %s (excluding user %s)\n", 
		collectionName, excludeUserID)

	var mustNotConditions []*qdrant.Condition
	if excludeUserID != "" {
		condition := qdrant.NewMatchText("id", excludeUserID)
		mustNotConditions = append(mustNotConditions, condition)
	}

	filter := &qdrant.Filter{}
	if len(mustNotConditions) > 0 {
		filter.MustNot = mustNotConditions
	}

	queryPoints := &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query:         qdrant.NewQueryDense(vector),
		Limit:         &limit,
		Filter:        filter,
		WithPayload:   qdrant.NewWithPayload(true),
		WithVectors:   qdrant.NewWithVectors(true),
	}

	similarPoints, err := qdrantClient.Query(ctx, queryPoints)
	if err != nil {
		return nil, fmt.Errorf("qdrant query failed for user search in %s: %v", collectionName, err)
	}

	duration := time.Since(start)
	fmt.Printf("User similarity search completed in %v with %d results\n", 
		duration, len(similarPoints))

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

    var embeddings [][]float32
    for _, gid := range goodreadsIDs {
        embedding, err := getEmbeddingByGoodreadsID(ctx, qdrantClient, gid, collectionName)
        if err != nil {
            fmt.Printf("Warning: could not get embedding for Goodreads ID %d: %v\n", gid, err)
            continue
        }
        embeddings = append(embeddings, embedding)
    }

    if len(embeddings) == 0 {
        return nil, fmt.Errorf("no embeddings found for the provided Goodreads IDs")
    }

    embeddingDim := len(embeddings[0])
    avgEmbedding := make([]float32, embeddingDim)

    for _, embedding := range embeddings {
        if len(embedding) != embeddingDim {
            return nil, fmt.Errorf("inconsistent embedding dimensions")
        }
        for i, val := range embedding {
            avgEmbedding[i] += val
        }
    }

    for i := range avgEmbedding {
        avgEmbedding[i] /= float32(len(embeddings))
    }

    service := &RecommendationService{QdrantClient: *qdrantClient}
    return service.querySimilarByVectorWithExclusion(ctx, avgEmbedding, goodreadsIDs, limit, collectionName)
}

// Functions for reranking
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
	if unionSize == 0 {
		return 0.0
	}
	return float64(intersectionSize) / float64(unionSize)
}

func mmrDiversify(candidatesToRank []RerankerBookInfo, k int, lambdaParam float64) []int64 {
	if len(candidatesToRank) == 0 || k <= 0 {
		return []int64{}
	}
	if k > len(candidatesToRank) {
		k = len(candidatesToRank)
	}

	// Initialize pool and select first candidate
	pool := make(map[int64]RerankerBookInfo)
	for _, c := range candidatesToRank {
		pool[c.GoodreadsID] = c
	}

	var firstCandidate RerankerBookInfo
	firstCandidateFound := false
	for _, candidate := range candidatesToRank {
		if !firstCandidateFound || candidate.AdjustedScore > firstCandidate.AdjustedScore {
			firstCandidate = candidate
			firstCandidateFound = true
		}
	}

	if !firstCandidateFound {
		return []int64{}
	}

	selectedIDs := []int64{firstCandidate.GoodreadsID}
	selectedRepresentations := []map[string]struct{}{firstCandidate.Representation}
	delete(pool, firstCandidate.GoodreadsID)

	// MMR loop
	for len(selectedIDs) < k && len(pool) > 0 {
		bestNextID := int64(-1)
		maxMMRScore := -1.0 * math.MaxFloat64

		for _, candidateInPool := range pool {
			relevance := float64(candidateInPool.AdjustedScore)
			maxSimToSelected := 0.0

			if len(selectedRepresentations) > 0 {
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
			break
		}
	}
	return selectedIDs
}

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
		pubYear = int32(bookRow.PublicationYear.Int64)
	}

	return RerankerBookInfo{
		GoodreadsID:     bookRow.GoodreadsID,
		PublicationYear: pubYear,
		Genres:          genres,
		Authors:         authors,
		Representation:  representation,
		InitialScore:    initialScore,
		AdjustedScore:   initialScore,
	}
}

func applyRerankingLogic(
	ctx context.Context,
	dbQueries *db.Queries,
	initialRawCandidates []InitialCandidate,
	userHistoryGoodreadsIDs map[int64]struct{},
	finalK int,
	lambdaParam float64,
	pubYearBoostFactor float64,
	pubYearBoostBaseYear int,
) ([]int64, error) {
	if len(initialRawCandidates) == 0 || finalK <= 0 {
		return []int64{}, nil
	}

	// Filter out user history and prepare for metadata fetching
	candidateIDsToFetchMeta := make([]int64, 0, len(initialRawCandidates))
	scoresMap := make(map[int64]float32)

	for _, ic := range initialRawCandidates {
		if _, existsInHistory := userHistoryGoodreadsIDs[ic.GoodreadsID]; !existsInHistory {
			candidateIDsToFetchMeta = append(candidateIDsToFetchMeta, ic.GoodreadsID)
			scoresMap[ic.GoodreadsID] = ic.Score
		}
	}

	if len(candidateIDsToFetchMeta) == 0 {
		return []int64{}, nil
	}

	// Fetch book metadata for filtered candidates
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

	// Create RerankerBookInfo and apply score adjustments
	rerankerCandidates := make([]RerankerBookInfo, 0, len(dbBookRows))
	for _, bookRow := range dbBookRows {
		initialScore, ok := scoresMap[bookRow.GoodreadsID]
		if !ok {
			fmt.Printf("Warning: book %d metadata fetched but no initial score found. Skipping.\n", bookRow.GoodreadsID)
			continue
		}

		info := newRerankerBookInfo(bookRow, initialScore)
		if info.PublicationYear > 0 {
			boost := float32(pubYearBoostFactor * float64(int(info.PublicationYear)-pubYearBoostBaseYear))
			info.AdjustedScore += boost
		}
		rerankerCandidates = append(rerankerCandidates, info)
	}

	if len(rerankerCandidates) == 0 {
		return []int64{}, nil
	}

	// Sort candidates by adjusted score
	sort.Slice(rerankerCandidates, func(i, j int) bool {
		return rerankerCandidates[i].AdjustedScore > rerankerCandidates[j].AdjustedScore
	})

	// Diversify via MMR
	rerankedIDs := mmrDiversify(rerankerCandidates, finalK, lambdaParam)
	return rerankedIDs, nil
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
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    ctx := c.Request.Context()
    inputGoodreadsIDs := make([]int64, 0, len(req.LikedBookIds))
    userHistoryGoodreadsIDs := make(map[int64]struct{})

    for _, idStr := range req.LikedBookIds {
        bookUUID, err := uuid.Parse(idStr)
        if err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Invalid book UUID format: %s", idStr)})
            return
        }
        pgUUID := pgtype.UUID{Bytes: bookUUID, Valid: true}

        book, err := h.DB.GetBookByID(ctx, pgUUID)
        if err != nil {
            fmt.Printf("Warning: Could not fetch book details for UUID %s: %v\n", idStr, err)
            continue
        }
        inputGoodreadsIDs = append(inputGoodreadsIDs, book.GoodreadsID)
        userHistoryGoodreadsIDs[book.GoodreadsID] = struct{}{}
    }

    if len(inputGoodreadsIDs) == 0 {
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    recommendationTargetCount := 10
    initialCandidateQueryLimit := uint64(recommendationTargetCount * rerankerInitialCandidateMultiplier)
    qdrantCollection := "gmf_book_embeddings"

    var avgEmbedding []float32
    var err error

    if len(inputGoodreadsIDs) > 0 {
        var allEmbeddingsTemp [][]float32
        for _, gid := range inputGoodreadsIDs {
            embedding, embErr := getEmbeddingByGoodreadsID(ctx, h.QdrantClient, gid, qdrantCollection)
            if embErr != nil {
                fmt.Printf("Warning: could not get embedding for Goodreads ID %d for averaging: %v\n", gid, embErr)
                continue
            }
            allEmbeddingsTemp = append(allEmbeddingsTemp, embedding)
        }
        if len(allEmbeddingsTemp) > 0 {
            embeddingDim := len(allEmbeddingsTemp[0])
            avgEmbedding = make([]float32, embeddingDim)
            for _, embedding := range allEmbeddingsTemp {
                for i, val := range embedding {
                    avgEmbedding[i] += val
                }
            }
            for i := range avgEmbedding {
                avgEmbedding[i] /= float32(len(allEmbeddingsTemp))
            }
        } else {
            fmt.Println("Warning: No embeddings found for liked books to calculate average.")
            c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
            return
        }
    } else {
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    recommendService := &RecommendationService{QdrantClient: *h.QdrantClient}
    similarPoints, err := recommendService.querySimilarByVectorWithExclusion(ctx, avgEmbedding, inputGoodreadsIDs, initialCandidateQueryLimit, qdrantCollection)
    if err != nil {
        fmt.Printf("Error finding similar points for anonymous recs: %v\n", err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to calculate recommendations"})
        return
    }

    if len(similarPoints) == 0 {
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    initialCandidatesForReranker := make([]InitialCandidate, 0, len(similarPoints))
    for _, point := range similarPoints {
        gid := int64(point.Id.GetNum())
        if _, existsInLiked := userHistoryGoodreadsIDs[gid]; !existsInLiked {
            initialCandidatesForReranker = append(initialCandidatesForReranker, InitialCandidate{
                GoodreadsID: gid,
                Score:       point.Score,
            })
        }
    }

    if len(initialCandidatesForReranker) == 0 {
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    rerankedGoodreadsIDs, err := applyRerankingLogic(
        ctx,
        h.DB,
        initialCandidatesForReranker,
        userHistoryGoodreadsIDs,
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
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, rerankedGoodreadsIDs)
    if err != nil {
        fmt.Printf("Error fetching reranked recommended book details from DB: %v\n", err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
        return
    }

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

    c.JSON(http.StatusOK, orderedDbBooks)
}

// GetContentBasedRecommendations returns content-based recommendations for a single book (by its UUID)
func (h *Handler) GetContentBasedRecommendations(c *gin.Context) {
    bookUUIDStr := c.Param("id")
    ctx := c.Request.Context()

    bookUUID, err := uuid.Parse(bookUUIDStr)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book UUID format"})
        return
    }
    pgBookUUID := pgtype.UUID{Bytes: bookUUID, Valid: true}

    sourceBook, err := h.DB.GetBookByID(ctx, pgBookUUID)
    if err != nil {
        fmt.Printf("Error fetching source book %s for similar: %v\n", bookUUIDStr, err)
        c.JSON(http.StatusNotFound, gin.H{"error": "Source book not found"})
        return
    }
    sourceGoodreadsID := sourceBook.GoodreadsID

    qdrantCollection := "gmf_book_embeddings"
    sourceEmbedding, err := getEmbeddingByGoodreadsID(ctx, h.QdrantClient, sourceGoodreadsID, qdrantCollection)
    if err != nil {
        fmt.Printf("Error getting embedding for source book GoodreadsID %d: %v\n", sourceGoodreadsID, err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not retrieve embedding for source book"})
        return
    }

    recommendationTargetCount := 10
    initialCandidateQueryLimit := uint64(recommendationTargetCount * rerankerInitialCandidateMultiplier)

    userHistoryGoodreadsIDs := map[int64]struct{}{sourceGoodreadsID: {}}

    recommendService := &RecommendationService{QdrantClient: *h.QdrantClient}
    similarPoints, err := recommendService.querySimilarByVectorWithExclusion(ctx, sourceEmbedding, []int64{sourceGoodreadsID}, initialCandidateQueryLimit, qdrantCollection)
    if err != nil {
        fmt.Printf("Error querying Qdrant for similar books to %d: %v\n", sourceGoodreadsID, err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to find similar books"})
        return
    }

    if len(similarPoints) == 0 {
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    initialCandidatesForReranker := make([]InitialCandidate, 0, len(similarPoints))
    for _, point := range similarPoints {
        gid := int64(point.Id.GetNum())
        if gid != sourceGoodreadsID {
            initialCandidatesForReranker = append(initialCandidatesForReranker, InitialCandidate{
                GoodreadsID: gid,
                Score:       point.Score,
            })
        }
    }

    if len(initialCandidatesForReranker) == 0 {
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    rerankedGoodreadsIDs, err := applyRerankingLogic(
        ctx,
        h.DB,
        initialCandidatesForReranker,
        userHistoryGoodreadsIDs,
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
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, rerankedGoodreadsIDs)
    if err != nil {
        fmt.Printf("Error fetching recommended book details from DB: %v\n", err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
        return
    }

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

    c.JSON(http.StatusOK, orderedDbBooks)
}

// GetCollaborativeRecommendations returns collaborative filtering recommendations for a user
func (rs *RecommendationService) GetCollaborativeRecommendations(c *gin.Context) {
	userID := c.Param("userId")
	ctx := c.Request.Context()
	
	start := time.Now()
	fmt.Printf("Starting collaborative recommendations for user %s\n", userID)

	qdrantCollection := "gmf_user_embeddings"
	userPoints, err := rs.QdrantClient.Get(ctx, &qdrant.GetPoints{
		CollectionName: qdrantCollection,
		Ids: []*qdrant.PointId{qdrant.NewID(userID)},
		WithVectors: qdrant.NewWithVectors(true),
	})

	if err != nil {
		fmt.Printf("Error retrieving user embedding for %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve user embedding"})
		return
	}

	if len(userPoints) == 0 {
		fmt.Printf("No user embedding found for user %s\n", userID)
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found or no embedding available"})
		return
	}

	if userPoints[0].Vectors == nil || userPoints[0].Vectors.GetVector() == nil {
		fmt.Printf("Invalid vector data for user %s\n", userID)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "User embedding vector is nil"})
		return
	}

	userVector := userPoints[0].Vectors.GetVector().Data
	limit := uint64(10)
	bookCollection := "gmf_book_embeddings"

	// Use Query API with proper configuration
	similarBookPoints, err := rs.QdrantClient.Query(ctx, &qdrant.QueryPoints{
		CollectionName: bookCollection,
		Query:         qdrant.NewQueryDense(userVector),
		Limit:         &limit,
		WithPayload:   qdrant.NewWithPayload(true),
		WithVectors:   qdrant.NewWithVectors(true),
	})

	if err != nil {
		fmt.Printf("Error querying book recommendations for user %s: %v\n", userID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve book recommendations"})
		return
	}

	// Convert Qdrant results to response with Goodreads IDs extracted from UUIDs
	var processedResults []gin.H
	for _, point := range similarBookPoints {
		if point.Id != nil {
			idStr := point.Id.GetUuid()
			// Extract the numeric ID from the UUID's namespace
			u, err := uuid.Parse(idStr)
			if err != nil {
				fmt.Printf("Warning: Could not parse UUID %s: %v\n", idStr, err)
				continue
			}
			// The namespace should contain the original Goodreads ID
			gidStr := string(u.NodeID())
			gid, err := strconv.ParseInt(gidStr, 10, 64)
			if err != nil {
				fmt.Printf("Warning: Could not parse Goodreads ID from UUID %s: %v\n", idStr, err)
				continue
			}
			processedResults = append(processedResults, gin.H{
				"goodreads_id": gid,
				"score": point.Score,
			})
		}
	}

	duration := time.Since(start)
	fmt.Printf("Collaborative recommendations completed for user %s in %v with %d results\n",
		userID, duration, len(processedResults))

	c.JSON(http.StatusOK, processedResults)
}

// GetSimilarUsers finds users with similar taste to a given user.
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
func (h *Handler) GetBookRecommendationsForUser(c *gin.Context) {
    userID := c.Param("id")
    ctx := c.Request.Context()

    qdrantUserCollection := "gmf_user_embeddings"
    qdrantGmfBookCollection := "gmf_book_embeddings"
    qdrantSbertBookCollection := "sbert_embeddings"

    recommendationTargetCount := 10
    limitPerSourceBeforeRerank := recommendationTargetCount * rerankerInitialCandidateMultiplier / 2
    if limitPerSourceBeforeRerank < recommendationTargetCount {
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
    userHistoryGoodreadsIDs := make(map[int64]struct{})
    for _, libBookRow := range userLibraryDBRows {
        libraryGoodreadsIDs = append(libraryGoodreadsIDs, libBookRow.GoodreadsID)
        userHistoryGoodreadsIDs[libBookRow.GoodreadsID] = struct{}{}
    }

    allInitialCandidates := make([]InitialCandidate, 0)
    seenCandidateGIDs := make(map[int64]struct{})

    userEmbedding, err := getEmbeddingByUserID(ctx, h.QdrantClient, userID, qdrantUserCollection)
    if err != nil {
        fmt.Printf("Error getting GMF user embedding for user ID %s: %v\n", userID, err)
    } else {
        recommendService := &RecommendationService{QdrantClient: *h.QdrantClient}
        gmfSimilarBookPoints, err := recommendService.querySimilarByVectorWithExclusion(ctx, userEmbedding, libraryGoodreadsIDs, uint64(limitPerSourceBeforeRerank), qdrantGmfBookCollection)
        if err != nil {
            fmt.Printf("Error querying GMF book recommendations for user %s: %v\n", userID, err)
        } else {
            for _, point := range gmfSimilarBookPoints {
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

    if len(libraryGoodreadsIDs) > 0 {
        sbertSimilarBookPoints, err := getAverageEmbeddingsAndFindSimilarWithExclusion(ctx, h.QdrantClient, libraryGoodreadsIDs, qdrantSbertBookCollection, uint64(limitPerSourceBeforeRerank))
        if err != nil {
            fmt.Printf("Error getting SBERT recommendations based on library average for user %s: %v\n", userID, err)
        } else {
            for _, point := range sbertSimilarBookPoints {
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

    if len(allInitialCandidates) == 0 {
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    rerankedGoodreadsIDs, err := applyRerankingLogic(
        ctx,
        h.DB,
        allInitialCandidates,
        userHistoryGoodreadsIDs,
        recommendationTargetCount,
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
        c.JSON(http.StatusOK, []db.GetBooksByGoodreadsIDsRow{})
        return
    }

    dbBooks, err := h.DB.GetBooksByGoodreadsIDs(ctx, rerankedGoodreadsIDs)
    if err != nil {
        fmt.Printf("Error fetching combined recommended book details from DB for user %s: %v\n", userID, err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch recommended book details"})
        return
    }

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

    c.JSON(http.StatusOK, orderedDbBooks)
}

// GetHybridRecommendations combines both content-based and collaborative filtering recommendations
func (rs *RecommendationService) GetHybridRecommendations(c *gin.Context) {
    bookIDStr := c.Param("bookId")
    userID := c.Param("userId")

    bookGID, err := strconv.ParseInt(bookIDStr, 10, 64)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID for content part"})
        return
    }

    ctx := c.Request.Context()
    contentLimit := uint64(5)

    // Get book embedding first
    bookEmbedding, err := getEmbeddingByGoodreadsID(ctx, &rs.QdrantClient, bookGID, "sbert_embeddings")
    if err != nil {
        fmt.Printf("Error getting book embedding for Goodreads ID %d: %v\n", bookGID, err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get book embedding"})
        return
    }

    // Get content-based recommendations
    contentBasedRecs, err := rs.querySimilarByVectorWithExclusion(ctx, bookEmbedding, []int64{bookGID}, contentLimit, "sbert_embeddings")
    if err != nil {
        fmt.Printf("Error getting content-based recommendations: %v\n", err)
    }

    // Get collaborative recommendations
    userPoints, err := rs.QdrantClient.Get(ctx, &qdrant.GetPoints{
        CollectionName: "gmf_user_embeddings",
        Ids: []*qdrant.PointId{
            qdrant.NewID(userID),
        },
        WithVectors: qdrant.NewWithVectors(true),
    })

    var collaborativeRecs []*qdrant.ScoredPoint
    if err == nil && len(userPoints) > 0 && userPoints[0].Vectors != nil && userPoints[0].Vectors.GetVector() != nil {
        collabLimit := uint64(5)
        collaborativeRecs, err = rs.querySimilarByVectorWithExclusion(ctx, userPoints[0].Vectors.GetVector().Data, []int64{bookGID}, collabLimit, "gmf_book_embeddings")
        if err != nil {
            fmt.Printf("Error getting collaborative recommendations: %v\n", err)
        }
    }

    response := gin.H{
        "content_based_raw": contentBasedRecs,
        "collaborative_raw": collaborativeRecs,
    }
    c.JSON(http.StatusOK, response)
}
