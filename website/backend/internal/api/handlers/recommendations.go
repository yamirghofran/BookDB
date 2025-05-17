package handlers

import (
	"context"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/qdrant/go-client/qdrant"
)

type RecommendationService struct {
	QdrantClient qdrant.Client
}

// GetAnonymousRecommendations returns book recommendations based on provided book IDs
func (h *Handler) GetAnonymousRecommendations(c *gin.Context) {
	var req struct {
		LikedBookIds []string `json:"likedBookIds" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	recommendations := []gin.H{
		{
			"id":    "1",
			"title": "Recommended Book 1",
			"score": 0.95,
		},
		{
			"id":    "2",
			"title": "Recommended Book 2",
			"score": 0.85,
		},
	}

	c.JSON(http.StatusOK, recommendations)
}

// GetContentBasedRecommendations returns content-based recommendations for a book
func (rs *RecommendationService) GetContentBasedRecommendations(c *gin.Context) {
	bookID := c.Param("bookId")

	// Convert bookID to int
	var bookIDInt int64
	if _, err := fmt.Sscanf(bookID, "%d", &bookIDInt); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID"})
		return
	}

	// Query nearest by ID
	similarPoints, err := rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "sbert_embeddings",
		Query:          qdrant.NewQueryID(qdrant.NewIDNum(13)),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve similar books"})
		return
	}

	c.JSON(http.StatusOK, similarPoints)
}

// GetCollaborativeRecommendations returns collaborative filtering recommendations for a user
func (rs *RecommendationService) GetCollaborativeRecommendations(c *gin.Context) {
	userID := c.Param("userId")

	// Get user embedding
	userPoints, err := rs.QdrantClient.Get(context.Background(), &qdrant.GetPoints{
		CollectionName: "gmf_user_embeddings",
		Ids: []*qdrant.PointId{
			qdrant.NewID(userID),
		},
		WithVectors: qdrant.NewWithVectors(true),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve user embedding"})
		return
	}

	if len(userPoints) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
		return
	}

	// Find similar books based on user embedding
	limit := uint64(10)
	similarBookPoints, err := rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "gmf_book_embeddings",
		Query:          qdrant.NewQueryDense(userPoints[0].Vectors.GetVector().Data),
		Limit:          &limit,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve book recommendations"})
		return
	}

	c.JSON(http.StatusOK, similarBookPoints)
}

// GetHybridRecommendations combines both content-based and collaborative filtering recommendations
func (rs *RecommendationService) GetHybridRecommendations(c *gin.Context) {
	bookID := c.Param("bookId")
	userID := c.Param("userId")

	// Convert bookID to int
	var bookIDInt int64
	if _, err := fmt.Sscanf(bookID, "%d", &bookIDInt); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID"})
		return
	}

	// Get the embedding for the specified book
	scrollResult, err := rs.QdrantClient.Scroll(context.Background(), &qdrant.ScrollPoints{
		CollectionName: "sbert_embeddings",
		Filter: &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatchInt("book_id", bookIDInt),
			},
		},
	})

	if err != nil || len(scrollResult) == 0 {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve book embedding"})
		return
	}

	// Get user embedding
	userPoints, err := rs.QdrantClient.Get(context.Background(), &qdrant.GetPoints{
		CollectionName: "gmf_user_embeddings",
		Ids: []*qdrant.PointId{
			qdrant.NewID(userID),
		},
		WithVectors: qdrant.NewWithVectors(true),
	})

	if err != nil || len(userPoints) == 0 {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve user embedding"})
		return
	}

	// Get content-based recommendations
	contentLimit := uint64(5)
	contentBasedRecs, err := rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "sbert_embeddings",
		Query:          qdrant.NewQueryID(scrollResult[0].Id),
		Limit:          &contentLimit,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve content-based recommendations"})
		return
	}

	// Get collaborative filtering recommendations
	collabLimit := uint64(5)
	collaborativeRecs, err := rs.QdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "gmf_book_embeddings",
		Query:          qdrant.NewQueryDense(userPoints[0].Vectors.GetVector().Data),
		Limit:          &collabLimit,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve collaborative recommendations"})
		return
	}

	// Combine recommendations
	response := gin.H{
		"content_based": contentBasedRecs,
		"collaborative": collaborativeRecs,
	}

	c.JSON(http.StatusOK, response)
}
