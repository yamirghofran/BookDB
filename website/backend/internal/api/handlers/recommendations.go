package handlers

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/yamirghofran/BookDB/internal/db"
)

// GetSimilarBooks returns books similar to a given book ID
func (h *Handler) GetSimilarBooks(c *gin.Context) {
	idStr := c.Param("id")

	// Parse UUID
	bookID, err := uuid.Parse(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID format"})
		return
	}

	// Convert to pgtype.UUID
	pgUUID := pgtype.UUID{
		Bytes: bookID,
		Valid: true,
	}

	// Get the book to make sure it exists
	_, err = h.DB.GetBookByID(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Book not found"})
		return
	}

	// For now, return a placeholder response
	// In a real implementation, you would use the Qdrant client to find similar books
	c.JSON(http.StatusOK, gin.H{
		"book_id": idStr,
		"similar_books": []gin.H{
			{"id": "1", "title": "Similar Book 1", "similarity": 0.95},
			{"id": "2", "title": "Similar Book 2", "similarity": 0.85},
			{"id": "3", "title": "Similar Book 3", "similarity": 0.75},
		},
	})
}

// GetRecommendationsForUser returns book recommendations for a user
func (h *Handler) GetRecommendationsForUser(c *gin.Context) {
	idStr := c.Param("id")

	// Parse UUID
	userID, err := uuid.Parse(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid user ID format"})
		return
	}

	// Convert to pgtype.UUID
	pgUUID := pgtype.UUID{
		Bytes: userID,
		Valid: true,
	}

	// Check if user exists
	_, err = h.DB.GetUserByID(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
		return
	}

	// For now, return a placeholder response
	// In a real implementation, you would analyze user's reading history and preferences
	c.JSON(http.StatusOK, gin.H{
		"user_id": idStr,
		"recommendations": []gin.H{
			{"id": "1", "title": "Recommended Book 1", "confidence": 0.95},
			{"id": "2", "title": "Recommended Book 2", "confidence": 0.85},
			{"id": "3", "title": "Recommended Book 3", "confidence": 0.75},
		},
	})
}

// GetTrendingBooks returns trending books based on popularity metrics
func (h *Handler) GetTrendingBooks(c *gin.Context) {
	limit := 10

	if limitParam := c.Query("limit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	// For now, just return the top rated books
	// In a real implementation, you might consider factors like recent views, purchases, etc.
	books, err := h.DB.ListBooks(c, db.ListBooksParams{
		Limit:  int32(limit),
		Offset: 0,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"trending_books": books,
	})
}
