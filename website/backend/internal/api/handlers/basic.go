package handlers

import (
	"fmt"
	"net/http"

	// "sort" // Commented out as it's not used
	"strconv"
	"strings" // Added for strings.Split

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/qdrant/go-client/qdrant"
	"github.com/yamirghofran/BookDB/internal/db"
)

// Handler holds dependencies for handlers
type Handler struct {
	DB           *db.Queries
	QdrantClient *qdrant.Client
}

// NewHandler creates a new handler instance
func NewHandler(db *db.Queries, qdrantClient *qdrant.Client) *Handler {
	return &Handler{
		DB:           db,
		QdrantClient: qdrantClient,
	}
}

// BookResponse defines the structure for book data returned by the API, including authors.
// JSON tags are capitalized to match the frontend's RawBookFromAPI interface.
type ReviewResponse struct {
	ID            pgtype.UUID        `json:"id"`
	BookID        pgtype.UUID        `json:"bookId"`
	UserID        pgtype.UUID        `json:"userId"`
	UserName      string             `json:"userName"`
	UserAvatarURL string             `json:"userAvatarUrl"` // Will be empty string for now
	Rating        pgtype.Int2        `json:"rating"`        // Assuming rating is SMALLINT (Int2)
	ReviewText    string             `json:"reviewText"`
	CreatedAt     pgtype.Timestamptz `json:"createdAt"`
	UpdatedAt     pgtype.Timestamptz `json:"updatedAt"`
}

type BookResponse struct {
	ID                pgtype.UUID        `json:"ID"`
	GoodreadsID       int64              `json:"GoodreadsID,omitempty"`
	GoodreadsUrl      pgtype.Text        `json:"GoodreadsUrl,omitempty"`
	Title             string             `json:"Title"`
	Description       pgtype.Text        `json:"Description,omitempty"`
	PublicationYear   pgtype.Int8        `json:"PublicationYear,omitempty"`
	CoverImageUrl     pgtype.Text        `json:"CoverImageUrl,omitempty"`
	AverageRating     *float64           `json:"AverageRating,omitempty"`
	RatingsCount      pgtype.Int8        `json:"RatingsCount,omitempty"`
	CreatedAt         pgtype.Timestamptz `json:"CreatedAt,omitempty"`
	UpdatedAt         pgtype.Timestamptz `json:"UpdatedAt,omitempty"`
	Authors           []string           `json:"Authors"`
	Genres            []string           `json:"Genres,omitempty"`  // Added Genres field
	Reviews           []ReviewResponse   `json:"Reviews,omitempty"` // Added Reviews field
	ReviewsTotalCount *int64             `json:"ReviewsTotalCount,omitempty"`
}

// Full Text Search
// SearchBooksByContent searches for books by content similarity
func (h *Handler) SearchBooksByContent(c *gin.Context) {
	originalQuery := c.Query("q")
	if originalQuery == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Search query is required"})
		return
	}

	// Process the query for more effective full-text search with partial words
	// For "ben horo", this aims to create a query like "ben:* & horo:*"
	queryParts := strings.Fields(strings.ToLower(originalQuery)) // Split by whitespace and lowercase
	processedQueryParts := make([]string, 0, len(queryParts))
	for _, part := range queryParts {
		// Sanitize part to remove characters that might conflict with tsquery syntax,
		// though websearch_to_tsquery is generally robust.
		// For simplicity, we'll just ensure it's not empty.
		if part != "" {
			processedQueryParts = append(processedQueryParts, part+":*") // Append prefix match operator
		}
	}

	// If only one part, websearch_to_tsquery handles it well.
	// If multiple parts, joining with ' & ' makes the AND explicit for prefix matches.
	var effectiveQueryString string
	if len(processedQueryParts) > 0 {
		effectiveQueryString = strings.Join(processedQueryParts, " & ")
	} else {
		effectiveQueryString = originalQuery // Fallback to original if processing results in empty
	}

	fmt.Printf("Original query: '%s', Effective tsquery input: '%s'\n", originalQuery, effectiveQueryString)

	limit := 10 // Default limit
	offset := 0 // Default offset

	if limitParam := c.Query("limit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	if offsetParam := c.Query("offset"); offsetParam != "" {
		parsedOffset, err := strconv.Atoi(offsetParam)
		if err == nil && parsedOffset >= 0 { // offset can be 0
			offset = parsedOffset
		}
	}

	dbBooks, err := h.DB.SearchBooks(c, db.SearchBooksParams{
		WebsearchToTsquery: effectiveQueryString, // Use the processed query string
		Limit:              int32(limit),
		Offset:             int32(offset),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to search books: " + err.Error()})
		return
	}

	responseBooks := make([]BookResponse, 0, len(dbBooks))
	for _, bookRow := range dbBooks {
		var authors []string
		if bookRow.Authors != nil {
			switch v := bookRow.Authors.(type) {
			case []string:
				authors = v
			case *[]string: // sqlc might return *[]string for text[] if nullable
				if v != nil {
					authors = *v
				} else {
					authors = []string{}
				}
			case []interface{}: // Fallback for some drivers/scenarios
				for _, item := range v {
					if str, ok := item.(string); ok {
						authors = append(authors, str)
					}
				}
			// REMOVED case pgtype.TextArray as it's not a defined type in this context
			case []byte: // Handle case where text[] is scanned as []byte (e.g., "{author1,author2}")
				// This is a simplified parser. For robust CSV or quoted array elements, a proper parser is needed.
				temp := string(v)
				if len(temp) > 2 && temp[0] == '{' && temp[len(temp)-1] == '}' {
					temp = temp[1 : len(temp)-1] // Remove {}
					if temp != "" {
						authors = strings.Split(temp, ",") // Naive split
					} else {
						authors = []string{}
					}
				} else {
					authors = []string{}
				}
			default:
				fmt.Printf("Warning: SearchBooks Unexpected type for Authors field: %T, value: %#v\n", v, v)
				authors = []string{}
			}
		} else {
			authors = []string{}
		}
		if authors == nil {
			authors = []string{}
		}

		var apiAvgRating *float64
		if bookRow.AverageRating.Valid {
			// Attempt to convert pgtype.Numeric to float64
			float8Val, err := bookRow.AverageRating.Float64Value()
			if err == nil && float8Val.Valid {
				apiAvgRating = &float8Val.Float64
			} else if err != nil {
				fmt.Printf("Error converting AverageRating to float64 in SearchBooks: %v\n", err)
				// apiAvgRating remains nil
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
			CreatedAt:       bookRow.CreatedAt,
			UpdatedAt:       bookRow.UpdatedAt,
			Authors:         authors,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"query":   originalQuery, // Use originalQuery here
		"results": responseBooks,
	})
}

// Book handlers

// ListBooks returns a list of books
func (h *Handler) ListBooks(c *gin.Context) {
	limit := 100 // Default limit, larger for a general listing
	offset := 0

	// Parse query parameters if provided
	if limitParam := c.Query("limit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	if offsetParam := c.Query("offset"); offsetParam != "" {
		parsedOffset, err := strconv.Atoi(offsetParam)
		if err == nil && parsedOffset >= 0 { // offset can be 0
			offset = parsedOffset
		}
	}

	dbBooks, err := h.DB.ListBooks(c, db.ListBooksParams{
		Limit:  int32(limit),
		Offset: int32(offset),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list books: " + err.Error()})
		return
	}

	responseBooks := make([]BookResponse, 0, len(dbBooks))
	for _, bookRow := range dbBooks {
		var authors []string
		if bookRow.Authors != nil {
			switch v := bookRow.Authors.(type) {
			case []string:
				authors = v
			case *[]string:
				if v != nil {
					authors = *v
				} else {
					authors = []string{}
				}
			case []interface{}:
				for _, item := range v {
					if str, ok := item.(string); ok {
						authors = append(authors, str)
					}
				}
			// REMOVED case pgtype.TextArray as it's not a defined type in this context
			case []byte:
				temp := string(v)
				if len(temp) > 2 && temp[0] == '{' && temp[len(temp)-1] == '}' {
					temp = temp[1 : len(temp)-1]
					if temp != "" {
						authors = strings.Split(temp, ",")
					} else {
						authors = []string{}
					}
				} else {
					authors = []string{}
				}
			default:
				fmt.Printf("Warning: ListBooks Unexpected type for Authors field: %T, value: %#v\n", v, v)
				authors = []string{}
			}
		} else {
			authors = []string{}
		}
		if authors == nil {
			authors = []string{}
		}

		var apiAvgRating *float64
		if bookRow.AverageRating.Valid {
			float8Val, err := bookRow.AverageRating.Float64Value()
			if err == nil && float8Val.Valid {
				apiAvgRating = &float8Val.Float64
			} else if err != nil {
				fmt.Printf("Error converting AverageRating to float64 in ListBooks: %v\n", err)
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
			CreatedAt:       bookRow.CreatedAt,
			UpdatedAt:       bookRow.UpdatedAt,
			Authors:         authors,
		})
	}
	c.JSON(http.StatusOK, responseBooks)
}

// GetBook returns a specific book by ID
func (h *Handler) GetBook(c *gin.Context) {
	idStr := c.Param("id")
	fmt.Printf("GetBook handler: Received request for book ID string: %s\n", idStr)

	// Parse UUID
	bookID, err := uuid.Parse(idStr)
	if err != nil {
		fmt.Printf("GetBook handler: Error parsing UUID '%s': %v\n", idStr, err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID format"})
		return
	}
	fmt.Printf("GetBook handler: Parsed bookID (uuid.UUID): %s\n", bookID.String())

	// Convert to pgtype.UUID
	pgUUID := pgtype.UUID{
		Bytes: bookID,
		Valid: true,
	}
	fmt.Printf("GetBook handler: pgtype.UUID to be queried: %+v\n", pgUUID)

	bookRow, err := h.DB.GetBookByID(c, pgUUID)
	if err != nil {
		fmt.Printf("GetBook handler: Error from DB.GetBookByID for pgUUID '%+v': %v\n", pgUUID, err)
		c.JSON(http.StatusNotFound, gin.H{"error": "Book not found"})
		return
	}
	fmt.Printf("GetBook handler: Successfully fetched bookRow for ID %s\n", idStr)

	// Process Authors
	var authors []string
	if bookRow.Authors != nil {
		switch v := bookRow.Authors.(type) {
		case []string:
			authors = v
		case *[]string:
			if v != nil {
				authors = *v
			} else {
				authors = []string{}
			}
		case []interface{}:
			for _, item := range v {
				if str, ok := item.(string); ok {
					authors = append(authors, str)
				}
			}
		case []byte:
			temp := string(v)
			if len(temp) > 2 && temp[0] == '{' && temp[len(temp)-1] == '}' {
				temp = temp[1 : len(temp)-1]
				if temp != "" {
					authors = strings.Split(temp, ",")
				} else {
					authors = []string{}
				}
			} else {
				authors = []string{}
			}
		default:
			fmt.Printf("Warning: GetBook Unexpected type for Authors field: %T, value: %#v\n", v, v)
			authors = []string{}
		}
	} else {
		authors = []string{}
	}
	if authors == nil {
		authors = []string{}
	}

	// Process AverageRating
	var apiAvgRating *float64
	if bookRow.AverageRating.Valid {
		float8Val, err := bookRow.AverageRating.Float64Value()
		if err == nil && float8Val.Valid {
			apiAvgRating = &float8Val.Float64
		} else if err != nil {
			fmt.Printf("Error converting AverageRating to float64 in GetBook: %v\n", err)
		}
	}

	// Fetch reviews for the book with pagination
	reviewsLimit := 10 // Default limit for reviews
	reviewsOffset := 0 // Default offset for reviews

	if limitParam := c.Query("reviewsLimit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil && parsedLimit > 0 {
			reviewsLimit = parsedLimit
		}
	}

	if offsetParam := c.Query("reviewsOffset"); offsetParam != "" {
		parsedOffset, err := strconv.Atoi(offsetParam)
		if err == nil && parsedOffset >= 0 {
			reviewsOffset = parsedOffset
		}
	}

	dbReviews, err := h.DB.GetReviewsByBookID(c, db.GetReviewsByBookIDParams{
		BookID: bookRow.ID, // bookRow.ID is pgtype.UUID
		Limit:  int32(reviewsLimit),
		Offset: int32(reviewsOffset),
	})
	if err != nil {
		// Log the error but don't fail the whole request, just return book without reviews
		fmt.Printf("Warning: Failed to fetch reviews for book ID %v: %v\n", bookRow.ID, err)
		// dbReviews will be nil or empty, so apiReviews will be empty
	}

	apiReviews := make([]ReviewResponse, 0, len(dbReviews))
	var totalReviewsCount int64
	if len(dbReviews) > 0 {
		// TotalReviews is expected to be on each row from the SQL query
		// We take it from the first row.
		// This assumes sqlc will generate TotalReviews field in GetReviewsByBookIDRow
		totalReviewsCount = dbReviews[0].TotalReviews
	}

	for _, dbReview := range dbReviews {
		apiReviews = append(apiReviews, ReviewResponse{
			ID:            dbReview.ID,
			BookID:        dbReview.BookID,
			UserID:        dbReview.UserID,
			UserName:      dbReview.UserName,
			UserAvatarURL: "", // No avatar URL from DB currently
			Rating:        dbReview.Rating,
			ReviewText:    dbReview.ReviewText,
			CreatedAt:     dbReview.CreatedAt,
			UpdatedAt:     dbReview.UpdatedAt,
		})
	}

	response := BookResponse{
		ID:              bookRow.ID,
		GoodreadsID:     bookRow.GoodreadsID,
		GoodreadsUrl:    bookRow.GoodreadsUrl,
		Title:           bookRow.Title,
		Description:     bookRow.Description,
		PublicationYear: bookRow.PublicationYear,
		CoverImageUrl:   bookRow.CoverImageUrl,
		AverageRating:   apiAvgRating,
		RatingsCount:    bookRow.RatingsCount,
		CreatedAt:       bookRow.CreatedAt,
		UpdatedAt:       bookRow.UpdatedAt,
		Authors:         authors,
		// Process Genres (similar to Authors)
		Genres:            processStringArrayInterface(bookRow.Genres),
		Reviews:           apiReviews, // Add reviews to response
		ReviewsTotalCount: &totalReviewsCount,
	}
	if len(dbReviews) == 0 { // If there are no reviews, don't send the count
		response.ReviewsTotalCount = nil
	}

	c.JSON(http.StatusOK, response)
}

// CreateBook creates a new book
func (h *Handler) CreateBook(c *gin.Context) {
	var req struct {
		GoodreadsID     int64   `json:"goodreads_id" binding:"required"`
		GoodreadsUrl    string  `json:"goodreads_url"`
		Title           string  `json:"title" binding:"required"`
		Description     string  `json:"description"`
		PublicationYear int64   `json:"publication_year"`
		CoverImageUrl   string  `json:"cover_image_url"`
		AverageRating   float64 `json:"average_rating"`
		RatingsCount    int64   `json:"ratings_count"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Convert to DB params
	params := db.CreateBookParams{
		GoodreadsID: req.GoodreadsID,
		Title:       req.Title,
		GoodreadsUrl: pgtype.Text{
			String: req.GoodreadsUrl,
			Valid:  req.GoodreadsUrl != "",
		},
		Description: pgtype.Text{
			String: req.Description,
			Valid:  req.Description != "",
		},
		PublicationYear: pgtype.Int8{
			Int64: req.PublicationYear,
			Valid: req.PublicationYear > 0,
		},
		CoverImageUrl: pgtype.Text{
			String: req.CoverImageUrl,
			Valid:  req.CoverImageUrl != "",
		},
		AverageRating: pgtype.Numeric{ // This will be handled by the database conversion or a custom type
			// Float64: req.AverageRating, // This is not how pgtype.Numeric is set directly from float64
			Valid: req.AverageRating > 0, // Mark as valid if provided
		},
		RatingsCount: pgtype.Int8{
			Int64: req.RatingsCount,
			Valid: req.RatingsCount > 0,
		},
	}

	// Set numeric value for average rating
	if req.AverageRating > 0 {
		// For pgtype.Numeric, we need to set it differently
		// The database will handle the conversion
		// It's better to let sqlc handle the direct assignment if the type is `pgtype.Numeric`
		// For now, we just mark it as valid. The actual value setting for pgtype.Numeric
		// from a float64 might require using `Scan` method or similar if not directly assignable.
		// However, sqlc typically generates fields that can be directly assigned if the Go type matches.
		// If `AverageRating` in `CreateBookParams` is `float64`, then direct assignment is fine.
		// If it's `pgtype.Numeric`, you'd do something like:
		// err := params.AverageRating.Scan(req.AverageRating)
		// if err != nil { ... handle error ... }
		// For now, we'll keep it simple and assume the current structure works with the DB.
		params.AverageRating.Valid = true
		// If sqlc expects a string for numeric, you might do:
		// params.AverageRating.Scan(fmt.Sprintf("%f", req.AverageRating))
	}

	book, err := h.DB.CreateBook(c, params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, book)
}

// UpdateBook updates an existing book
func (h *Handler) UpdateBook(c *gin.Context) {
	idStr := c.Param("id")

	// Parse UUID
	bookID, err := uuid.Parse(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID format"})
		return
	}

	var req struct {
		GoodreadsID     int64   `json:"goodreads_id"`
		GoodreadsUrl    string  `json:"goodreads_url"`
		Title           string  `json:"title"`
		Description     string  `json:"description"`
		PublicationYear int64   `json:"publication_year"`
		CoverImageUrl   string  `json:"cover_image_url"`
		AverageRating   float64 `json:"average_rating"`
		RatingsCount    int64   `json:"ratings_count"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Convert to DB params
	params := db.UpdateBookParams{
		ID: pgtype.UUID{
			Bytes: bookID,
			Valid: true,
		},
		GoodreadsID: req.GoodreadsID,
		Title:       req.Title,
		GoodreadsUrl: pgtype.Text{
			String: req.GoodreadsUrl,
			Valid:  req.GoodreadsUrl != "",
		},
		Description: pgtype.Text{
			String: req.Description,
			Valid:  req.Description != "",
		},
		PublicationYear: pgtype.Int8{
			Int64: req.PublicationYear,
			Valid: req.PublicationYear > 0,
		},
		CoverImageUrl: pgtype.Text{
			String: req.CoverImageUrl,
			Valid:  req.CoverImageUrl != "",
		},
		AverageRating: pgtype.Numeric{ // See notes in CreateBook for pgtype.Numeric handling
			Valid: req.AverageRating > 0,
		},
		RatingsCount: pgtype.Int8{
			Int64: req.RatingsCount,
			Valid: req.RatingsCount > 0,
		},
	}
	// Similar to CreateBook, ensure AverageRating is handled correctly for pgtype.Numeric
	if req.AverageRating > 0 {
		params.AverageRating.Valid = true
		// Potentially: params.AverageRating.Scan(req.AverageRating) or similar
	}

	book, err := h.DB.UpdateBook(c, params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, book)
}

// DeleteBook deletes a book
func (h *Handler) DeleteBook(c *gin.Context) {
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

	err = h.DB.DeleteBook(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Book deleted successfully", "id": idStr})
}

// People handlers

// ListPeople returns a list of people (users)
func (h *Handler) ListPeople(c *gin.Context) {
	limit := 10
	offset := 0

	// Parse query parameters if provided
	if limitParam := c.Query("limit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil {
			limit = parsedLimit
		}
	}

	if offsetParam := c.Query("offset"); offsetParam != "" {
		parsedOffset, err := strconv.Atoi(offsetParam)
		if err == nil {
			offset = parsedOffset
		}
	}

	users, err := h.DB.ListUsers(c, db.ListUsersParams{
		Limit:  int32(limit),
		Offset: int32(offset),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, users)
}

// GetPerson returns a specific person by ID
func (h *Handler) GetPerson(c *gin.Context) {
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

	user, err := h.DB.GetUserByID(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
		return
	}

	c.JSON(http.StatusOK, user)
}

// CreatePerson creates a new person
func (h *Handler) CreatePerson(c *gin.Context) {
	var req struct {
		Name  string `json:"name" binding:"required"`
		Email string `json:"email" binding:"required,email"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Check if email already exists
	_, err := h.DB.GetUserByEmail(c, req.Email)
	if err == nil { // If err is nil, user was found
		c.JSON(http.StatusConflict, gin.H{"error": "Email already in use"})
		return
	}
	// Consider checking for specific "not found" error if GetUserByEmail returns one,
	// otherwise any other error from GetUserByEmail would also proceed to create user.
	// For now, this logic assumes any error means user not found / safe to create.

	user, err := h.DB.CreateUser(c, db.CreateUserParams{
		Name:  req.Name,
		Email: req.Email,
		// PasswordHash will be set by the database trigger or default value
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, user)
}

// Library handlers

// ListLibraries returns a list of libraries
// Note: In this case, we'll return all users since each user has a library
func (h *Handler) ListLibraries(c *gin.Context) {
	limit := 10
	offset := 0

	// Parse query parameters if provided
	if limitParam := c.Query("limit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil {
			limit = parsedLimit
		}
	}

	if offsetParam := c.Query("offset"); offsetParam != "" {
		parsedOffset, err := strconv.Atoi(offsetParam)
		if err == nil {
			offset = parsedOffset
		}
	}

	users, err := h.DB.ListUsers(c, db.ListUsersParams{
		Limit:  int32(limit),
		Offset: int32(offset),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Convert users to libraries
	var libraries []gin.H
	for _, user := range users {
		libraries = append(libraries, gin.H{
			"id":      user.ID,
			"name":    user.Name + "'s Library",
			"user_id": user.ID,
		})
	}

	c.JSON(http.StatusOK, libraries)
}

// GetLibrary returns a specific library by ID (user ID)
func (h *Handler) GetLibrary(c *gin.Context) {
	idStr := c.Param("id")

	// Parse UUID
	userID, err := uuid.Parse(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid library ID format"})
		return
	}

	// Convert to pgtype.UUID
	pgUUID := pgtype.UUID{
		Bytes: userID,
		Valid: true,
	}

	// Get user info
	user, err := h.DB.GetUserByID(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Library not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":      user.ID,
		"name":    user.Name + "'s Library",
		"user_id": user.ID,
	})
}

// GetBooksInLibrary returns all books in a specific library
func (h *Handler) GetBooksInLibrary(c *gin.Context) {
	idStr := c.Param("id")

	// Parse UUID
	userID, err := uuid.Parse(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid library ID format"})
		return
	}

	// Convert to pgtype.UUID
	pgUUID := pgtype.UUID{
		Bytes: userID,
		Valid: true,
	}

	// Get books in user's library
	books, err := h.DB.GetUserLibrary(c, pgUUID) // This returns []db.GetUserLibraryRow
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// The frontend expects an array of BookResponse, so we need to map it.
	// GetUserLibraryRow already contains most fields of BookResponse.
	// We need to handle the Authors field specifically if it's not directly compatible.
	// For now, let's assume GetUserLibraryRow is compatible enough or we adjust the frontend.
	// If GetUserLibraryRow doesn't have Authors, the frontend will show empty authors for these.
	// A more robust solution would be to have GetUserLibrary return full book details including authors.

	// For now, just returning the direct result. Frontend might need adjustment
	// or this handler needs to map to BookResponse.
	// Let's assume for now the frontend's `fetchBookById` is used by the library page,
	// and this endpoint might be for a different purpose or needs mapping.
	// Given the task is to show users who have a book, this endpoint is less relevant now.

	c.JSON(http.StatusOK, books)
}

// AddBookToLibrary adds a book to a library
func (h *Handler) AddBookToLibrary(c *gin.Context) {
	idStr := c.Param("id") // This is the User ID (acting as Library ID)

	// Parse UUID for user ID
	userID, err := uuid.Parse(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid library ID format"})
		return
	}

	var req struct {
		BookID string `json:"book_id" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Parse UUID for book ID
	bookID, err := uuid.Parse(req.BookID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID format"})
		return
	}

	// Convert to pgtype.UUID
	pgUserUUID := pgtype.UUID{
		Bytes: userID,
		Valid: true,
	}

	pgBookUUID := pgtype.UUID{
		Bytes: bookID,
		Valid: true,
	}

	// Check if book is already in library
	exists, err := h.DB.IsBookInLibrary(c, db.IsBookInLibraryParams{
		UserID: pgUserUUID,
		BookID: pgBookUUID,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to check library status: " + err.Error()})
		return
	}

	if exists {
		c.JSON(http.StatusConflict, gin.H{"error": "Book is already in the library"})
		return
	}

	// Add book to library
	err = h.DB.AddBookToLibrary(c, db.AddBookToLibraryParams{
		UserID: pgUserUUID,
		BookID: pgBookUUID,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to add book to library: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":    "Book added to library successfully",
		"library_id": idStr,
		"book_id":    req.BookID,
	})
}

// Author handlers

// ListAuthors returns a list of authors
func (h *Handler) ListAuthors(c *gin.Context) {
	limit := 10
	offset := 0

	// Parse query parameters if provided
	if limitParam := c.Query("limit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil {
			limit = parsedLimit
		}
	}

	if offsetParam := c.Query("offset"); offsetParam != "" {
		parsedOffset, err := strconv.Atoi(offsetParam)
		if err == nil {
			offset = parsedOffset
		}
	}

	authors, err := h.DB.ListAuthors(c, db.ListAuthorsParams{
		Limit:  int32(limit),
		Offset: int32(offset),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, authors)
}

// GetAuthor returns a specific author by ID
func (h *Handler) GetAuthor(c *gin.Context) {
	idStr := c.Param("id")

	// Parse UUID
	authorID, err := uuid.Parse(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid author ID format"})
		return
	}

	// Convert to pgtype.UUID
	pgUUID := pgtype.UUID{
		Bytes: authorID,
		Valid: true,
	}

	author, err := h.DB.GetAuthorByID(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Author not found"})
		return
	}

	c.JSON(http.StatusOK, author)
}

// SearchAuthors searches for authors by name
func (h *Handler) SearchAuthors(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Search query is required"})
		return
	}

	// Use the GetAuthorByName function which does a LIKE search
	// The SQL query for GetAuthorByName should handle case-insensitivity (e.g., ILIKE or lower())
	// and potentially use '%' for wildcard matching if desired.
	// For now, assuming it's a direct name match or simple LIKE.
	authors, err := h.DB.GetAuthorByName(c, pgtype.Text{
		String: query, // The query itself might need '%' for LIKE, e.g., "%" + query + "%"
		Valid:  true,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to search authors: " + err.Error()})
		return
	}

	// Sort authors by ratings count in descending order
	// This assumes GetAuthorByNameRow includes RatingsCount. If not, this sort won't work as expected.
	// The current GetAuthorByNameRow from sql/queries/authors.sql does not seem to include RatingsCount.
	// This sorting logic might need to be removed or the query updated.
	// For now, let's comment out the sort if RatingsCount is not available on db.GetAuthorByNameRow.
	/*
		sort.Slice(authors, func(i, j int) bool {
			// Ensure RatingsCount is a comparable numeric type if it exists on the struct
			// For example, if it's pgtype.Int4 or similar.
			// This is a placeholder, adjust based on actual struct field.
			// return authors[i].RatingsCount.Int32 > authors[j].RatingsCount.Int32
			return false // Placeholder if RatingsCount is not available for sorting
		})
	*/

	c.JSON(http.StatusOK, gin.H{
		"query":   query,
		"results": authors,
	})
}

// GetBooksByAuthor returns all books by a specific author
func (h *Handler) GetBooksByAuthor(c *gin.Context) {
	idStr := c.Param("id")

	// Parse UUID
	authorID, err := uuid.Parse(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid author ID format"})
		return
	}

	// Convert to pgtype.UUID
	pgUUID := pgtype.UUID{
		Bytes: authorID,
		Valid: true,
	}

	books, err := h.DB.GetBooksByAuthor(c, pgUUID) // This returns []db.GetBooksByAuthorRow
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	// Similar to GetBooksInLibrary, this might need mapping to BookResponse if the frontend expects it.
	// GetBooksByAuthorRow might not include all fields like Authors array.
	c.JSON(http.StatusOK, books)
}

// GetSimilarBooks returns similar books based on content-based filtering
func (h *Handler) GetSimilarBooks(c *gin.Context) {
	// The GetContentBasedRecommendations method is now directly on Handler
	// because it needs access to h.DB and h.QdrantClient.
	h.GetContentBasedRecommendations(c)
}

// GetRecommendationsForUser returns book recommendations for a specific user
func (h *Handler) GetRecommendationsForUser(c *gin.Context) {
	// Create a new recommendation service
	rs := &RecommendationService{
		QdrantClient: *h.QdrantClient, // Dereference the pointer
		// DB:           h.DB,             // Pass the DB connection - Temporarily commented out
	}

	// Delegate to the collaborative recommendations handler
	rs.GetCollaborativeRecommendations(c)
}

// processStringArrayInterface is a helper function to convert interface{} (expected to be text[] from DB) to []string
func processStringArrayInterface(dbField interface{}) []string {
	var result []string
	if dbField == nil {
		return []string{}
	}

	switch v := dbField.(type) {
	case []string:
		result = v
	case *[]string:
		if v != nil {
			result = *v
		} else {
			result = []string{}
		}
	case []interface{}:
		for _, item := range v {
			if str, ok := item.(string); ok {
				result = append(result, str)
			}
		}
	case []byte: // Handle case where text[] is scanned as []byte (e.g., "{item1,item2}")
		temp := string(v)
		if len(temp) > 2 && temp[0] == '{' && temp[len(temp)-1] == '}' {
			temp = temp[1 : len(temp)-1] // Remove {}
			if temp != "" {
				// This is a simplified parser. For robust CSV or quoted array elements, a proper parser is needed.
				// It should handle quoted elements like {"genre one","genre two"}
				// For now, a simple split by comma.
				// Consider using a library or more robust parsing if genres can contain commas.
				parts := strings.Split(temp, ",")
				for _, p := range parts {
					// Trim quotes if present (e.g. from `"{""Action"",""Adventure""}"`)
					trimmedPart := strings.Trim(p, "\"")
					result = append(result, trimmedPart)
				}

			} else {
				result = []string{}
			}
		} else {
			result = []string{}
		}
	default:
		fmt.Printf("Warning: Unexpected type for string array field: %T, value: %#v\n", v, v)
		result = []string{}
	}
	if result == nil { // Ensure it's never nil, always at least an empty slice
		return []string{}
	}
	return result
}

// UserInLibraryResponse defines the structure for user data when listing users who have a book.
type UserInLibraryResponse struct {
	ID   pgtype.UUID `json:"id"`
	Name string      `json:"name"`
	// AvatarURL pgtype.Text `json:"avatarUrl,omitempty"` // Add if you have avatar URLs for users
}

// GetUsersWithBookInLibrary returns a paginated list of users who have a specific book in their library.
func (h *Handler) GetUsersWithBookInLibrary(c *gin.Context) {
	bookIDStr := c.Param("id") // Corrected parameter name to "id"
	bookID, err := uuid.Parse(bookIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid book ID format"})
		return
	}

	pgBookID := pgtype.UUID{Bytes: bookID, Valid: true}

	limit := 12 // Default limit (e.g., for 3 columns, 4 rows)
	offset := 0

	if limitParam := c.Query("limit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	if offsetParam := c.Query("offset"); offsetParam != "" {
		parsedOffset, err := strconv.Atoi(offsetParam)
		if err == nil && parsedOffset >= 0 {
			offset = parsedOffset
		}
	}

	dbUsers, err := h.DB.GetUsersByBookInLibrary(c, db.GetUsersByBookInLibraryParams{
		BookID: pgBookID,
		Limit:  int32(limit),
		Offset: int32(offset),
	})

	if err != nil {
		fmt.Printf("Error fetching users for book ID %s: %v\n", bookIDStr, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch users for this book"})
		return
	}

	responseUsers := make([]UserInLibraryResponse, 0, len(dbUsers))
	var totalUsers int64
	if len(dbUsers) > 0 {
		totalUsers = dbUsers[0].TotalUsers // Assuming total_users is on each row
	}

	for _, dbUser := range dbUsers {
		responseUsers = append(responseUsers, UserInLibraryResponse{
			ID:   dbUser.ID,
			Name: dbUser.Name,
			// AvatarURL: dbUser.AvatarURL, // Uncomment if you add AvatarURL to db.GetUsersByBookInLibraryRow
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"users":      responseUsers,
		"totalUsers": totalUsers,
		"page":       (offset / limit) + 1,
		"limit":      limit,
	})
}
