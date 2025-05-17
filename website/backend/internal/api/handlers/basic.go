package handlers

import (
	"net/http"
	"sort"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/yamirghofran/BookDB/internal/db"
)

// Handler holds dependencies for handlers
type Handler struct {
	DB *db.Queries
}

// NewHandler creates a new handler instance
func NewHandler(db *db.Queries) *Handler {
	return &Handler{
		DB: db,
	}
}

// Full Text Search
// SearchBooksByContent searches for books by content similarity
func (h *Handler) SearchBooksByContent(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Search query is required"})
		return
	}

	limit := 10
	offset := 0

	if limitParam := c.Query("limit"); limitParam != "" {
		parsedLimit, err := strconv.Atoi(limitParam)
		if err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	if offsetParam := c.Query("offset"); offsetParam != "" {
		parsedOffset, err := strconv.Atoi(offsetParam)
		if err == nil && parsedOffset > 0 {
			offset = parsedOffset
		}
	}

	// Use the PostgreSQL full-text search
	books, err := h.DB.SearchBooks(c, db.SearchBooksParams{
		WebsearchToTsquery: query,
		Limit:              int32(limit),
		Offset:             int32(offset),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"query":   query,
		"results": books,
	})
}

// Book handlers

// ListBooks returns a list of books
func (h *Handler) ListBooks(c *gin.Context) {
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

	books, err := h.DB.ListBooks(c, db.ListBooksParams{
		Limit:  int32(limit),
		Offset: int32(offset),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, books)
}

// GetBook returns a specific book by ID
func (h *Handler) GetBook(c *gin.Context) {
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

	book, err := h.DB.GetBookByID(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Book not found"})
		return
	}

	c.JSON(http.StatusOK, book)
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
		AverageRating: pgtype.Numeric{
			Valid: req.AverageRating > 0,
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
		params.AverageRating.Valid = true
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
		AverageRating: pgtype.Numeric{
			Valid: req.AverageRating > 0,
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
		params.AverageRating.Valid = true
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
	if err == nil {
		c.JSON(http.StatusConflict, gin.H{"error": "Email already in use"})
		return
	}

	user, err := h.DB.CreateUser(c, db.CreateUserParams{
		Name:  req.Name,
		Email: req.Email,
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
	books, err := h.DB.GetUserLibrary(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, books)
}

// AddBookToLibrary adds a book to a library
func (h *Handler) AddBookToLibrary(c *gin.Context) {
	idStr := c.Param("id")

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
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
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
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
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
	authors, err := h.DB.GetAuthorByName(c, pgtype.Text{
		String: query,
		Valid:  true,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Sort authors by ratings count in descending order
	sort.Slice(authors, func(i, j int) bool {
		return authors[i].RatingsCount.Int32 > authors[j].RatingsCount.Int32
	})

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

	books, err := h.DB.GetBooksByAuthor(c, pgUUID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, books)
}
