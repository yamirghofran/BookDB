package api

import (
	"net/http"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/qdrant/go-client/qdrant"
	"github.com/yamirghofran/BookDB/internal/api/handlers"
	"github.com/yamirghofran/BookDB/internal/db"
)

// Server holds all dependencies needed for the API
type Server struct {
	DB           *db.Queries
	QdrantClient *qdrant.Client
	Router       *gin.Engine
	Handlers     *handlers.Handler
}

// NewServer creates a new Server instance with all dependencies
func NewServer(db *db.Queries, qdrantClient *qdrant.Client) *Server {
	server := &Server{
		DB:           db,
		QdrantClient: qdrantClient,
		Handlers:     handlers.NewHandler(db, qdrantClient),
	}

	server.setupRouter()
	return server
}

// setupRouter configures the Gin router with all routes and middleware
func (s *Server) setupRouter() {
	router := gin.Default()

	// Setup CORS middleware
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000"}, // Update with your frontend URL
		AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Serve static files (frontend)
	router.StaticFS("/static", http.Dir("./static"))

	// API routes
	api := router.Group("/api")
	{
		// Health check endpoint
		api.GET("/health", s.healthCheck)

		// Books endpoints
		books := api.Group("/books")
		{
			books.GET("", s.Handlers.ListBooks)
			books.GET("/:id", s.Handlers.GetBook)
			books.POST("", s.Handlers.CreateBook)
			books.PUT("/:id", s.Handlers.UpdateBook)
			books.DELETE("/:id", s.Handlers.DeleteBook)
			books.GET("/search", s.Handlers.SearchBooksByContent)
			books.GET("/:id/library-users", s.Handlers.GetUsersWithBookInLibrary) // Changed route
		}

		// People endpoints
		people := api.Group("/people")
		{
			people.GET("", s.Handlers.ListPeople)
			people.GET("/:id", s.Handlers.GetPerson)
			people.POST("", s.Handlers.CreatePerson)
			people.GET("/:id/details", s.Handlers.GetPersonDetails) // New route for person details
		}

		// Library endpoints
		libraries := api.Group("/libraries")
		{
			libraries.GET("", s.Handlers.ListLibraries)
			libraries.GET("/:id", s.Handlers.GetLibrary)
			libraries.GET("/:id/books", s.Handlers.GetBooksInLibrary)
			libraries.POST("/:id/books", s.Handlers.AddBookToLibrary)
		}

		// Recommendations endpoints
		recommendations := api.Group("/recommendations")
		{
			recommendations.GET("/books/:id/similar", s.Handlers.GetSimilarBooks)
			// recommendations.GET("/users/:id", s.Handlers.GetRecommendationsForUser) // This was for collaborative filtering by user for books
			recommendations.POST("/anonymous", s.Handlers.GetAnonymousRecommendations)
			recommendations.GET("/users/:id/similar", s.Handlers.GetSimilarUsers)             // New route for similar users
			recommendations.GET("/users/:id/books", s.Handlers.GetBookRecommendationsForUser) // New route for book recommendations for a user
		}

		// Auth endpoints
		auth := api.Group("/auth")
		{
			auth.POST("/login", s.Handlers.Login)
			auth.POST("/register", s.Handlers.Register)
		}

		// Protected routes (require authentication)
		protected := api.Group("/user")
		protected.Use(s.Handlers.AuthMiddleware())
		{
			protected.GET("/me", s.Handlers.GetMe)
		}

		// Authors endpoints
		authors := api.Group("/authors")
		{
			authors.GET("", s.Handlers.ListAuthors)
			authors.GET("/:id", s.Handlers.GetAuthor)
			authors.GET("/:id/books", s.Handlers.GetBooksByAuthor)
			authors.GET("/search", s.Handlers.SearchAuthors)
		}
	}

	// Catch-all route to serve the frontend (for SPA)
	router.NoRoute(func(c *gin.Context) {
		c.File("./static/index.html")
	})

	s.Router = router
}

// Run starts the HTTP server
func (s *Server) Run(addr string) error {
	return s.Router.Run(addr)
}

// healthCheck is a simple health check endpoint
func (s *Server) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"time":   time.Now(),
	})
}
