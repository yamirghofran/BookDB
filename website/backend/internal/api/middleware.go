package api

import (
	// Added for logging
	// Added for http status codes
	"os"      // Import os package to read environment variables
	"strings" // Import strings package for TrimSuffix

	"github.com/gin-gonic/gin"
)

// CORSMiddleware adds CORS headers to allow cross-origin requests
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Read the allowed origin from environment variable
		frontendURL := os.Getenv("FRONTEND_URL")
		if frontendURL == "" {
			// Fallback or log an error if not set, depending on requirements
			// For development, you might fallback to a common dev URL,
			// but in production, it should likely be a fatal error if not set.
			// Using "*" here would defeat the purpose of the fix.
			// Let's fallback to the typical Vite dev server URL for now.
			frontendURL = "http://localhost:3000" // Or log fatal
		}
		// Trim trailing slash if present before setting the header
		c.Writer.Header().Set("Access-Control-Allow-Origin", strings.TrimSuffix(frontendURL, "/"))
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}
