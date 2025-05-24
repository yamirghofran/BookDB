package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// Match the Pydantic model in FastAPI
type SentimentRequest struct {
	Text string `json:"text"`
}

// Match the Pydantic response model in FastAPI
type SentimentResponse struct {
	Text           string             `json:"text"`
	PredictedLabel string             `json:"predicted_label"`
	Probabilities  map[string]float64 `json:"probabilities"` // Maps label name to probability
}

func main() {
	// URL of the running FastAPI service
	url := "http://localhost:8000/predict/" // Make sure the port matches

	// Input text examples
	textsToAnalyze := []string{
		"This movie was absolutely fantastic, I loved every minute!",
		"What a complete waste of time, the plot was nonsensical.",
		"It was an okay experience, neither good nor bad.",
	}

	// Create an HTTP client with a timeout
	client := &http.Client{
		Timeout: 15 * time.Second, // Add a reasonable timeout
	}

	for _, text := range textsToAnalyze {
		fmt.Printf("\n--- Analyzing text: \"%s\" ---\n", text)

		// Prepare the request data
		requestData := SentimentRequest{Text: text}
		jsonData, err := json.Marshal(requestData)
		if err != nil {
			log.Printf("Error marshaling JSON for '%s': %v", text, err)
			continue // Skip to next text
		}

		// Create the POST request
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
		if err != nil {
			log.Printf("Error creating request for '%s': %v", text, err)
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json") // Inform server we accept JSON

		// Send the request
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("Error sending request for '%s': %v", text, err)
			continue
		}
		defer resp.Body.Close() // Ensure body is closed

		// Read the response body
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Printf("Error reading response body for '%s': %v", text, err)
			continue
		}

		// Check the HTTP status code
		if resp.StatusCode != http.StatusOK {
			log.Printf("Error response from server for '%s' (Status: %s): %s", text, resp.Status, string(body))
			continue
		}

		// Unmarshal the JSON response
		var responseData SentimentResponse
		err = json.Unmarshal(body, &responseData)
		if err != nil {
			log.Printf("Error unmarshaling JSON response for '%s': %v", text, err)
			log.Printf("Raw response body: %s", string(body)) // Log raw body on error
			continue
		}

		// Print the results
		fmt.Printf("Input Text: %s\n", responseData.Text)
		fmt.Printf("Predicted Sentiment: %s\n", responseData.PredictedLabel)
		fmt.Println("Probabilities:")
		for label, prob := range responseData.Probabilities {
			fmt.Printf("  %s: %.4f\n", label, prob)
		}
	}
}
