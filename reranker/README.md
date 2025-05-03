# BookDB Reranker

A recommendation reranking module for the BookDB project that combines multiple sources of book recommendations, filters out books that users already own, and provides a unified recommendation list with additional book details.

## Overview

The BookDB Reranker takes recommendations from multiple sources (e.g., machine learning models and collaborative filtering) and:

1. Filters out books that are already in the user's library using the UserLibrary table
2. Significantly boosts scores for books that appear in both recommendation lists
3. Considers user interaction data, giving higher scores to books with more interactions
4. Normalizes scores from different recommendation systems
5. Combines scores using configurable weights
6. Reranks recommendations based on combined scores
7. Enriches the final recommendations with book details from the database

## Process Flow

The reranking process follows these steps:

1. Query the UserLibrary table to identify books the user already owns
2. Take two input lists of recommendations (assumed to be in descending order of similarity)
3. Remove any books that appear in the user's library
4. Identify books that appear in both recommendation lists and boost their scores
5. Retrieve interaction counts for each book and apply an additional boost based on interaction frequency
6. Combine all scores to create a single unified ranking
7. Return the top N recommended books

## Inputs

The reranker accepts the following inputs:

- **User ID**: String identifier (UUID) for the user receiving recommendations
- **Model Recommendations**: List of dictionaries containing ML model-based recommendations with:
  - `book_id`: String identifier for the book
  - `score`: Float confidence score (0.0-1.0) from the model
- **Collaborative Filtering (CF) Recommendations**: List of dictionaries containing CF-based recommendations with:
  - `book_id`: String identifier for the book
  - `score`: Float confidence score (0.0-1.0) from the CF system
- **Model Weight**: Float value (0.0-1.0) determining how much weight to give ML model scores vs CF scores
- **Top N**: Integer specifying how many final recommendations to return
- **Boost Factor**: Optional parameter determining how much to boost books appearing in both recommendation sources

## Database Schema

The reranker relies on the following database tables:

- **UserLibrary**: Stores user-owned books
  ```sql
  CREATE TABLE UserLibrary (
      user_id UUID NOT NULL REFERENCES Users(id) ON DELETE CASCADE,
      book_id VARCHAR NOT NULL,
      PRIMARY KEY (user_id, book_id)
  )
  ```
- **user_book_interactions**: Tracks user interactions with books to influence recommendations

## Outputs

The reranker produces a list of dictionaries containing:

- `book_id`: String identifier for the recommended book
- `score`: Float combined confidence score (0.0-1.0)
- `model_score`: Normalized score from the ML model
- `cf_score`: Normalized score from the collaborative filtering system
- `sources`: List of strings indicating which systems recommended this book (`"model"`, `"cf"`, or both)
- `interaction_count`: Number of interactions the user has had with similar books
- `from_both_sources`: Boolean indicating whether the book was recommended by both systems
- `details`: Dictionary containing book metadata (when enriched):
  - `title`: Book title
  - `author`: Book author
  - `year_published`: Publication year
  - `genre`: Book genre

## Usage

The reranker can be used in two ways:

### 1. Command Line Interface

```bash
python -m reranker.main --user_id=user123 --model_weight=0.6 --top_n=10 --enrich < recommendations.json
```

Where `recommendations.json` contains:
```json
{
  "model_recommendations": [
    {"book_id": "book1", "score": 0.95},
    {"book_id": "book2", "score": 0.87}
  ],
  "cf_recommendations": [
    {"book_id": "book3", "score": 0.92},
    {"book_id": "book6", "score": 0.85}
  ]
}
```

### 2. Integration with a Python Pipeline

```python
from reranker import BookReranker

def book_recommendation_pipeline(user_id, config_path="config.yaml"):
    # Initialize components
    reranker = BookReranker(config_path=config_path)
    
    try:
        # Get recommendations from different sources
        # These come from your actual recommendation models
        ml_recommendations = get_ml_model_recommendations(user_id)
        cf_recommendations = get_collaborative_filtering_recommendations(user_id)
        
        # Rerank and filter recommendations
        reranked = reranker.rerank_recommendations(
            user_id=user_id,
            model_recs=ml_recommendations,
            cf_recs=cf_recommendations,
            model_weight=0.5,
            top_n=10
        )
        
        # Optionally enrich with book details
        final_recommendations = reranker.enrich_recommendations(reranked)
        
        return final_recommendations
    
    finally:
        # Ensure resources are properly closed
        reranker.close()
```

## Configuration

The reranker expects a YAML configuration file with the following structure:

```yaml
database:
  host: localhost
  port: 5432
  name: bookdb
  user: postgres
  password: your_password
```

## Dependencies

- Python 3.6+
- PyYAML
- psycopg2
- pandas
- numpy