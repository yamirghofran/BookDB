# BookDB Reranker

A recommendation reranking module for the BookDB project that combines multiple sources of book recommendations, filters out books that users already own, and provides a unified recommendation list with additional book details.

## Overview

The BookDB Reranker takes recommendations from multiple sources (e.g., machine learning models and collaborative filtering) and:

1. Filters out books that are already in the user's library
2. Normalizes scores from different recommendation systems
3. Combines scores using configurable weights
4. Reranks recommendations based on combined scores
5. Enriches the final recommendations with book details from the database

## Inputs

The reranker accepts the following inputs:

- **User ID**: String identifier for the user receiving recommendations
- **Model Recommendations**: List of dictionaries containing ML model-based recommendations with:
  - `book_id`: String identifier for the book
  - `score`: Float confidence score (0.0-1.0) from the model
- **Collaborative Filtering (CF) Recommendations**: List of dictionaries containing CF-based recommendations with:
  - `book_id`: String identifier for the book
  - `score`: Float confidence score (0.0-1.0) from the CF system
- **Model Weight**: Float value (0.0-1.0) determining how much weight to give ML model scores vs CF scores
- **Top N**: Integer specifying how many final recommendations to return

## Outputs

The reranker produces a list of dictionaries containing:

- `book_id`: String identifier for the recommended book
- `score`: Float combined confidence score (0.0-1.0)
- `model_score`: Normalized score from the ML model
- `cf_score`: Normalized score from the collaborative filtering system
- `sources`: List of strings indicating which systems recommended this book (`"model"`, `"cf"`, or both)
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