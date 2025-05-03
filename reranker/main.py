import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BookReranker")


class BookReranker:
    """
    Reranker for book recommendations that:
    1. Filters out books already in the user's library
    2. Reranks recommendations based on combined model confidence scores
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the reranker with configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.db_conn = None

        # Connect to database
        try:
            self.db_conn = self._connect_to_db()
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def _connect_to_db(self) -> psycopg2.extensions.connection:
        """
        Connect to PostgreSQL database using configuration settings.

        Returns:
            Database connection object
        """
        db_config = self.config.get("database", {})
        return psycopg2.connect(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
            database=db_config.get("name", "bookdb"),
            user=db_config.get("user", "postgres"),
            password=db_config.get("password", ""),
        )

    def get_user_library(self, user_id: str) -> List[str]:
        """
        Get the list of book IDs that the user already has in their library.

        Args:
            user_id: ID of the user

        Returns:
            List of book IDs in the user's library
        """
        try:
            cursor = self.db_conn.cursor()
            # Adjust table and column names according to your actual database schema
            cursor.execute(
                "SELECT book_id FROM user_books WHERE user_id = %s", (user_id,)
            )
            user_books = [str(row[0]) for row in cursor.fetchall()]
            cursor.close()
            logger.info(f"Found {len(user_books)} books in user {user_id}'s library")
            return user_books
        except Exception as e:
            logger.error(f"Error fetching user library: {e}")
            return []

    def rerank_recommendations(
        self,
        user_id: str,
        model_recs: List[Dict[str, Any]],
        cf_recs: List[Dict[str, Any]],
        model_weight: float = 0.5,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rerank book recommendations by:
        1. Removing books already in user's library
        2. Combining and reranking based on confidence scores

        Args:
            user_id: ID of the user
            model_recs: Recommendations from ML model (should contain 'book_id' and 'score' keys)
            cf_recs: Recommendations from collaborative filtering (should contain 'book_id' and 'score' keys)
            model_weight: Weight to assign to ML model scores (CF weight = 1 - model_weight)
            top_n: Number of recommendations to return

        Returns:
            List of reranked book recommendations
        """
        # Get user library to filter out books they already own
        user_library = set(self.get_user_library(user_id))
        logger.info(f"Filtering out {len(user_library)} books from user's library")

        # Check if we have any recommendations to process
        if not model_recs and not cf_recs:
            logger.warning("No recommendations provided from either source")
            return []

        # Normalize model recommendations scores
        if model_recs:
            model_scores = [rec.get("score", 0) for rec in model_recs]
            if model_scores:
                max_model_score = max(model_scores)
                min_model_score = min(model_scores)
                score_range = max_model_score - min_model_score
                if score_range > 0:
                    for rec in model_recs:
                        rec["normalized_score"] = (
                            rec.get("score", 0) - min_model_score
                        ) / score_range
                else:
                    for rec in model_recs:
                        rec["normalized_score"] = 1.0
            else:
                for rec in model_recs:
                    rec["normalized_score"] = 0.0

        # Normalize CF recommendations scores
        if cf_recs:
            cf_scores = [rec.get("score", 0) for rec in cf_recs]
            if cf_scores:
                max_cf_score = max(cf_scores)
                min_cf_score = min(cf_scores)
                score_range = max_cf_score - min_cf_score
                if score_range > 0:
                    for rec in cf_recs:
                        rec["normalized_score"] = (
                            rec.get("score", 0) - min_cf_score
                        ) / score_range
                else:
                    for rec in cf_recs:
                        rec["normalized_score"] = 1.0
            else:
                for rec in cf_recs:
                    rec["normalized_score"] = 0.0

        # Create a dictionary of all recommendations with their sources
        all_recs = {}

        # Add ML model recommendations
        for rec in model_recs:
            book_id = rec.get("book_id")
            if book_id and book_id not in user_library:
                if book_id not in all_recs:
                    all_recs[book_id] = {
                        "book_id": book_id,
                        "model_score": rec.get("normalized_score", 0),
                        "cf_score": 0,
                        "sources": ["model"],
                    }
                else:
                    all_recs[book_id]["model_score"] = rec.get("normalized_score", 0)
                    if "model" not in all_recs[book_id]["sources"]:
                        all_recs[book_id]["sources"].append("model")

        # Add CF recommendations
        for rec in cf_recs:
            book_id = rec.get("book_id")
            if book_id and book_id not in user_library:
                if book_id not in all_recs:
                    all_recs[book_id] = {
                        "book_id": book_id,
                        "model_score": 0,
                        "cf_score": rec.get("normalized_score", 0),
                        "sources": ["cf"],
                    }
                else:
                    all_recs[book_id]["cf_score"] = rec.get("normalized_score", 0)
                    if "cf" not in all_recs[book_id]["sources"]:
                        all_recs[book_id]["sources"].append("cf")

        # Calculate combined scores
        combined_recs = []
        for book_id, rec_data in all_recs.items():
            combined_score = (
                model_weight * rec_data["model_score"]
                + (1 - model_weight) * rec_data["cf_score"]
            )

            combined_recs.append(
                {
                    "book_id": book_id,
                    "score": combined_score,
                    "model_score": rec_data["model_score"],
                    "cf_score": rec_data["cf_score"],
                    "sources": rec_data["sources"],
                }
            )

        # Sort recommendations by combined score
        reranked_recs = sorted(combined_recs, key=lambda x: x["score"], reverse=True)

        # Limit to top N recommendations
        top_recommendations = reranked_recs[:top_n]

        logger.info(
            f"Generated {len(top_recommendations)} reranked recommendations for user {user_id}"
        )
        return top_recommendations

    def get_book_details(self, book_ids: List[str]) -> Dict[str, Dict]:
        """
        Get additional book details from the database for the recommended books.

        Args:
            book_ids: List of book IDs to fetch details for

        Returns:
            Dictionary mapping book_id to book details
        """
        if not book_ids:
            return {}

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            placeholders = ", ".join(["%s"] * len(book_ids))
            query = f"""
                SELECT * FROM books 
                WHERE book_id IN ({placeholders})
            """
            cursor.execute(query, tuple(book_ids))
            results = cursor.fetchall()
            cursor.close()

            # Convert to dictionary with book_id as key
            book_details = {str(row["book_id"]): dict(row) for row in results}
            return book_details
        except Exception as e:
            logger.error(f"Error fetching book details: {e}")
            return {}

    def enrich_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich recommendation results with book details.

        Args:
            recommendations: List of recommendation dictionaries with book_id

        Returns:
            List of recommendations with added book details
        """
        book_ids = [rec["book_id"] for rec in recommendations]
        book_details = self.get_book_details(book_ids)

        enriched_recs = []
        for rec in recommendations:
            book_id = rec["book_id"]
            if book_id in book_details:
                # Create a new dict with recommendation data and book details
                enriched_rec = {**rec, "details": book_details[book_id]}
                enriched_recs.append(enriched_rec)
            else:
                enriched_recs.append(rec)

        return enriched_recs

    def close(self):
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")


def main():
    """
    Process book recommendations using the reranker.
    
    The reranker:
    1. Takes model and collaborative filtering recommendations as inputs
    2. Removes books already in the user's library
    3. Reranks based on combined confidence scores
    4. Returns the new recommendation ranks
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Rerank book recommendations")
    parser.add_argument("--user_id", required=True, help="ID of the user to get recommendations for")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model_weight", type=float, default=0.5, help="Weight for model scores (0.0-1.0)")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations to return")
    parser.add_argument("--enrich", action="store_true", help="Enrich recommendations with book details")
    args = parser.parse_args()
    
    try:
        # Initialize reranker with provided config
        reranker = BookReranker(config_path=args.config)
        
        # Get recommendations from standard input or API
        # This part would be customized based on how your system passes recommendations
        import json
        import sys
        
        try:
            input_data = json.load(sys.stdin)
            model_recs = input_data.get("model_recommendations", [])
            cf_recs = input_data.get("cf_recommendations", [])
        except json.JSONDecodeError:
            logger.error("Failed to parse input JSON")
            return 1
            
        # Get reranked recommendations
        reranked = reranker.rerank_recommendations(
            user_id=args.user_id,
            model_recs=model_recs,
            cf_recs=cf_recs,
            model_weight=args.model_weight,
            top_n=args.top_n,
        )
        
        # Optionally enrich with book details
        if args.enrich:
            reranked = reranker.enrich_recommendations(reranked)
        
        # Output results as JSON
        print(json.dumps({"recommendations": reranked}))

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return 1
    finally:
        # Ensure resources are properly closed
        if "reranker" in locals():
            reranker.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
