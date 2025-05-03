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

    def __init__(self, config_path: str = "../config.yaml"):
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
    """Run a test of the reranker with sample data."""
    try:
        # Initialize reranker
        reranker = BookReranker()

        # Sample user and recommendations (for testing)
        sample_user_id = "test_user_123"

        # Sample ML model recommendations
        model_recs = [
            {"book_id": "book1", "score": 0.95},
            {"book_id": "book2", "score": 0.87},
            {"book_id": "book3", "score": 0.82},
            {"book_id": "book4", "score": 0.76},
        ]

        # Sample CF recommendations
        cf_recs = [
            {"book_id": "book3", "score": 0.92},
            {"book_id": "book5", "score": 0.88},
            {"book_id": "book6", "score": 0.85},
            {"book_id": "book2", "score": 0.79},
        ]

        # Get reranked recommendations
        reranked = reranker.rerank_recommendations(
            user_id=sample_user_id,
            model_recs=model_recs,
            cf_recs=cf_recs,
            model_weight=0.6,
            top_n=5,
        )

        # Enrich with book details
        enriched_recs = reranker.enrich_recommendations(reranked)

        # Print results
        print("\nReranked Book Recommendations:")
        for i, rec in enumerate(enriched_recs, 1):
            print(f"{i}. Book ID: {rec['book_id']}, Score: {rec['score']:.4f}")
            print(f"   Sources: {', '.join(rec['sources'])}")
            print(
                f"   Model Score: {rec['model_score']:.4f}, CF Score: {rec['cf_score']:.4f}"
            )
            if "details" in rec:
                details = rec["details"]
                title = details.get("title", "Unknown")
                author = details.get("author", "Unknown")
                print(f"   Title: {title}, Author: {author}")
            print()

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
