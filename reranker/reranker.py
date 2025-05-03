import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

class LocalBookReranker:
    def __init__(
        self,
        user_embeddings_path: str = "../embeddings/gmf_user_embeddings.parquet",
        book_embeddings_path: str = "../embeddings/gmf_book_embeddings.parquet",
        sbert_embeddings_path: str = "../embeddings/SBERT_embeddings.parquet",
        diversity_weight: float = 0.3,
        gmf_weight: float = 0.7,
        sbert_weight: float = 0.3,
        k: int = 10
    ):
        print("Loading embeddings...")
        # Load with proper data types
        self.user_embeddings_df = pd.read_parquet(user_embeddings_path)
        self.book_embeddings_df = pd.read_parquet(book_embeddings_path)
        self.sbert_embeddings_df = pd.read_parquet(sbert_embeddings_path)
        
        # Convert IDs to proper types and ensure they're integers
        self.user_embeddings_df['user_id'] = self.user_embeddings_df['user_id'].astype(int)
        self.book_embeddings_df['item_id'] = self.book_embeddings_df['item_id'].astype(int)
        
        # Store GMF embedding column names
        self.gmf_embedding_cols = [str(i) for i in range(32)]
        
        self.diversity_weight = diversity_weight
        self.gmf_weight = gmf_weight
        self.sbert_weight = sbert_weight
        self.k = k
        print("Embeddings loaded successfully!")

    def _get_initial_ranking(
        self,
        candidates: List[str],
        gmf_scores: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Create initial ranking based on GMF scores"""
        print(f"Debug - Number of candidates: {len(candidates)}")
        print(f"Debug - Number of GMF scores: {len(gmf_scores)}")
        print(f"Debug - First few candidates: {candidates[:5]}")
        print(f"Debug - First few GMF scores keys: {list(gmf_scores.keys())[:5]}")
        print(f"Debug - Sample GMF scores values: {list(gmf_scores.values())[:5]}")
        
        # Convert float book IDs to integer strings
        gmf_scores_fixed = {
            str(int(float(k))): v 
            for k, v in gmf_scores.items()
        }
        
        print(f"Debug - First few fixed GMF scores keys: {list(gmf_scores_fixed.keys())[:5]}")
        
        # Ensure all IDs are strings and filter out any missing scores
        scored_candidates = []
        for book_id in candidates:
            str_id = str(book_id)
            if str_id in gmf_scores_fixed:
                score = gmf_scores_fixed[str_id]
                if not (np.isnan(score) or np.isinf(score)):
                    scored_candidates.append((str_id, score))
        
        print(f"Debug - Number of scored candidates: {len(scored_candidates)}")
        if scored_candidates:
            print(f"Debug - First few scored candidates: {scored_candidates[:5]}")
        
        # Sort by score in descending order
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    def _compute_gmf_scores(
        self,
        user_emb: List[float],
        book_embeddings: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Compute GMF scores using dot product"""
        scores = {}
        for book_id, book_emb in book_embeddings.items():
            score = np.dot(user_emb, book_emb)
            if not (np.isnan(score) or np.isinf(score)):  # Filter out invalid scores
                scores[str(book_id)] = float(score)  # Ensure score is a regular float
        return scores

    def get_recommendations(
        self,
        user_id: str,
        query: str = None,
        candidate_books: List[str] = None
    ) -> List[str]:
        """Get reranked book recommendations for a user"""
        print(f"\nGetting recommendations for user {user_id}")
        
        # Convert user_id to int for comparison
        user_id_int = int(user_id)
        
        # Get user embedding
        user_emb = self._get_user_embedding(user_id_int)
        if user_emb is None:
            raise ValueError(f"No embedding found for user {user_id}")
        
        # Get candidate books if not provided
        if candidate_books is None:
            candidate_books = [str(id) for id in self.book_embeddings_df['item_id'].tolist()]
        
        print(f"Number of candidate books: {len(candidate_books)}")
        
        # Get book embeddings and compute GMF scores
        book_embeddings = self._get_book_embeddings(candidate_books)
        print(f"Number of book embeddings retrieved: {len(book_embeddings)}")
        
        if not book_embeddings:
            print("Warning: No book embeddings retrieved!")
            return []
        
        gmf_scores = self._compute_gmf_scores(user_emb, book_embeddings)
        print(f"Number of GMF scores computed: {len(gmf_scores)}")
        
        # Get initial ranking
        initial_ranking = self._get_initial_ranking(candidate_books, gmf_scores)
        print(f"Length of initial ranking: {len(initial_ranking)}")
        
        if len(initial_ranking) == 0:
            print("Warning: Initial ranking is empty!")
            return []
        
        return [book_id for book_id, _ in initial_ranking[:self.k]]

    def _get_user_embedding(self, user_id: int) -> List[float]:
        """Get user embedding from dataframe"""
        user_row = self.user_embeddings_df[
            self.user_embeddings_df['user_id'] == user_id
        ]
        if len(user_row) == 0:
            return None
        return user_row[self.gmf_embedding_cols].iloc[0].values.tolist()

    def _get_book_embeddings(self, book_ids: List[str]) -> Dict[str, List[float]]:
        """Get book embeddings from dataframe"""
        # Convert book_ids to integers for comparison
        book_ids_int = [int(id) for id in book_ids]
        book_rows = self.book_embeddings_df[
            self.book_embeddings_df['item_id'].isin(book_ids_int)
        ]
        
        # Create dictionary with string keys for consistency
        return {
            str(row['item_id']): row[self.gmf_embedding_cols].values.tolist()
            for _, row in book_rows.iterrows()
        } 