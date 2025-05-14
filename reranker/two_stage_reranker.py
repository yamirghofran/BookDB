import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn

class TwoStageReranker:
    def __init__(
        self,
        user_embeddings_path: str,
        book_embeddings_path: str,
        sbert_embeddings_path: str,
        interactions_path: str,
        books_path: str,
        k_candidates: int = 200,
        k_final: int = 10,
        diversity_weight: float = 0.3
    ):
        """
        Initialize the two-stage reranker.
        
        Args:
            user_embeddings_path: Path to GMF user embeddings
            book_embeddings_path: Path to GMF book embeddings
            sbert_embeddings_path: Path to SBERT book embeddings
            interactions_path: Path to user-book interactions
            k_candidates: Number of candidates to generate (default: 200)
            k_final: Number of final recommendations (default: 10)
            diversity_weight: Weight for diversity in reranking (default: 0.3)
        """
        print("Loading embeddings and data...")
        # Load embeddings
        self.user_embeddings_df = pd.read_parquet(user_embeddings_path)
        self.book_embeddings_df = pd.read_parquet(book_embeddings_path)
        self.sbert_embeddings_df = pd.read_parquet(sbert_embeddings_path)
        self.interactions_df = pd.read_parquet(interactions_path)
        self.books_df = pd.read_parquet(books_path)
        # Convert IDs to proper types
        self.user_embeddings_df['user_id'] = self.user_embeddings_df['user_id'].astype(int)
        self.book_embeddings_df['item_id'] = self.book_embeddings_df['item_id'].astype(int)
        
        # Store embedding column names
        self.gmf_embedding_cols = [str(i) for i in range(32)]
        
        # Initialize parameters
        self.k_candidates = k_candidates
        self.k_final = k_final
        self.diversity_weight = diversity_weight
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        
        # Initialize LightGBM model
        self.model = None
        
        print("Initialization complete!")

    def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get user embedding from dataframe"""
        user_row = self.user_embeddings_df[
            self.user_embeddings_df['user_id'] == user_id
        ]
        if len(user_row) == 0:
            return None
        return user_row[self.gmf_embedding_cols].iloc[0].values

    def _get_book_embeddings(self, book_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get book embeddings from dataframe"""
        book_ids_int = [int(id) for id in book_ids]
        book_rows = self.book_embeddings_df[
            self.book_embeddings_df['item_id'].isin(book_ids_int)
        ]
        
        return {
            str(row['item_id']): row[self.gmf_embedding_cols].values
            for _, row in book_rows.iterrows()
        }

    def _get_sbert_embeddings(self, book_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get SBERT embeddings for books"""
        book_rows = self.sbert_embeddings_df[
            self.sbert_embeddings_df['book_id'].isin(book_ids)
        ]
        
        return {
            str(row['book_id']): np.array(eval(row['embedding']))
            for _, row in book_rows.iterrows()
        }

    def _compute_gmf_scores(
        self,
        user_emb: np.ndarray,
        book_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute GMF scores using dot product"""
        scores = {}
        for book_id, book_emb in book_embeddings.items():
            score = np.dot(user_emb, book_emb)
            if not (np.isnan(score) or np.isinf(score)):
                scores[str(book_id)] = float(score)
        return scores

    def _compute_sbert_scores(
        self,
        user_history: List[str],
        candidate_books: List[str]
    ) -> Dict[str, float]:
        """Compute SBERT similarity scores"""
        # Get user's book embeddings
        user_book_embeddings = self._get_sbert_embeddings(user_history)
        if not user_book_embeddings:
            return {book_id: 0.0 for book_id in candidate_books}
        
        # Get candidate book embeddings
        candidate_embeddings = self._get_sbert_embeddings(candidate_books)
        
        # Compute average similarity to user's books
        scores = {}
        for book_id, book_emb in candidate_embeddings.items():
            similarities = []
            for user_book_emb in user_book_embeddings.values():
                similarity = 1 - cosine(book_emb, user_book_emb)
                similarities.append(similarity)
            scores[book_id] = np.mean(similarities) if similarities else 0.0
        
        return scores

    def _get_book_features(self, book_id: str) -> Dict[str, float]:
        """Extract book features"""
        
        features = {
            'total_ratings': self.books_df['ratings_count'],
            'avg_rating': self.books_df['average_rating'],
        }
        
        return features

    def _generate_candidates(
        self,
        user_id: int,
        k: int = 200
    ) -> List[str]:
        """Generate candidate books using both GMF and SBERT"""
        # Get user embedding
        user_emb = self._get_user_embedding(user_id)
        if user_emb is None:
            return []
        
        # Get user's history
        user_history = self.interactions_df[
            self.interactions_df['user_id'] == user_id
        ]['book_id'].astype(str).tolist()
        
        # Get all book embeddings
        all_book_ids = [str(id) for id in self.book_embeddings_df['item_id'].tolist()]
        book_embeddings = self._get_book_embeddings(all_book_ids)
        
        # Compute GMF scores
        gmf_scores = self._compute_gmf_scores(user_emb, book_embeddings)
        
        # Compute SBERT scores
        sbert_scores = self._compute_sbert_scores(user_history, all_book_ids)
        
        # Combine scores
        combined_scores = []
        for book_id in all_book_ids:
            if book_id in gmf_scores and book_id in sbert_scores:
                combined_score = (
                    0.7 * gmf_scores[book_id] +  # GMF weight
                    0.3 * sbert_scores[book_id]  # SBERT weight
                )
                combined_scores.append((book_id, combined_score))
        
        # Sort by combined score and return top k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in combined_scores[:k]]

    def _prepare_features(
        self,
        user_id: int,
        candidate_books: List[str]
    ) -> pd.DataFrame:
        """Prepare features for reranking"""
        features = []
        
        # Get user embedding
        user_emb = self._get_user_embedding(user_id)
        
        # Get book embeddings
        book_embeddings = self._get_book_embeddings(candidate_books)
        sbert_embeddings = self._get_sbert_embeddings(candidate_books)
        
        for book_id in candidate_books:
            # Get book features
            book_features = self._get_book_features(book_id)
            
            # Compute GMF score
            gmf_score = np.dot(user_emb, book_embeddings[book_id])
            
            # Compute SBERT similarity
            sbert_sim = sbert_embeddings.get(book_id, np.zeros(384))
            sbert_score = np.mean(sbert_sim) if sbert_sim is not None else 0
            
            # Combine all features
            feature_dict = {
                'book_id': book_id,
                'gmf_score': gmf_score,
                'sbert_score': sbert_score,
                **book_features
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)

    def train(
        self,
        train_interactions: pd.DataFrame,
        val_interactions: Optional[pd.DataFrame] = None
    ):
        """Train the reranker model"""
        print("Preparing training data...")
        
        # Prepare training features
        train_features = []
        
        for user_id in train_interactions['user_id'].unique():
            # Get user's interaction history
            user_history = train_interactions[
                train_interactions['user_id'] == user_id
            ]['book_id'].astype(str).tolist()
            
            if not user_history:
                continue
            
            # Generate candidates
            candidates = self._generate_candidates(user_id, self.k_candidates)
            
            # Get user embedding
            user_emb = self._get_user_embedding(user_id)
            if user_emb is None:
                continue
                
            # Get book embeddings for candidates
            book_embeddings = self._get_book_embeddings(candidates)
            
            # Compute GMF scores for classification
            gmf_scores = self._compute_gmf_scores(user_emb, book_embeddings)
            
            # Use books with high GMF scores as positive examples
            # Take top 20% of candidates by GMF score
            sorted_candidates = sorted(
                gmf_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            num_positive = max(1, int(len(candidates) * 0.2))
            positive_books = [book_id for book_id, _ in sorted_candidates[:num_positive]]
            
            # Prepare features
            features_df = self._prepare_features(user_id, candidates)
            
            # Add labels (1 for positive books, 0 for others)
            features_df['label'] = features_df['book_id'].isin(positive_books).astype(int)
            
            train_features.append(features_df)
        
        # Combine all features
        train_data = pd.concat(train_features, ignore_index=True)
        
        # Prepare validation data if provided
        val_data = None
        if val_interactions is not None:
            val_features = []
            for user_id in val_interactions['user_id'].unique():
                user_history = val_interactions[
                    val_interactions['user_id'] == user_id
                ]['book_id'].astype(str).tolist()
                
                if not user_history:
                    continue
                
                candidates = self._generate_candidates(user_id, self.k_candidates)
                
                # Get user embedding
                user_emb = self._get_user_embedding(user_id)
                if user_emb is None:
                    continue
                    
                # Get book embeddings for candidates
                book_embeddings = self._get_book_embeddings(candidates)
                
                # Compute GMF scores for classification
                gmf_scores = self._compute_gmf_scores(user_emb, book_embeddings)
                
                # Use books with high GMF scores as positive examples
                sorted_candidates = sorted(
                    gmf_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                num_positive = max(1, int(len(candidates) * 0.2))
                positive_books = [book_id for book_id, _ in sorted_candidates[:num_positive]]
                
                features_df = self._prepare_features(user_id, candidates)
                features_df['label'] = features_df['book_id'].isin(positive_books).astype(int)
                val_features.append(features_df)
            
            val_data = pd.concat(val_features, ignore_index=True)
        
        # Prepare features for training
        feature_cols = [col for col in train_data.columns if col not in ['book_id', 'label']]
        X_train = train_data[feature_cols]
        y_train = train_data['label']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation data if available
        if val_data is not None:
            X_val = val_data[feature_cols]
            y_val = val_data['label']
            X_val_scaled = self.scaler.transform(X_val)
            valid_data = lgb.Dataset(X_val_scaled, label=y_val)
        else:
            valid_data = None
        
        # Create training dataset
        train_dataset = lgb.Dataset(X_train_scaled, label=y_train)
        
        # Set up LightGBM parameters
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'min_sum_hessian_in_leaf': 1e-3,
            'random_state': 42
        }
        
        print("Training model...")
        # Train the model
        self.model = lgb.train(
            params,
            train_dataset,
            num_boost_round=100,
            valid_sets=[valid_data] if valid_data else None,
            early_stopping_rounds=10 if valid_data else None
        )
        
        print("Training complete!")

    def get_recommendations(
        self,
        user_id: int,
        k: Optional[int] = None
    ) -> List[str]:
        """Get reranked book recommendations for a user"""
        if k is None:
            k = self.k_final
        
        # Generate candidates
        candidates = self._generate_candidates(user_id, self.k_candidates)
        
        if not candidates:
            return []
        
        # Prepare features
        features_df = self._prepare_features(user_id, candidates)
        
        # Scale features
        feature_cols = [col for col in features_df.columns if col != 'book_id']
        X = self.scaler.transform(features_df[feature_cols])
        
        # Get predictions
        scores = self.model.predict(X)
        
        # Combine book IDs with scores
        book_scores = list(zip(features_df['book_id'], scores))
        
        # Sort by score and return top k
        book_scores.sort(key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in book_scores[:k]]

    def evaluate(
        self,
        test_interactions: pd.DataFrame,
        k: int = 10
    ) -> Dict[str, float]:
        """Evaluate the reranker on test data"""
        metrics = {
            'ndcg@5': 0.0,
            'ndcg@10': 0.0,
            'recall@5': 0.0,
            'recall@10': 0.0
        }
        
        total_users = 0
        
        for user_id in test_interactions['user_id'].unique():
            # Get user's positive interactions
            positive_books = set(test_interactions[
                (test_interactions['user_id'] == user_id) &
                (test_interactions['rating'] >= 4)
            ]['book_id'].astype(str))
            
            if not positive_books:
                continue
            
            # Get recommendations
            recommendations = self.get_recommendations(user_id, k=k)
            
            # Compute metrics
            hits_5 = len(set(recommendations[:5]) & positive_books)
            hits_10 = len(set(recommendations[:10]) & positive_books)
            
            # NDCG@5
            dcg_5 = sum(1 / np.log2(i + 2) for i, book in enumerate(recommendations[:5])
                       if book in positive_books)
            idcg_5 = sum(1 / np.log2(i + 2) for i in range(min(5, len(positive_books))))
            metrics['ndcg@5'] += dcg_5 / idcg_5 if idcg_5 > 0 else 0
            
            # NDCG@10
            dcg_10 = sum(1 / np.log2(i + 2) for i, book in enumerate(recommendations[:10])
                        if book in positive_books)
            idcg_10 = sum(1 / np.log2(i + 2) for i in range(min(10, len(positive_books))))
            metrics['ndcg@10'] += dcg_10 / idcg_10 if idcg_10 > 0 else 0
            
            # Recall@5
            metrics['recall@5'] += hits_5 / len(positive_books)
            
            # Recall@10
            metrics['recall@10'] += hits_10 / len(positive_books)
            
            total_users += 1
        
        # Average metrics
        if total_users > 0:
            for metric in metrics:
                metrics[metric] /= total_users
        
        return metrics 