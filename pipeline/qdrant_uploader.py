import os
import pandas as pd
import dask.dataframe as dd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from abc import ABC, abstractmethod
from typing import Dict, Any
from .core import PipelineStep
import logging

class QdrantManager:
    def __init__(self, url="http://localhost:6333"):
        self.client = QdrantClient(url=url)

    def create_collection_if_not_exists(self, collection_name, vector_size, distance_metric):
        try:
            self.client.get_collection(collection_name=collection_name)
            logging.info(f"Collection '{collection_name}' already exists.")
        except Exception:
            logging.info(f"Creating collection '{collection_name}'...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance_metric),
            )
            logging.info(f"Collection '{collection_name}' created.")

    def batch_upload_points(self, collection_name, points, batch_size=500):
        if not points:
            logging.warning(f"No points to upload for collection '{collection_name}'.")
            return
        logging.info(f"Upserting {len(points)} points to '{collection_name}' in batches of {batch_size}...")
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch, wait=True)
        logging.info(f"Uploaded {len(points)} points to '{collection_name}'.")

class BaseEmbeddingProcessor(ABC):
    def __init__(self, embedding_file_path):
        self.embedding_file_path = embedding_file_path
        self.raw_embeddings_ddf = None
        self.processed_embeddings_ddf = None

    def _load_parquet_dask(self):
        logging.info(f"Loading embeddings from {self.embedding_file_path}...")
        self.raw_embeddings_ddf = dd.read_parquet(self.embedding_file_path)
        return self.raw_embeddings_ddf

    def _combine_embedding_columns(self, ddf, num_embedding_cols=32):
        embedding_cols = [str(i) for i in range(num_embedding_cols)]
        def combine_embeddings_func(row):
            return row[embedding_cols].tolist()
        ddf['embedding'] = ddf.apply(
            combine_embeddings_func,
            axis=1,
            meta=('embedding', 'object')
        )
        return ddf.drop(columns=embedding_cols)

    @abstractmethod
    def process_embeddings(self):
        pass

    @abstractmethod
    def generate_qdrant_points(self):
        pass

class GMFUserEmbeddingProcessor(BaseEmbeddingProcessor):
    def __init__(self, embedding_file_path, user_id_map_path, num_embedding_cols=32):
        super().__init__(embedding_file_path)
        self.user_id_map_path = user_id_map_path
        self.num_embedding_cols = num_embedding_cols

    def process_embeddings(self):
        self._load_parquet_dask()
        combined_ddf = self._combine_embedding_columns(self.raw_embeddings_ddf, self.num_embedding_cols)
        user_id_map_df = pd.read_csv(self.user_id_map_path)
        user_id_map_ddf = dd.from_pandas(user_id_map_df, npartitions=1)
        merged_ddf = combined_ddf.merge(
            user_id_map_ddf,
            left_on='user_id',
            right_on='new_userId',
            how='inner'
        )
        self.processed_embeddings_ddf = merged_ddf[['original_userId', 'embedding']]
        self.processed_embeddings_ddf = self.processed_embeddings_ddf.rename(
            columns={'original_userId': 'id'}
        )
        logging.info("GMF User embeddings processed.")
        return self.processed_embeddings_ddf

    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError("Embeddings not processed yet. Call process_embeddings() first.")
        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        for _, row in computed_df.iterrows():
            user_id_val = str(row['id'])
            points.append(PointStruct(
                id=user_id_val,
                vector=row['embedding'],
                payload={"user_id": user_id_val}
            ))
        logging.info(f"Generated {len(points)} Qdrant points for GMF User embeddings.")
        return points
    
class GMFBookEmbeddingProcessor(BaseEmbeddingProcessor):
    def __init__(self, embedding_file_path, item_id_map_path, num_embedding_cols=32):
        super().__init__(embedding_file_path)
        self.item_id_map_path = item_id_map_path
        self.num_embedding_cols = num_embedding_cols

    def process_embeddings(self):
        self._load_parquet_dask()
        combined_ddf = self._combine_embedding_columns(self.raw_embeddings_ddf, self.num_embedding_cols)
        item_id_map_df = pd.read_csv(self.item_id_map_path)
        item_id_map_ddf = dd.from_pandas(item_id_map_df, npartitions=1)
        merged_ddf = combined_ddf.merge(
            item_id_map_ddf,
            left_on='item_id',
            right_on='new_itemId',
            how='inner'
        )
        self.processed_embeddings_ddf = merged_ddf[['original_itemId', 'embedding']]
        self.processed_embeddings_ddf = self.processed_embeddings_ddf.rename(
            columns={'original_itemId': 'id'}
        )
        logging.info("GMF Book embeddings processed.")
        return self.processed_embeddings_ddf

    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError("Embeddings not processed yet. Call process_embeddings() first.")
        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        correct_book_id_column_gmf = 'id'
        for _, row in computed_df.iterrows():
            try:
                point_id = int(row[correct_book_id_column_gmf])
            except ValueError:
                logging.warning(f"Could not convert ID '{row[correct_book_id_column_gmf]}' to int for GMF book. Using as string.")
                point_id = str(row[correct_book_id_column_gmf])
            points.append(PointStruct(
                id=point_id,
                vector=row['embedding'],
                payload={"item_id": point_id}
            ))
        logging.info(f"Generated {len(points)} Qdrant points for GMF Book embeddings.")
        return points
    
class SBERTEmbeddingProcessor(BaseEmbeddingProcessor):
    def __init__(self, embedding_file_path):
        super().__init__(embedding_file_path)
    def process_embeddings(self):
        self._load_parquet_dask()
        if 'book_id' not in self.raw_embeddings_ddf.columns or \
           'embedding' not in self.raw_embeddings_ddf.columns:
            raise ValueError("SBERT embeddings Dask DataFrame must contain 'book_id' and 'embedding' columns.")
        self.processed_embeddings_ddf = self.raw_embeddings_ddf[['book_id', 'embedding', 'text']].rename(
            columns={'book_id': 'id'}
        )
        logging.info("SBERT Book embeddings processed.")
        return self.processed_embeddings_ddf
    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError("Embeddings not processed yet. Call process_embeddings() first.")
        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        correct_book_id_column_sbert = 'id'
        for _, row in computed_df.iterrows():
            try:
                point_id = int(row[correct_book_id_column_sbert])
            except ValueError:
                logging.warning(f"Could not convert ID '{row[correct_book_id_column_sbert]}' to int for SBERT. Using as string.")
                point_id = str(row[correct_book_id_column_sbert])
            payload = {
                "book_id": point_id,
                "text": row.get("text", "")
            }
            points.append(PointStruct(
                id=point_id,
                vector=row['embedding'],
                payload=payload
            ))
        logging.info(f"Generated {len(points)} Qdrant points for SBERT Book embeddings.")
        return points

class QdrantUploaderStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        self.qdrant_url = "http://localhost:6333"
        self.base_data_path = "data/"
        self.base_embedding_path = "embeddings/"
        self.user_id_map_path = os.path.join(self.base_data_path, 'user_id_map_reduced.csv')
        self.item_id_map_path = os.path.join(self.base_data_path, 'item_id_map_reduced.csv')
        self.gmf_user_embeddings_path = os.path.join(self.base_embedding_path, 'gmf_user_embeddings.parquet')
        self.gmf_book_embeddings_path = os.path.join(self.base_embedding_path, 'gmf_book_embeddings.parquet')
        self.sbert_embeddings_path = os.path.join(self.base_embedding_path, 'sbert_embeddings.parquet')
        self.gmf_user_collection = "gmf_user_embeddings"
        self.gmf_book_collection = "gmf_book_embeddings"
        self.sbert_collection = "sbert_embeddings"
        self.gmf_vector_size = 32
        self.sbert_vector_size = 384
        self.sbert_distance = Distance.COSINE
        self.gmf_distance = Distance.DOT

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self.qdrant_url = self.config.get("qdrant_url", self.qdrant_url)
        self.base_data_path = self.config.get("base_data_path", self.base_data_path)
        self.base_embedding_path = self.config.get("base_embedding_path", self.base_embedding_path)
        self.user_id_map_path = self.config.get("user_id_map_path", self.user_id_map_path)
        self.item_id_map_path = self.config.get("item_id_map_path", self.item_id_map_path)
        self.gmf_user_embeddings_path = self.config.get("gmf_user_embeddings_path", self.gmf_user_embeddings_path)
        self.gmf_book_embeddings_path = self.config.get("gmf_book_embeddings_path", self.gmf_book_embeddings_path)
        self.sbert_embeddings_path = self.config.get("sbert_embeddings_path", self.sbert_embeddings_path)
        self.gmf_user_collection = self.config.get("gmf_user_collection", self.gmf_user_collection)
        self.gmf_book_collection = self.config.get("gmf_book_collection", self.gmf_book_collection)
        self.sbert_collection = self.config.get("sbert_collection", self.sbert_collection)
        self.gmf_vector_size = self.config.get("gmf_vector_size", self.gmf_vector_size)
        self.sbert_vector_size = self.config.get("sbert_vector_size", self.sbert_vector_size)
        self.sbert_distance = self.config.get("sbert_distance", self.sbert_distance)
        self.gmf_distance = self.config.get("gmf_distance", self.gmf_distance)

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting Qdrant upload step...")
        outputs = {}
        qdrant_manager = QdrantManager(url=self.qdrant_url)
        # GMF User Embeddings
        gmf_user_processor = GMFUserEmbeddingProcessor(self.gmf_user_embeddings_path, self.user_id_map_path, num_embedding_cols=self.gmf_vector_size)
        gmf_user_processor.process_embeddings()
        gmf_user_points = gmf_user_processor.generate_qdrant_points()
        qdrant_manager.create_collection_if_not_exists(self.gmf_user_collection, vector_size=self.gmf_vector_size, distance_metric=self.gmf_distance)
        qdrant_manager.batch_upload_points(self.gmf_user_collection, gmf_user_points)
        outputs["gmf_user_collection"] = self.gmf_user_collection
        outputs["gmf_user_count"] = len(gmf_user_points)
        # GMF Book Embeddings
        gmf_book_processor = GMFBookEmbeddingProcessor(self.gmf_book_embeddings_path, self.item_id_map_path, num_embedding_cols=self.gmf_vector_size)
        gmf_book_processor.process_embeddings()
        gmf_book_points = gmf_book_processor.generate_qdrant_points()
        qdrant_manager.create_collection_if_not_exists(self.gmf_book_collection, vector_size=self.gmf_vector_size, distance_metric=self.gmf_distance)
        qdrant_manager.batch_upload_points(self.gmf_book_collection, gmf_book_points)
        outputs["gmf_book_collection"] = self.gmf_book_collection
        outputs["gmf_book_count"] = len(gmf_book_points)
        # SBERT Book Embeddings
        sbert_processor = SBERTEmbeddingProcessor(self.sbert_embeddings_path)
        sbert_processor.process_embeddings()
        sbert_points = sbert_processor.generate_qdrant_points()
        qdrant_manager.create_collection_if_not_exists(self.sbert_collection, vector_size=self.sbert_vector_size, distance_metric=self.sbert_distance)
        qdrant_manager.batch_upload_points(self.sbert_collection, sbert_points)
        outputs["sbert_collection"] = self.sbert_collection
        outputs["sbert_count"] = len(sbert_points)
        self.logger.info("Qdrant upload step finished.")
        self.output_data = outputs
        return outputs

# No main block, this is now a pipeline step