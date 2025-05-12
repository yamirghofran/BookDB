import pandas as pd
import dask.dataframe as dd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from abc import ABC, abstractmethod


class QdrantManager:
    def __init__(self, url="http://localhost:6333"):
        self.client = QdrantClient(url=url)

    def create_collection_if_not_exists(self, collection_name, vector_size, distance_metric):
        try:
            self.client.get_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception: # Replace with specific Qdrant exception if available
            print(f"Creating collection '{collection_name}'...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance_metric),
            )
            print(f"Collection '{collection_name}' created.")

    def batch_upload_points(self, collection_name, points, batch_size=500):
        if not points:
            print(f"No points to upload for collection '{collection_name}'.")
            return

        print(f"Upserting {len(points)} points to '{collection_name}' in batches of {batch_size}...")
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch, wait=True)
        print(f"Uploaded {len(points)} points to '{collection_name}'.")

class BaseEmbeddingProcessor(ABC):
    def __init__(self, embedding_file_path):
        self.embedding_file_path = embedding_file_path
        self.raw_embeddings_ddf = None
        self.processed_embeddings_ddf = None

    def _load_parquet_dask(self):
        print(f"Loading embeddings from {self.embedding_file_path}...")
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
        """
        Loads and processes embeddings.
        Should set self.processed_embeddings_ddf to a Dask DataFrame
        with at least an 'id' column and an 'embedding' column.
        """
        pass

    @abstractmethod
    def generate_qdrant_points(self):
        """
        Generates a list of Qdrant PointStruct objects from self.processed_embeddings_ddf.
        """
        pass

class GMFUserEmbeddingProcessor(BaseEmbeddingProcessor):
    def __init__(self, embedding_file_path, user_id_map_path, num_embedding_cols=32):
        super().__init__(embedding_file_path)
        self.user_id_map_path = user_id_map_path
        self.num_embedding_cols = num_embedding_cols

    def process_embeddings(self):
        self._load_parquet_dask()
        
        # Combine embedding columns
        combined_ddf = self._combine_embedding_columns(self.raw_embeddings_ddf, self.num_embedding_cols)
        
        # Load user ID map
        user_id_map_df = pd.read_csv(self.user_id_map_path)
        user_id_map_ddf = dd.from_pandas(user_id_map_df, npartitions=1) # Convert to Dask DF

        # Merge with user ID map
        merged_ddf = combined_ddf.merge(
            user_id_map_ddf,
            left_on='user_id', # Assumes 'user_id' is the key in gmf_user_embeddings
            right_on='new_userId',
            how='inner'
        )
        
        # Select and rename columns
        self.processed_embeddings_ddf = merged_ddf[['original_userId', 'embedding']]
        self.processed_embeddings_ddf = self.processed_embeddings_ddf.rename(
            columns={'original_userId': 'id'}
        )
        print("GMF User embeddings processed.")
        return self.processed_embeddings_ddf

    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError("Embeddings not processed yet. Call process_embeddings() first.")
        
        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        for _, row in computed_df.iterrows():
            user_id_val = str(row['id']) # Ensure ID is string for Qdrant
            points.append(PointStruct(
                id=user_id_val,
                vector=row['embedding'],
                payload={"user_id": user_id_val}
            ))
        print(f"Generated {len(points)} Qdrant points for GMF User embeddings.")
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
            left_on='item_id', # Assumes 'item_id' is the key in gmf_book_embeddings
            right_on='new_itemId',
            how='inner'
        )
        
        self.processed_embeddings_ddf = merged_ddf[['original_itemId', 'embedding']]
        self.processed_embeddings_ddf = self.processed_embeddings_ddf.rename(
            columns={'original_itemId': 'id'}
        )
        print("GMF Book embeddings processed.")
        return self.processed_embeddings_ddf

    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError("Embeddings not processed yet. Call process_embeddings() first.")

        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        correct_book_id_column_gmf = 'id' # 'id' is the renamed 'original_itemId'
        for _, row in computed_df.iterrows():
            try:
                point_id = int(row[correct_book_id_column_gmf])
            except ValueError:
                print(f"Warning: Could not convert ID '{row[correct_book_id_column_gmf]}' to int for GMF book. Using as string.")
                point_id = str(row[correct_book_id_column_gmf])
            
            points.append(PointStruct(
                id=point_id,
                vector=row['embedding'],
                payload={"item_id": point_id} # Use the original name for payload consistency
            ))
        print(f"Generated {len(points)} Qdrant points for GMF Book embeddings.")
        return points
    
class SBERTEmbeddingProcessor(BaseEmbeddingProcessor):
    def __init__(self, embedding_file_path):
        super().__init__(embedding_file_path)
        # SBERT embeddings might already have 'embedding' as a list and 'book_id'
        # No num_embedding_cols needed if 'embedding' column is already a list/vector

    def process_embeddings(self):
        self._load_parquet_dask()
        # Assuming SBERT parquet has 'book_id' and 'embedding' (as a list)
        # and potentially a 'text' column.
        # If 'embedding' is not a list but separate columns, you'd call _combine_embedding_columns here.
        
        # Ensure required columns exist
        if 'book_id' not in self.raw_embeddings_ddf.columns or \
           'embedding' not in self.raw_embeddings_ddf.columns:
            raise ValueError("SBERT embeddings Dask DataFrame must contain 'book_id' and 'embedding' columns.")

        self.processed_embeddings_ddf = self.raw_embeddings_ddf[['book_id', 'embedding', 'text']].rename(
            columns={'book_id': 'id'}
        ) # Include 'text' if available
        print("SBERT Book embeddings processed.")
        return self.processed_embeddings_ddf

    def generate_qdrant_points(self):
        if self.processed_embeddings_ddf is None:
            raise ValueError("Embeddings not processed yet. Call process_embeddings() first.")

        computed_df = self.processed_embeddings_ddf.compute()
        points = []
        correct_book_id_column_sbert = 'id' # 'id' is the renamed 'book_id'

        for _, row in computed_df.iterrows():
            try:
                point_id = int(row[correct_book_id_column_sbert])
            except ValueError:
                print(f"Warning: Could not convert ID '{row[correct_book_id_column_sbert]}' to int for SBERT. Using as string.")
                point_id = str(row[correct_book_id_column_sbert])
            
            payload = {
                "book_id": point_id, # Use the original name for payload consistency
                "text": row.get("text", "") # Include 'text' in payload, handle if missing
            }
            points.append(PointStruct(
                id=point_id,
                vector=row['embedding'],
                payload=payload
            ))
        print(f"Generated {len(points)} Qdrant points for SBERT Book embeddings.")
        return points
    
# --- Main Pipeline ---
def run_upload_pipeline():
    # Initialize Qdrant Manager
    qdrant_manager = QdrantManager(url="http://localhost:6333")

    # Define paths (adjust as per your project structure)
    base_data_path = '../data/'
    base_embedding_path = '../embeddings/' # Assuming your script is in 'scripts' and embeddings are in 'embeddings'

    user_id_map_path = os.path.join(base_data_path, 'user_id_map_reduced.csv')
    item_id_map_path = os.path.join(base_data_path, 'item_id_map_reduced.csv')
    
    gmf_user_embeddings_path = os.path.join(base_embedding_path, 'gmf_user_embeddings.parquet')
    gmf_book_embeddings_path = os.path.join(base_embedding_path, 'gmf_book_embeddings.parquet')
    sbert_embeddings_path = os.path.join(base_embedding_path, 'sbert_embeddings.parquet') # Corrected path

    # --- Process and Upload GMF User Embeddings ---
    print("\n--- Processing GMF User Embeddings ---")
    gmf_user_processor = GMFUserEmbeddingProcessor(gmf_user_embeddings_path, user_id_map_path, num_embedding_cols=32)
    gmf_user_processor.process_embeddings()
    gmf_user_points = gmf_user_processor.generate_qdrant_points()
    
    qdrant_manager.create_collection_if_not_exists("gmf_user_embeddings", vector_size=32, distance_metric=Distance.DOT)
    qdrant_manager.batch_upload_points("gmf_user_embeddings", gmf_user_points)
    print("-" * 30)

    # --- Process and Upload GMF Book Embeddings ---
    print("\n--- Processing GMF Book Embeddings ---")
    gmf_book_processor = GMFBookEmbeddingProcessor(gmf_book_embeddings_path, item_id_map_path, num_embedding_cols=32)
    gmf_book_processor.process_embeddings()
    gmf_book_points = gmf_book_processor.generate_qdrant_points()

    qdrant_manager.create_collection_if_not_exists("gmf_book_embeddings", vector_size=32, distance_metric=Distance.DOT)
    qdrant_manager.batch_upload_points("gmf_book_embeddings", gmf_book_points)
    print("-" * 30)

    # --- Process and Upload SBERT Book Embeddings ---
    print("\n--- Processing SBERT Book Embeddings ---")
    # Note: SBERT embeddings might have a different vector size (e.g., 384 for SBERT base models)
    # And the 'embedding' column might already be a list. Adjust SBERTEmbeddingProcessor if needed.
    sbert_processor = SBERTEmbeddingProcessor(sbert_embeddings_path)
    sbert_processor.process_embeddings()
    sbert_points = sbert_processor.generate_qdrant_points()

    qdrant_manager.create_collection_if_not_exists("sbert_embeddings", vector_size=384, distance_metric=Distance.COSINE) # Assuming 384 for SBERT
    qdrant_manager.batch_upload_points("sbert_embeddings", sbert_points)
    print("-" * 30)

    print("\nAll uploads complete.")

if __name__ == "__main__":
    import os # Make sure os is imported if using os.path.join
    # Set pandas display options if needed (usually not for a script like this)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    run_upload_pipeline()