# Configuration for the complete ML pipeline
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import os

@dataclass
class GlobalConfig:
    """Global configuration settings."""
    base_data_path: Path = Path("data")
    log_level: str = "INFO"
    random_state: int = 42
    
    def __post_init__(self):
        # Ensure base data path exists
        self.base_data_path.mkdir(exist_ok=True)

@dataclass
class DataPreprocessorConfig:
    """Configuration for data preprocessing step."""
    # Input files
    books_input_json: str = "goodreads_books.json.gz"
    interactions_csv_input: str = "goodreads_interactions.csv"
    interactions_dedup_input_json: str = "goodreads_interactions_dedup.json.gz"
    reviews_input_json: str = "goodreads_reviews_dedup.json.gz"
    book_works_input_json: str = "goodreads_book_works.json.gz"
    authors_input_json: str = "goodreads_book_authors.json.gz"
    
    # Output files
    books_output_parquet: str = "books.parquet"
    interactions_csv_output_parquet: str = "interactions.parquet"
    interactions_dedup_output_parquet: str = "interactions_dedup.parquet"
    reviews_output_parquet: str = "reviews.parquet"
    book_works_output_parquet: str = "books_works.parquet"
    authors_output_parquet: str = "authors.parquet"
    
    # ID mapping files
    book_id_map_path: str = "data/book_id_map.csv"
    user_id_map_path: str = "data/user_id_map.csv"
    
    # Processing parameters
    interactions_dedup_chunk_size: int = 150000
    interactions_dedup_progress_file: str = "chunk_progress.txt"
    
    def get_input_path(self, global_config: GlobalConfig) -> Path:
        return global_config.base_data_path
    
    def get_output_path(self, global_config: GlobalConfig) -> Path:
        return global_config.base_data_path

@dataclass
class DataReducerConfig:
    """Configuration for data reduction step."""
    # Filtering parameters
    min_rating: int = 4
    min_user_interactions: int = 100
    min_book_interactions: int = 500
    
    # Input files
    interactions_dedup_file: str = "interactions_dedup.parquet"
    books_file: str = "new_books.parquet"
    reviews_file: str = "reviews_dedup.parquet"
    
    # Output files
    reduced_user_ids_file: str = "reduced_user_ids.csv"
    reduced_book_ids_file: str = "reduced_book_ids.csv"
    reduced_interactions_file: str = "reduced_interactions.parquet"
    reduced_books_file: str = "reduced_books.parquet"
    reduced_reviews_file: str = "reduced_reviews.parquet"
    
    def get_input_path(self, global_config: GlobalConfig) -> Path:
        return global_config.base_data_path
    
    def get_output_path(self, global_config: GlobalConfig) -> Path:
        return global_config.base_data_path

@dataclass
class UUIDProcessorConfig:
    """Configuration for UUID conversion step."""
    # Input files
    authors_input_path: str = "data/authors.parquet"
    books_input_path: str = "data/books.parquet"
    
    # Output files
    new_authors_output_path: str = "data/new_authors.parquet"
    new_books_output_path: str = "data/new_books.parquet"
    author_map_csv_path: str = "data/author_id_map.csv"

@dataclass
class SBERTTrainerConfig:
    """Configuration for SBERT fine-tuning step."""
    # Model parameters
    model_name: str = "all-MiniLM-L6-v2"
    random_state: int = 42
    
    # Input files
    books_data_file: str = "data/reduced_books.parquet"
    authors_data_file: str = "data/new_authors.parquet"
    
    # Output files
    book_texts_file: str = "data/book_texts.parquet"
    triplets_data_file: str = "data/books_triplets.parquet"
    output_base_path: str = "sbert-output"
    
    # Training parameters
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    triplet_margin: float = 0.5
    warmup_steps_ratio: float = 0.1
    evaluation_steps: int = 500
    save_steps: int = 500
    checkpoint_limit: int = 3
    
    # Data split parameters
    test_split_size: float = 0.2
    validation_split_size: float = 0.1
    max_negative_search_attempts: int = 10
    
    def __post_init__(self):
        self.eval_output_path = os.path.join(self.output_base_path, 'eval')
        self.checkpoint_path = os.path.join(self.output_base_path, 'checkpoints')

@dataclass
class SBERTEmbedderConfig:
    """Configuration for SBERT embedding generation step."""
    # Model path
    model_path: str = "sbert-output/finetuning-all-MiniLM-L6-v2-books"
    
    # Input/Output
    input_file: str = "data/book_texts.parquet"
    output_file: str = "embeddings/sbert_embeddings.parquet"
    
    # Processing parameters
    batch_size: int = 256
    chunk_size: int = 20000
    id_column: str = "book_id"
    text_column: str = "text"

@dataclass
class NCFProcessorConfig:
    """Configuration for NCF data preparation step."""
    # Input files
    interactions_input_file: str = "data/reduced_interactions.parquet"
    user_id_map_file: str = "data/user_id_map.csv"
    book_id_map_file: str = "data/book_id_map.csv"
    
    # Output files
    final_output_file: str = "data/interactions_prepared_ncf_reduced.parquet"
    user_map_output_file: str = "data/ncf_user_id_map_reduced.csv"
    item_map_output_file: str = "data/ncf_item_id_map_reduced.csv"
    
    # Date format for timestamp conversion
    timestamp_format: str = "%a %b %d %H:%M:%S %z %Y"

@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD"))
    
    def get_connection_dict(self) -> Dict[str, Any]:
        return {
            "dbname": self.name,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port
        }

@dataclass
class PostgresUploaderConfig:
    """Configuration for PostgreSQL upload step."""
    # Input data files
    books_file: str = "data/reduced_books.parquet"
    interactions_file: str = "data/reduced_interactions.parquet"
    reviews_file: str = "data/reduced_reviews.parquet"
    authors_file: str = "data/new_authors.parquet"
    users_file: str = "data/ncf_user_id_map_reduced.csv"
    item_id_map_file: str = "data/ncf_item_id_map_reduced.csv"
    
    # Processing parameters
    batch_size: int = 1000
    log_level: str = "DEBUG"

@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    
    # Collection configurations
    sbert_collection: str = "sbert_embeddings"
    gmf_user_collection: str = "gmf_user_embeddings"
    gmf_book_collection: str = "gmf_book_embeddings"
    
    # Vector configurations
    sbert_vector_size: int = 384
    gmf_vector_size: int = 32
    
    # Distance metrics
    sbert_distance: str = "COSINE"
    gmf_distance: str = "DOT"
    
    # Upload parameters
    batch_size: int = 500

@dataclass
class QdrantUploaderConfig:
    """Configuration for Qdrant upload step."""
    # Input files
    sbert_embeddings_path: str = "embeddings/sbert_embeddings.parquet"
    gmf_user_embeddings_path: str = "embeddings/gmf_user_embeddings.parquet"
    gmf_book_embeddings_path: str = "embeddings/gmf_book_embeddings.parquet"
    user_id_map_path: str = "data/ncf_user_id_map_reduced.csv"
    item_id_map_path: str = "data/ncf_item_id_map_reduced.csv"

@dataclass
class NCFTrainingConfig:
    """Configuration for NCF training."""
    # Data paths
    prepared_data_path: str = "data/interactions_prepared_ncf_reduced.parquet"
    ncf_target_dir: str = "neural-collaborative-filtering/src/data"
    ncf_target_file: str = "interactions.parquet"
    results_dir: str = "results/ncf"
    
    # Script paths
    train_script: str = "neural-collaborative-filtering/src/train.py"
    export_script: str = "neural-collaborative-filtering/src/export_gmf_embeddings.py"
    
    # Training parameters
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.001
    embedding_dim: int = 32

@dataclass
class PipelineConfig:
    """Main pipeline configuration containing all step configs."""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    data_preprocessor: DataPreprocessorConfig = field(default_factory=DataPreprocessorConfig)
    data_reducer: DataReducerConfig = field(default_factory=DataReducerConfig)
    uuid_processor: UUIDProcessorConfig = field(default_factory=UUIDProcessorConfig)
    sbert_trainer: SBERTTrainerConfig = field(default_factory=SBERTTrainerConfig)
    sbert_embedder: SBERTEmbedderConfig = field(default_factory=SBERTEmbedderConfig)
    ncf_processor: NCFProcessorConfig = field(default_factory=NCFProcessorConfig)
    ncf_training: NCFTrainingConfig = field(default_factory=NCFTrainingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    postgres_uploader: PostgresUploaderConfig = field(default_factory=PostgresUploaderConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    qdrant_uploader: QdrantUploaderConfig = field(default_factory=QdrantUploaderConfig)
    
    # Pipeline control
    run_ncf: bool = field(default_factory=lambda: os.getenv("RUN_NCF", "true").lower() == "true")
    
    def __post_init__(self):
        """Post-initialization to set up computed paths and create directories."""
        # Ensure all output directories exist
        self.global_config.base_data_path.mkdir(exist_ok=True)
        (self.global_config.base_data_path).mkdir(exist_ok=True)
        Path(self.sbert_trainer.output_base_path).mkdir(exist_ok=True)
        Path(self.sbert_trainer.eval_output_path).mkdir(exist_ok=True)
        Path(self.sbert_trainer.checkpoint_path).mkdir(exist_ok=True)
        Path("embeddings").mkdir(exist_ok=True)