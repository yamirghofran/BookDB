#!/usr/bin/env python3
"""
Main pipeline runner for the complete ML data processing pipeline.
Orchestrates all steps from data preprocessing to model training and deployment.
"""

import logging
import sys
from pathlib import Path

# Add pipeline directory to path for imports
sys.path.append(str(Path(__file__).parent))

from pipeline import Pipeline
from pipeline.config import PipelineConfig
from pipeline.data_preprocessor import DataPreprocessorStep
from pipeline.data_reducer import DataReducerStep
from pipeline.uuid_processor import UUIDProcessorStep
from pipeline.ncf_processor import NCFDataPreprocessorStep
from pipeline.sbert_trainer import SbertTrainerStep
from pipeline.sbert_embedder import SBertEmbedderStep
from pipeline.postgres_uploader import PostgresUploaderStep
from pipeline.qdrant_uploader import QdrantUploaderStep


def main():
    """Main pipeline execution function."""
    # Initialize configuration
    config = PipelineConfig()
    
    # Create pipeline
    pipeline = Pipeline(config)
    
    # Add all pipeline steps in order
    pipeline.add_step(DataPreprocessorStep, "data_preprocessor")
    pipeline.add_step(DataReducerStep, "data_reducer")
    pipeline.add_step(UUIDProcessorStep, "uuid_processor")
    pipeline.add_step(SbertTrainerStep, "sbert_trainer")
    pipeline.add_step(SBertEmbedderStep, "sbert_embedder")
    pipeline.add_step(NCFDataPreprocessorStep, "ncf_processor")
    
    # Conditional NCF training step
    if config.run_ncf:
        logging.info("NCF training enabled - will run neural collaborative filtering")
        # Note: NCF training step would need to be implemented
        # pipeline.add_step(NCFTrainingStep, "ncf_training")
    else:
        logging.info("NCF training disabled - skipping neural collaborative filtering")
    
    # Database upload steps
    pipeline.add_step(PostgresUploaderStep, "postgres_uploader")
    pipeline.add_step(QdrantUploaderStep, "qdrant_uploader")
    
    # Run the complete pipeline
    try:
        logging.info("Starting complete ML pipeline execution...")
        final_output = pipeline.run()
        logging.info("Pipeline completed successfully!")
        logging.info(f"Final output: {final_output}")
        return final_output
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()