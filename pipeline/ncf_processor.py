import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
from typing import Dict, Any
from .core import PipelineStep

class NCFDataPreprocessorStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        self.interactions_path = "data/reduced_interactions.parquet"
        self.user_map_path = "data/user_id_map.csv"
        self.book_map_path = "data/book_id_map.csv"
        self.output_dir = "data/processed_ncf/"
        self.interactions_prepared_df = None
        self.user_map = None
        self.item_map = None
        self.interactions_final_df = None
        self.user_map_output_path = None
        self.item_map_output_path = None
        self.final_output_path = None

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self.interactions_path = self.config.get("interactions_path", self.interactions_path)
        self.user_map_path = self.config.get("user_map_path", self.user_map_path)
        self.book_map_path = self.config.get("book_map_path", self.book_map_path)
        self.output_dir = self.config.get("output_dir", self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.user_map_output_path = os.path.join(self.output_dir, "user_id_map_reduced.csv")
        self.item_map_output_path = os.path.join(self.output_dir, "item_id_map_reduced.csv")
        self.final_output_path = os.path.join(self.output_dir, "interactions_prepared_ncf_reduced.parquet")

    def load_data(self):
        self.logger.info("Loading datasets...")
        self.interactions_reduced_df = dd.read_parquet(self.interactions_path)
        # self.user_id_map_orig = pd.read_csv(self.user_map_path) # Original maps, if needed
        # self.book_id_map_orig = pd.read_csv(self.book_map_path) # Original maps, if needed
        self.logger.info("Datasets loaded.")

    def _preprocess_interactions(self):
        self.logger.info("Preprocessing interactions...")
        interactions_df = self.interactions_reduced_df[['user_id', 'book_id', 'rating', 'date_updated']].rename(columns={
            'user_id': 'userId',
            'book_id': 'itemId',
            'date_updated': 'timestamp'
        })
        interactions_df['timestamp'] = dd.to_datetime(
            interactions_df['timestamp'],
            format='%a %b %d %H:%M:%S %z %Y',
            errors='coerce',
            utc=True
        )
        interactions_df['timestamp'] = (interactions_df['timestamp'].astype(np.int64) // 10**9)
        self.interactions_prepared_df = interactions_df
        self.logger.info("Interactions preprocessed.")
        self.logger.debug(f"Head of prepared interactions: {self.interactions_prepared_df.head()}")

    def _create_and_save_mappings(self):
        if self.interactions_prepared_df is None:
            raise ValueError("Interactions must be preprocessed before creating mappings.")
        self.logger.info("Creating reduced mappings...")
        unique_users = self.interactions_prepared_df['userId'].unique().compute()
        unique_items = self.interactions_prepared_df['itemId'].unique().compute()
        self.user_map = pd.Series(range(len(unique_users)), index=unique_users)
        self.item_map = pd.Series(range(len(unique_items)), index=unique_items)
        user_map_df = self.user_map.reset_index()
        user_map_df.columns = ['original_userId', 'new_userId']
        user_map_df.to_csv(self.user_map_output_path, index=False)
        self.logger.info(f"Reduced user ID mapping saved to {self.user_map_output_path}. Total users: {len(user_map_df)}")
        item_map_df = self.item_map.reset_index()
        item_map_df.columns = ['original_itemId', 'new_itemId']
        item_map_df.to_csv(self.item_map_output_path, index=False)
        self.logger.info(f"Reduced item ID mapping saved to {self.item_map_output_path}. Total items: {len(item_map_df)}")
        self.logger.info("Mappings created and saved.")

    def _apply_mappings(self):
        if self.interactions_prepared_df is None or self.user_map is None or self.item_map is None:
            raise ValueError("Data must be loaded, preprocessed, and mappings created before applying them.")
        self.logger.info("Applying mappings to interactions DataFrame...")
        interactions_final_df = self.interactions_prepared_df.copy()
        interactions_final_df['userId'] = interactions_final_df['userId'].map(self.user_map, meta=('userId', 'int64'))
        interactions_final_df['itemId'] = interactions_final_df['itemId'].map(self.item_map, meta=('itemId', 'int64'))
        self.interactions_final_df = interactions_final_df
        self.logger.info("Mappings applied.")
        self.logger.debug(f"Head of final DataFrame with new integer IDs: {self.interactions_final_df.head()}")
        
    def _save_final_data(self):
        if self.interactions_final_df is None:
            raise ValueError("Final interactions DataFrame not available to save.")
        self.logger.info("Saving final processed data...")
        self.interactions_final_df.repartition(npartitions=1).to_parquet(
            self.final_output_path,
            write_index=False
        )
        self.logger.info(f"Prepared interactions DataFrame saved to single file: {self.final_output_path}")

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting NCF data preparation step...")
        outputs = {}
        self.load_data()
        self._preprocess_interactions()
        self._create_and_save_mappings()
        self._apply_mappings()
        self._save_final_data()
        outputs["user_map_csv"] = self.user_map_output_path
        outputs["item_map_csv"] = self.item_map_output_path
        outputs["interactions_prepared_parquet"] = self.final_output_path
        self.logger.info("NCF data preparation pipeline finished.")
        self.output_data = outputs
        return outputs