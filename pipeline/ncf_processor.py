import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import datetime
from typing import Dict, Any
from .core import PipelineStep
from utils import send_discord_webhook

class NCFDataPreprocessorStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        self.interactions_path = "data/reduced_interactions.parquet"
        self.user_map_path = "data/user_id_map.csv"
        self.book_map_path = "data/book_id_map.csv"
        self.output_dir = "data/ncf/"
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
        self.user_map_output_path = os.path.join(self.output_dir, "ncf_user_id_map_reduced.csv")
        self.item_map_output_path = os.path.join(self.output_dir, "ncf_item_id_map_reduced.csv")
        self.final_output_path = os.path.join(self.output_dir, "interactions_prepared_ncf_reduced.parquet")

    def _send_notification(self, title: str, description: str, color: int = 0x00FF00, fields: list = None, error: bool = False):
        """Send a Discord notification with consistent formatting."""
        try:
            embed = {
                "title": f"🤖 {title}" if not error else f"❌ {title}",
                "description": description,
                "color": color if not error else 0xFF0000,  # Red for errors
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "footer": {"text": f"Pipeline Step: {self.name}"}
            }
            
            if fields:
                embed["fields"] = fields
                
            send_discord_webhook(
                content=None,
                embed=embed,
                username="BookDB Pipeline"
            )
        except Exception as e:
            self.logger.warning(f"Failed to send Discord notification: {e}")

    def load_data(self):
        try:
            self.logger.info("Loading datasets...")
            self.interactions_reduced_df = dd.read_parquet(self.interactions_path)
            self.user_id_map_orig = pd.read_csv(self.user_map_path)
            self.book_id_map_orig = pd.read_csv(self.book_map_path)
            self.logger.info("Datasets loaded.")
            
            # Send success notification
            interactions_count = len(self.interactions_reduced_df.compute())
            self._send_notification(
                "NCF Data Loading Complete",
                f"Successfully loaded datasets for NCF preprocessing",
                fields=[
                    {"name": "Interactions File", "value": f"`{os.path.basename(self.interactions_path)}`", "inline": True},
                    {"name": "User Map File", "value": f"`{os.path.basename(self.user_map_path)}`", "inline": True},
                    {"name": "Book Map File", "value": f"`{os.path.basename(self.book_map_path)}`", "inline": True},
                    {"name": "Interactions Count", "value": f"{interactions_count:,}", "inline": True},
                    {"name": "Original Users", "value": f"{len(self.user_id_map_orig):,}", "inline": True},
                    {"name": "Original Books", "value": f"{len(self.book_id_map_orig):,}", "inline": True}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to load NCF data: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "NCF Data Loading Failed",
                error_msg,
                error=True
            )
            raise

    def _preprocess_interactions(self):
        try:
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
            
            # Send success notification
            self._send_notification(
                "NCF Interactions Preprocessing Complete",
                f"Successfully preprocessed interactions for NCF format",
                fields=[
                    {"name": "Columns Renamed", "value": "user_id → userId, book_id → itemId", "inline": False},
                    {"name": "Timestamp Processing", "value": "Converted to Unix timestamp", "inline": True},
                    {"name": "Final Columns", "value": "userId, itemId, rating, timestamp", "inline": True}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to preprocess interactions: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "NCF Interactions Preprocessing Failed",
                error_msg,
                error=True
            )
            raise

    def _create_and_save_mappings(self):
        if self.interactions_prepared_df is None:
            raise ValueError("Interactions must be preprocessed before creating mappings.")
        try:
            self.logger.info("Creating reduced mappings...")
            unique_users = self.interactions_prepared_df['userId'].unique().compute()
            unique_items = self.interactions_prepared_df['itemId'].unique().compute()
            self.user_map = pd.Series(range(len(unique_users)), index=unique_users)
            self.item_map = pd.Series(range(len(unique_items)), index=unique_items)
            user_map_df = self.user_map.reset_index()
            user_map_df.columns = ['userId', 'ncf_userId']
            user_map_df.to_csv(self.user_map_output_path, index=False)
            self.logger.info(f"Reduced user ID mapping saved to {self.user_map_output_path}. Total users: {len(user_map_df)}")
            item_map_df = self.item_map.reset_index()
            item_map_df.columns = ['itemId', 'ncf_itemId']
            item_map_df.to_csv(self.item_map_output_path, index=False)
            self.logger.info(f"Reduced item ID mapping saved to {self.item_map_output_path}. Total items: {len(item_map_df)}")
            self.logger.info("Mappings created and saved.")
            
            # Send success notification
            self._send_notification(
                "NCF ID Mappings Created",
                f"Successfully created continuous integer mappings for NCF model",
                fields=[
                    {"name": "Unique Users", "value": f"{len(unique_users):,}", "inline": True},
                    {"name": "Unique Items", "value": f"{len(unique_items):,}", "inline": True},
                    {"name": "User Map Range", "value": f"0 to {len(unique_users)-1}", "inline": True},
                    {"name": "Item Map Range", "value": f"0 to {len(unique_items)-1}", "inline": True},
                    {"name": "User Map File", "value": f"`{os.path.basename(self.user_map_output_path)}`", "inline": True},
                    {"name": "Item Map File", "value": f"`{os.path.basename(self.item_map_output_path)}`", "inline": True}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to create NCF mappings: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "NCF ID Mappings Creation Failed",
                error_msg,
                error=True
            )
            raise

    def _apply_mappings(self):
        if self.interactions_prepared_df is None or self.user_map is None or self.item_map is None:
            raise ValueError("Data must be loaded, preprocessed, and mappings created before applying them.")
        try:
            self.logger.info("Applying mappings to interactions DataFrame...")
            interactions_final_df = self.interactions_prepared_df.copy()
            interactions_final_df['userId'] = interactions_final_df['userId'].map(self.user_map, meta=('userId', 'int64'))
            interactions_final_df['itemId'] = interactions_final_df['itemId'].map(self.item_map, meta=('itemId', 'int64'))
            self.interactions_final_df = interactions_final_df
            self.logger.info("Mappings applied.")
            self.logger.debug(f"Head of final DataFrame with new integer IDs: {self.interactions_final_df.head()}")
            
            # Send success notification
            self._send_notification(
                "NCF ID Mappings Applied",
                f"Successfully applied continuous integer IDs to interactions dataset",
                fields=[
                    {"name": "User IDs", "value": f"Mapped to 0-{len(self.user_map)-1}", "inline": True},
                    {"name": "Item IDs", "value": f"Mapped to 0-{len(self.item_map)-1}", "inline": True},
                    {"name": "Data Ready", "value": "✅ For NCF model training", "inline": True}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to apply NCF mappings: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "NCF ID Mappings Application Failed",
                error_msg,
                error=True
            )
            raise

    def _save_final_data(self):
        if self.interactions_final_df is None:
            raise ValueError("Final interactions DataFrame not available to save.")
        try:
            self.logger.info("Saving final processed data...")
            self.interactions_final_df.repartition(npartitions=1).to_parquet(
                self.final_output_path,
                write_index=False
            )
            self.logger.info(f"Prepared interactions DataFrame saved to single file: {self.final_output_path}")
            
            # Get final stats
            final_count = len(self.interactions_final_df.compute())
            
            # Send success notification
            self._send_notification(
                "NCF Data Preparation Complete",
                f"Successfully saved NCF-ready dataset",
                fields=[
                    {"name": "Final Records", "value": f"{final_count:,}", "inline": True},
                    {"name": "Output File", "value": f"`{os.path.basename(self.final_output_path)}`", "inline": True},
                    {"name": "Format", "value": "Single Parquet file", "inline": True},
                    {"name": "Ready For", "value": "🤖 NCF Model Training", "inline": True}
                ]
            )
        except Exception as e:
            error_msg = f"Failed to save NCF final data: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "NCF Data Saving Failed",
                error_msg,
                error=True
            )
            raise

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting NCF data preparation step...")
        
        # Send pipeline start notification
        self._send_notification(
            "NCF Data Preparation Started",
            f"Beginning NCF preprocessing pipeline: **{self.name}**",
            color=0x0099FF,  # Blue for start
            fields=[
                {"name": "Input Directory", "value": f"`{os.path.dirname(self.interactions_path)}`", "inline": True},
                {"name": "Output Directory", "value": f"`{self.output_dir}`", "inline": True},
                {"name": "Target Model", "value": "Neural Collaborative Filtering", "inline": False}
            ]
        )
        
        try:
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
            
            # Send pipeline completion notification
            self._send_notification(
                "NCF Data Preparation Complete! 🎉",
                f"All NCF preprocessing tasks completed successfully for pipeline: **{self.name}**",
                color=0x00FF00,  # Green for success
                fields=[
                    {"name": "Data Loading", "value": "✅ Complete", "inline": True},
                    {"name": "Preprocessing", "value": "✅ Complete", "inline": True},
                    {"name": "ID Mappings", "value": "✅ Complete", "inline": True},
                    {"name": "Final Dataset", "value": "✅ Complete", "inline": True},
                    {"name": "Users Mapped", "value": f"{len(self.user_map):,}", "inline": True},
                    {"name": "Items Mapped", "value": f"{len(self.item_map):,}", "inline": True},
                    {"name": "Output Directory", "value": f"`{self.output_dir}`", "inline": False},
                    {"name": "Ready For", "value": "🤖 NCF Model Training", "inline": True}
                ]
            )
            
            self.output_data = outputs
            return outputs
            
        except Exception as e:
            error_msg = f"NCF data preparation pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "NCF Data Preparation Pipeline Failed",
                error_msg,
                error=True
            )
            raise