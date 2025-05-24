import pandas as pd
import uuid
import os
import datetime
from typing import Dict, Any
from .core import PipelineStep
from utils import send_discord_webhook

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class UUIDProcessorStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        # Defaults, can be overridden by config
        self.authors_input_path = "data/authors.parquet"
        self.books_input_path = "data/books.parquet"
        self.new_authors_output_path = "data/new_authors.parquet"
        self.new_books_output_path = "data/new_books.parquet"
        self.author_map_csv_path = "data/author_id_map.csv"
        self.author_id_map_dict = {}

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self.authors_input_path = self.config.get("authors_input_path", self.authors_input_path)
        self.books_input_path = self.config.get("books_input_path", self.books_input_path)
        self.new_authors_output_path = self.config.get("new_authors_output_path", self.new_authors_output_path)
        self.new_books_output_path = self.config.get("new_books_output_path", self.new_books_output_path)
        self.author_map_csv_path = self.config.get("author_map_csv_path", self.author_map_csv_path)
        # Ensure output directories exist
        for path in [self.new_authors_output_path, self.new_books_output_path, self.author_map_csv_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def _send_notification(self, title: str, description: str, color: int = 0x00FF00, fields: list = None, error: bool = False):
        """Send a Discord notification with consistent formatting."""
        try:
            embed = {
                "title": f"🔄 {title}" if not error else f"❌ {title}",
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

    def _generate_author_id_map(self, authors_df):
        self.author_id_map_dict = {
            old_id: str(uuid.uuid4())
            for old_id in authors_df['author_id'].unique()
        }
        return self.author_id_map_dict

    def _save_author_id_map_to_csv(self):
        if not self.author_id_map_dict:
            raise ValueError("Author ID map has not been generated yet.")
        map_items = list(self.author_id_map_dict.items())
        author_id_map_df = pd.DataFrame(map_items, columns=['old_id', 'new_id'])
        author_id_map_df.to_csv(self.author_map_csv_path, index=False)
        self.logger.info(f"Author ID map saved to {self.author_map_csv_path}")

    def _load_author_id_map_from_csv(self):
        try:
            author_id_map_df = pd.read_csv(self.author_map_csv_path)
            return pd.Series(
                author_id_map_df.new_id.values,
                index=author_id_map_df.old_id.astype(str)
            ).to_dict()
        except FileNotFoundError:
            self.logger.warning(f"Author ID map file not found at {self.author_map_csv_path}.")
            return {}

    def process_authors(self):
        try:
            authors_df = pd.read_parquet(self.authors_input_path)
            self._generate_author_id_map(authors_df)
            self._save_author_id_map_to_csv()
            new_authors_df = authors_df.copy()
            new_authors_df['author_id'] = new_authors_df['author_id'].map(self.author_id_map_dict)
            new_authors_df.to_parquet(self.new_authors_output_path)
            self.logger.info(f"New authors dataframe saved to {self.new_authors_output_path}")
            
            # Send success notification
            self._send_notification(
                "Author UUID Processing Complete",
                f"Successfully converted **{len(self.author_id_map_dict):,}** author IDs to UUIDs",
                fields=[
                    {"name": "Input File", "value": f"`{os.path.basename(self.authors_input_path)}`", "inline": True},
                    {"name": "Output File", "value": f"`{os.path.basename(self.new_authors_output_path)}`", "inline": True},
                    {"name": "Authors Processed", "value": f"{len(authors_df):,}", "inline": True},
                    {"name": "UUID Mappings", "value": f"{len(self.author_id_map_dict):,}", "inline": True},
                    {"name": "ID Map File", "value": f"`{os.path.basename(self.author_map_csv_path)}`", "inline": True}
                ]
            )
            
            return self.author_id_map_dict
        except Exception as e:
            error_msg = f"Failed to process authors: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Author UUID Processing Failed",
                error_msg,
                error=True
            )
            raise

    def process_books(self, author_id_map_from_authors_step=None):
        try:
            books_df = pd.read_parquet(self.books_input_path)
            if author_id_map_from_authors_step:
                map_for_books_lookup = {str(k): v for k, v in author_id_map_from_authors_step.items()}
            else:
                map_for_books_lookup = self._load_author_id_map_from_csv()
            if not map_for_books_lookup:
                raise ValueError("Author ID map is not available. Run process_authors first or ensure map file exists and is loaded.")
            new_books_df = books_df.copy()
            def map_author_ids_in_list(author_list_of_dicts):
                if not isinstance(author_list_of_dicts, list):
                    return []
                updated_author_uuids = []
                for author_dict in author_list_of_dicts:
                    if isinstance(author_dict, dict) and 'author_id' in author_dict:
                        old_id_as_string = str(author_dict['author_id'])
                        if old_id_as_string in map_for_books_lookup:
                            updated_author_uuids.append(map_for_books_lookup[old_id_as_string])
                        else:
                            self.logger.warning(f"Old author_id '{old_id_as_string}' not found in map. Skipping this author.")
                return updated_author_uuids
            new_books_df['authors'] = new_books_df['authors'].apply(map_author_ids_in_list)
            new_books_df.to_parquet(self.new_books_output_path)
            self.logger.info(f"New books dataframe saved to {self.new_books_output_path}")
            
            # Send success notification
            self._send_notification(
                "Book UUID Processing Complete",
                f"Successfully updated author references in **{len(books_df):,}** books",
                fields=[
                    {"name": "Input File", "value": f"`{os.path.basename(self.books_input_path)}`", "inline": True},
                    {"name": "Output File", "value": f"`{os.path.basename(self.new_books_output_path)}`", "inline": True},
                    {"name": "Books Processed", "value": f"{len(books_df):,}", "inline": True},
                    {"name": "Author Mappings Used", "value": f"{len(map_for_books_lookup):,}", "inline": True}
                ]
            )
            
            return self.new_books_output_path
        except Exception as e:
            error_msg = f"Failed to process books: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Book UUID Processing Failed",
                error_msg,
                error=True
            )
            raise

    def run(self) -> Dict[str, Any]:
        self.logger.info(f"Starting UUID conversion step: {self.name}")
        
        # Send pipeline start notification
        self._send_notification(
            "UUID Conversion Started",
            f"Beginning UUID conversion pipeline: **{self.name}**",
            color=0x0099FF,  # Blue for start
            fields=[
                {"name": "Authors Input", "value": f"`{os.path.basename(self.authors_input_path)}`", "inline": True},
                {"name": "Books Input", "value": f"`{os.path.basename(self.books_input_path)}`", "inline": True}
            ]
        )
        
        try:
            outputs = {}
            author_id_map = self.process_authors()
            outputs["author_id_map_csv"] = self.author_map_csv_path
            outputs["new_authors_parquet"] = self.new_authors_output_path
            outputs["new_books_parquet"] = self.process_books(author_id_map_from_authors_step=author_id_map)
            outputs["author_id_map_dict"] = author_id_map
            self.logger.info(f"UUID conversion step {self.name} finished successfully.")
            
            # Send pipeline completion notification
            self._send_notification(
                "UUID Conversion Complete! 🎉",
                f"All UUID conversion tasks completed successfully for pipeline: **{self.name}**",
                color=0x00FF00,  # Green for success
                fields=[
                    {"name": "Authors", "value": "✅ Complete", "inline": True},
                    {"name": "Books", "value": "✅ Complete", "inline": True},
                    {"name": "UUID Mappings", "value": f"{len(author_id_map):,}", "inline": True},
                    {"name": "Authors Output", "value": f"`{os.path.basename(self.new_authors_output_path)}`", "inline": True},
                    {"name": "Books Output", "value": f"`{os.path.basename(self.new_books_output_path)}`", "inline": True},
                    {"name": "Map File", "value": f"`{os.path.basename(self.author_map_csv_path)}`", "inline": True}
                ]
            )
            
            self.output_data = outputs
            return outputs
            
        except Exception as e:
            error_msg = f"UUID conversion pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "UUID Conversion Pipeline Failed",
                error_msg,
                error=True
            )
            raise