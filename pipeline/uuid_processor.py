import pandas as pd
import uuid
import os
from typing import Dict, Any
from .core import PipelineStep

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
        authors_df = pd.read_parquet(self.authors_input_path)
        self._generate_author_id_map(authors_df)
        self._save_author_id_map_to_csv()
        new_authors_df = authors_df.copy()
        new_authors_df['author_id'] = new_authors_df['author_id'].map(self.author_id_map_dict)
        new_authors_df.to_parquet(self.new_authors_output_path)
        self.logger.info(f"New authors dataframe saved to {self.new_authors_output_path}")
        return self.author_id_map_dict

    def process_books(self, author_id_map_from_authors_step=None):
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
        return self.new_books_output_path

    def run(self) -> Dict[str, Any]:
        self.logger.info(f"Starting UUID conversion step: {self.name}")
        outputs = {}
        author_id_map = self.process_authors()
        outputs["author_id_map_csv"] = self.author_map_csv_path
        outputs["new_authors_parquet"] = self.new_authors_output_path
        outputs["new_books_parquet"] = self.process_books(author_id_map_from_authors_step=author_id_map)
        outputs["author_id_map_dict"] = author_id_map
        self.logger.info(f"UUID conversion step {self.name} finished successfully.")
        self.output_data = outputs
        return outputs