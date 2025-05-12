import pandas as pd
import uuid
import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class UUIDProcessor:
    def __init__(self, authors_input_path, books_input_path,
                 new_authors_output_path, new_books_output_path,
                 author_map_csv_path):
        self.authors_input_path = authors_input_path
        self.books_input_path = books_input_path
        self.new_authors_output_path = new_authors_output_path
        self.new_books_output_path = new_books_output_path
        self.author_map_csv_path = author_map_csv_path
        
        # Ensure output directories exist
        for path in [new_authors_output_path, new_books_output_path, author_map_csv_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.author_id_map_dict = {} # Stores old_id -> new_uuid (strings)

    def _generate_author_id_map(self, authors_df):
        """Generates a map from old author_ids to new UUIDs."""
        # Keys will be of the same type as authors_df['author_id'].unique()
        # Values are string UUIDs.
        self.author_id_map_dict = {
            old_id: str(uuid.uuid4())
            for old_id in authors_df['author_id'].unique()
        }
        return self.author_id_map_dict

    def _save_author_id_map_to_csv(self):
        """Saves the author ID map to a CSV file."""
        if not self.author_id_map_dict:
            raise ValueError("Author ID map has not been generated yet.")
        
        # Convert map items to list of tuples for DataFrame creation
        map_items = list(self.author_id_map_dict.items())
        author_id_map_df = pd.DataFrame(map_items, columns=['old_id', 'new_id'])
        author_id_map_df.to_csv(self.author_map_csv_path, index=False)
        print(f"Author ID map saved to {self.author_map_csv_path}")

    def _load_author_id_map_from_csv(self):
        """Loads the author ID map from a CSV file into a dictionary with string keys."""
        try:
            author_id_map_df = pd.read_csv(self.author_map_csv_path)
            # Ensure old_id is string for consistent dictionary keys, as used in books processing
            return pd.Series(
                author_id_map_df.new_id.values, 
                index=author_id_map_df.old_id.astype(str)
            ).to_dict()
        except FileNotFoundError:
            print(f"Warning: Author ID map file not found at {self.author_map_csv_path}.")
            return {}

    def process_authors(self):
        """
        Reads authors data, generates UUIDs for author_id,
        updates the dataframe, and saves the new dataframe and the ID map.
        Returns the generated author ID map (old_id_type -> new_uuid_str).
        """
        authors_df = pd.read_parquet(self.authors_input_path)
        
        self._generate_author_id_map(authors_df) # Populates self.author_id_map_dict
        self._save_author_id_map_to_csv()

        new_authors_df = authors_df.copy()
        # .map() will use the keys from self.author_id_map_dict directly
        new_authors_df['author_id'] = new_authors_df['author_id'].map(self.author_id_map_dict)
        
        new_authors_df.to_parquet(self.new_authors_output_path)
        print(f"New authors dataframe saved to {self.new_authors_output_path}")
        return self.author_id_map_dict

    def process_books(self, author_id_map_from_authors_step=None):
        """
        Reads books data, updates author IDs in the 'authors' column using
        the provided or loaded ID map, and saves the new dataframe.
        The map used for lookup will have string keys for old IDs.
        """
        books_df = pd.read_parquet(self.books_input_path)
        
        map_for_books_lookup = {}
        if author_id_map_from_authors_step:
            # Convert keys to string for consistent lookup, as author['author_id'] is str-converted
            map_for_books_lookup = {str(k): v for k, v in author_id_map_from_authors_step.items()}
        else:
            map_for_books_lookup = self._load_author_id_map_from_csv() # Already has string keys

        if not map_for_books_lookup:
            raise ValueError(
                "Author ID map is not available. "
                "Run process_authors first or ensure map file exists and is loaded."
            )

        new_books_df = books_df.copy()

        def map_author_ids_in_list(author_list_of_dicts):
            if not isinstance(author_list_of_dicts, list):
                # Or log error, return empty list, or original data based on requirements
                return [] 
            
            updated_author_uuids = []
            for author_dict in author_list_of_dicts:
                if isinstance(author_dict, dict) and 'author_id' in author_dict:
                    old_id_as_string = str(author_dict['author_id'])
                    if old_id_as_string in map_for_books_lookup:
                        updated_author_uuids.append(map_for_books_lookup[old_id_as_string])
                    else:
                        # Handle missing IDs: log, skip, raise error, or add placeholder
                        print(f"Warning: Old author_id '{old_id_as_string}' not found in map. Skipping this author.")
                # else: could log malformed author entry
            return updated_author_uuids

        new_books_df['authors'] = new_books_df['authors'].apply(map_author_ids_in_list)
        
        new_books_df.to_parquet(self.new_books_output_path)
        print(f"New books dataframe saved to {self.new_books_output_path}")

    def run_pipeline(self):
        """Runs the full data processing pipeline."""
        print("Starting UUID conversion pipeline...")
        # Process authors and get the map (keys are original type)
        generated_author_map = self.process_authors()
        # Process books, passing the in-memory map
        self.process_books(author_id_map_from_authors_step=generated_author_map)
        print("UUID conversion pipeline finished.")

# Example of how to use the class:
if __name__ == "__main__":
    # Define base path for data relative to this script's location
    # Assumes script is in /Users/yamirghofran0/bookdbio/scripts/
    # and data is in /Users/yamirghofran0/bookdbio/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(script_dir, "..", "data")
    
    # Ensure consistent output paths, e.g., all outputs go into the base_data_dir
    # Original script saved some outputs to "data/" which could be ambiguous.
    # Here, all paths are explicitly relative to `base_data_dir`.

    processor = UUIDProcessor(
        authors_input_path=os.path.join(base_data_dir, "authors_df.parquet"),
        books_input_path=os.path.join(base_data_dir, "books_df.parquet"),
        new_authors_output_path=os.path.join(base_data_dir, "new_authors.parquet"),
        new_books_output_path=os.path.join(base_data_dir, "new_books.parquet"),
        author_map_csv_path=os.path.join(base_data_dir, "author_id_map.csv")
    )
    
    processor.run_pipeline()