import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import TripletEvaluator
import torch
from torch.utils.data import DataLoader
import math
import os
import sys
import warnings
import logging
import json
import random
from datasets import load_dataset
from typing import Dict, List, Tuple, Any, Optional

# --- Configuration ---
RANDOM_STATE = 42
MODEL_NAME = 'all-MiniLM-L6-v2'
BOOKS_DATA_FILE = "../data/reduced_books.parquet"
AUTHORS_DATA_FILE = "../data/new_authors.parquet"
BOOK_TEXTS_FILE = "../data/book_texts.parquet"
TRIPLETS_DATA_FILE = "../data/books_triplets.parquet"
OUTPUT_BASE_PATH = f'sbert-output/finetuning-{MODEL_NAME}-books'
EVAL_OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, 'eval')
CHECKPOINT_PATH = os.path.join(OUTPUT_BASE_PATH, 'checkpoints')

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
TRIPLET_MARGIN = 0.5
WARMUP_STEPS_RATIO = 0.1
EVALUATION_STEPS = 500
SAVE_STEPS = 1000
CHECKPOINT_LIMIT = 2
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.5 # Relative to the test_split size (0.5 * 0.2 = 0.1 of total)
MAX_NEGATIVE_SEARCH_ATTEMPTS = 100

# Setup
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
os.makedirs(EVAL_OUTPUT_PATH, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def get_device() -> str:
    """Determines the best available device (MPS, CUDA, CPU)."""
    if torch.backends.mps.is_available():
        logging.info("Using Apple Metal Performance Shaders (MPS).")
        return "mps"
    elif torch.cuda.is_available():
        logging.info("Using CUDA.")
        return "cuda"
    else:
        logging.warning("Training on CPU will be very slow.")
        return "cpu"

# --- Classes ---

class DataLoaderAnalyzer:
    """Loads and analyzes the initial datasets."""
    def __init__(self, books_path: str, authors_path: str):
        self.books_path = books_path
        self.authors_path = authors_path
        self.books_df_dd: Optional[dd.DataFrame] = None
        self.authors_df_pd: Optional[pd.DataFrame] = None
        self.books_df_pd: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Loads the books (Dask) and authors (Pandas) data."""
        logging.info(f"Loading books data from {self.books_path}")
        self.books_df_dd = dd.read_parquet(self.books_path)
        logging.info(f"Loading authors data from {self.authors_path}")
        # Assuming authors data is small enough for Pandas
        self.authors_df_pd = pd.read_parquet(self.authors_path)
        logging.info("Data loading complete.")

    def analyze_dataframe(self, df: dd.DataFrame) -> pd.DataFrame:
        """Analyzes a Dask DataFrame for column info and nulls."""
        logging.info("Analyzing DataFrame...")
        cols = df.columns
        dtypes = df.dtypes
        total_rows = len(df) # Dask len is efficient

        results = []
        for col in cols:
            non_null_count = df[col].count().compute()
            null_count = total_rows - non_null_count
            null_percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0
            results.append({
                'Column': col,
                'Data Type': str(dtypes[col]),
                'Non-Null Count': non_null_count,
                'Null Count': null_count,
                'Null Percentage': f'{null_percentage:.2f}%'
            })
        results_df = pd.DataFrame(results)
        logging.info("DataFrame analysis complete.")
        return results_df.sort_values('Null Percentage', ascending=False)

    def run_analysis(self) -> None:
        """Loads data and prints the analysis."""
        if self.books_df_dd is None:
            self.load_data()
        analysis_results = self.analyze_dataframe(self.books_df_dd)
        print("\n--- DataFrame Analysis ---")
        print(analysis_results)
        print("-------------------------\n")

    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Computes Dask DataFrame to Pandas and returns both."""
        if self.books_df_dd is None or self.authors_df_pd is None:
            self.load_data()

        if self.books_df_pd is None:
             logging.info("Converting books Dask DataFrame to Pandas...")
             self.books_df_pd = self.books_df_dd.compute()
             if 'book_id' in self.books_df_pd.columns:
                 self.books_df_pd.set_index('book_id', inplace=True)
             logging.info("Conversion complete.")

        return self.books_df_pd, self.authors_df_pd


class TextPreprocessor:
    """Creates combined text representations for books."""
    def __init__(self, books_df: pd.DataFrame, authors_df: pd.DataFrame, output_path: str):
        self.books_df = books_df
        self.authors_df = authors_df.set_index('author_id') # Faster lookup
        self.output_path = output_path
        self.book_texts_df: Optional[pd.DataFrame] = None
        # Define genre and ignore keywords here or load from config
        self.genre_keywords = sorted([
            'action', 'adventure', 'comedy', 'crime', 'mystery', 'textbook', 'children', 'mathematics', 'fantasy',
            'historical', 'horror', 'romance', 'satire', 'science fiction',
            'scifi', 'speculative fiction', 'thriller', 'western', 'paranormal',
            'dystopian', 'urban fantasy', 'contemporary', 'young adult', 'ya',
            'middle grade', 'children\'s', 'literary fiction', 'magic realism',
            'historical fiction', 'gothic', 'suspense', 'biography', 'memoir',
            'nonfiction', 'poetry', 'drama', 'historical romance',
            'fantasy romance', 'romantic suspense', 'science fiction romance',
            'contemporary romance', 'paranormal romance', 'epic fantasy',
            'dark fantasy', 'sword and sorcery', 'steampunk', 'cyberpunk',
            'apocalyptic', 'post-apocalyptic', 'alternate history',
            'superhero', 'mythology', 'fairy tales', 'folklore', 'war',
            'military fiction', 'spy fiction', 'political fiction', 'social science fiction',
            'techno-thriller', 'medical thriller', 'legal thriller',
            'psychological thriller', 'cozy mystery', 'hardboiled', 'noir',
            'coming-of-age', 'lgbtq+', 'christian fiction', 'religious fiction',
            'humor', 'travel', 'food', 'cooking', 'health', 'self-help',
            'business', 'finance', 'history', 'science', 'technology', 'nature',
            'art', 'music', 'philosophy', 'education', 'true crime', 'spiritual',
            'anthology', 'short stories', 'plays', 'screenplays', 'graphic novel',
            'comics', 'manga', 'erotica', 'new adult', 'chick lit', 'womens fiction',
            'sports fiction', 'family saga', ' Regency romance', 'literature'
        ], key=len, reverse=True)
        self.ignore_keywords = ['to-read', 'owned', 'hardcover', 'shelfari-favorites', 'series', 'might-read',
                           'dnf-d', 'hambly-barbara', 'strong-females', 'first-in-series',
                           'no-thanks-series-collections-boxes', 'entertaining-but-limited',
                           'kate-own', 'e-book', 'compliation', 'my-books',
                           'books-i-own-but-have-not-read', 'everything-owned', 'books-to-find',
                           'i-own-it', 'favorite', 'not-read', 'read-some-day', 'library',
                           'audiobooks', 'status-borrowed', 'owned-books',
                           'spec-fic-awd-locus-nom', '01', 'hardbacks', 'paper', 'german',
                           'hardback', 'physical-scifi-fantasy', 'childhood-favorites',
                           'bundle-same-author', 'aa-sifi-fantasy', 'ready-to-read',
                           'bought-on-flee-markets', 'fantasy-general', 'hardcopy', 'box-2',
                           'unfinished', 'magic', 'duplicates', 'favorites', 'books-i-own',
                           'fantasy-classic', 'own-hard-copy', 'fantasy-read',
                           'book-club-edition', 'sci-fi-or-fantasy', 'fiction-fantasy',
                           'fiction-literature-poetry', 'paused-hiatus', 'status—borrowed',
                           'recs-fantasy', 'fantasy-scifi', 'omnibus', 'speculative',
                           'sf--fantasy', 'in-my-home-library', 'fant-myth-para-vamps',
                           'read-in-my-20s']

    def _extract_genres(self, popular_shelves: Any) -> List[str]:
        """Extracts potential genres from popular shelves."""
        try:
            if not isinstance(popular_shelves, np.ndarray) or len(popular_shelves) == 0:
                return []
            found_genres = set()
            for shelf in popular_shelves:
                if not isinstance(shelf, dict) or 'name' not in shelf: continue
                shelf_name = shelf['name'].lower().strip()
                if any(ignore in shelf_name for ignore in self.ignore_keywords): continue
                for keyword in self.genre_keywords:
                    if keyword in shelf_name:
                        found_genres.add(keyword)
            return sorted(list(found_genres))
        except Exception as e:
            logging.error(f"Error in _extract_genres: {e}", exc_info=True)
            return []

    def create_book_texts(self) -> None:
        """Generates the combined text for each book."""
        logging.info("Creating book text representations...")
        book_texts = []
        for index, row in self.books_df.iterrows():
            title = row.get('title', '')
            description = row.get('description', '')
            if not title or not description: continue

            genres = self._extract_genres(row.get('popular_shelves', []))
            author_ids = row.get('authors', [])
            authors = []
            for author_id in author_ids:
                try:
                    author_name = self.authors_df.loc[author_id, 'name']
                    authors.append(author_name)
                except KeyError:
                    # logging.warning(f"Author ID {author_id} not found for book {index}.")
                    pass # Optionally log missing authors

            book_text = f"Title: {title} | Genres: {', '.join(genres)} | Description: {description} | Authors: {', '.join(authors)}"
            book_texts.append({'book_id': index, 'text': book_text})

        self.book_texts_df = pd.DataFrame(book_texts)
        logging.info(f"Created {len(self.book_texts_df)} book text entries.")

    def save_book_texts(self) -> None:
        """Saves the generated book texts to a Parquet file."""
        if self.book_texts_df is None:
            self.create_book_texts()

        if not self.book_texts_df.empty:
            logging.info(f"Saving book texts to {self.output_path}")
            self.book_texts_df.to_parquet(self.output_path, index=False)
            logging.info("Book texts saved successfully.")
        else:
            logging.warning("No book texts generated, skipping save.")

    def get_book_texts_df(self) -> pd.DataFrame:
        """Returns the book texts DataFrame."""
        if self.book_texts_df is None:
             # Try loading if not generated in this run
            if os.path.exists(self.output_path):
                logging.info(f"Loading existing book texts from {self.output_path}")
                self.book_texts_df = pd.read_parquet(self.output_path)
            else:
                self.create_book_texts()
                self.save_book_texts()
        return self.book_texts_df


class TripletGenerator:
    """Generates (anchor, positive, negative) triplets for training."""
    def __init__(self, books_df: pd.DataFrame, book_texts_df: pd.DataFrame, output_path: str, max_neg_attempts: int):
        self.books_df = books_df # Assumes index is 'book_id'
        self.book_texts_map = book_texts_df.set_index('book_id')['text'].to_dict()
        self.output_path = output_path
        self.max_neg_attempts = max_neg_attempts
        self.triplets_df: Optional[pd.DataFrame] = None
        self.all_book_ids_with_text = list(self.book_texts_map.keys())

    def generate_triplets(self) -> None:
        """Creates the triplet data."""
        logging.info(f"Starting triplet generation for {len(self.book_texts_map)} books...")
        triplet_data = []
        processed_anchors = 0

        for anchor_id, anchor_text in self.book_texts_map.items():
            try:
                anchor_info = self.books_df.loc[anchor_id]
                similar_books = anchor_info.get('similar_books', [])
                if not similar_books: continue

                forbidden_ids = {anchor_id} | set(similar_books)

                for positive_id_str in similar_books:
                    try:
                        positive_id = int(positive_id_str) # Ensure ID is integer for lookup
                    except (ValueError, TypeError):
                        # logging.warning(f"Invalid similar_book ID '{positive_id_str}' for anchor {anchor_id}. Skipping.")
                        continue

                    positive_text = self.book_texts_map.get(positive_id)
                    if positive_text is None:
                        # logging.warning(f"Text not found for positive book ID {positive_id} (anchor {anchor_id}). Skipping.")
                        continue

                    negative_id = None
                    negative_text = None
                    for _ in range(self.max_neg_attempts):
                        potential_neg_id = random.choice(self.all_book_ids_with_text)
                        if potential_neg_id not in forbidden_ids:
                            potential_neg_text = self.book_texts_map.get(potential_neg_id)
                            if potential_neg_text is not None:
                                negative_id = potential_neg_id
                                negative_text = potential_neg_text
                                break # Found a negative

                    if negative_id and negative_text:
                        triplet_data.append({
                            'anchor': anchor_text,
                            'positive': positive_text,
                            'negative': negative_text
                        })
                    # else:
                        # logging.warning(f"Could not find suitable negative for anchor {anchor_id}, positive {positive_id}")

            except KeyError:
                logging.warning(f"Anchor book ID {anchor_id} not found in books_df. Skipping.")
                continue
            except Exception as e:
                 logging.error(f"Error processing anchor {anchor_id}: {e}", exc_info=True)
                 continue

            processed_anchors += 1
            if processed_anchors % 1000 == 0:
                 logging.info(f"Processed {processed_anchors} anchors...")


        self.triplets_df = pd.DataFrame(triplet_data)
        logging.info(f"Generated {len(self.triplets_df)} triplets.")

    def save_triplets(self) -> None:
        """Saves the generated triplets to a Parquet file."""
        if self.triplets_df is None:
            self.generate_triplets()

        if not self.triplets_df.empty:
            logging.info(f"Saving triplets to {self.output_path}")
            self.triplets_df.to_parquet(self.output_path, index=False)
            logging.info("Triplets saved successfully.")
            print("\n--- Sample Triplets ---")
            print(self.triplets_df.head())
            print("-----------------------\n")
        else:
            logging.warning("No triplets were generated, skipping save.")

    def get_triplets_df(self) -> pd.DataFrame:
        """Returns the triplets DataFrame."""
        if self.triplets_df is None:
             # Try loading if not generated in this run
            if os.path.exists(self.output_path):
                logging.info(f"Loading existing triplets from {self.output_path}")
                self.triplets_df = pd.read_parquet(self.output_path)
            else:
                self.generate_triplets()
                self.save_triplets()
        return self.triplets_df


class DatasetSplitter:
    """Splits the dataset into train, validation, and test sets."""
    def __init__(self, data_path: str, test_size: float, val_size: float, seed: int):
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size # Relative to the test split
        self.seed = seed
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def split_data(self) -> None:
        """Loads and splits the data using Hugging Face datasets."""
        logging.info(f"Loading dataset from {self.data_path}")
        full_dataset = load_dataset('parquet', data_files=self.data_path, split='train')

        logging.info(f"Splitting data: test_size={self.test_size}, validation_size={self.val_size} (relative)")
        train_testvalid_split = full_dataset.train_test_split(test_size=self.test_size, seed=self.seed)
        self.train_dataset = train_testvalid_split['train']
        test_valid_dataset = train_testvalid_split['test']

        test_validation_split = test_valid_dataset.train_test_split(test_size=self.val_size, seed=self.seed)
        self.validation_dataset = test_validation_split['train']
        self.test_dataset = test_validation_split['test']

        logging.info(f"Split complete: Train={len(self.train_dataset)}, Validation={len(self.validation_dataset)}, Test={len(self.test_dataset)}")

    def get_datasets(self) -> Tuple[Any, Any, Any]:
        """Returns the split datasets."""
        if not all([self.train_dataset, self.validation_dataset, self.test_dataset]):
            self.split_data()
        return self.train_dataset, self.validation_dataset, self.test_dataset

    def get_split_texts(self, dataset_type: str = 'validation') -> Tuple[List[str], List[str], List[str]]:
        """Extracts anchor, positive, negative texts from a specified dataset split."""
        if not all([self.train_dataset, self.validation_dataset, self.test_dataset]):
            self.split_data()

        dataset = None
        if dataset_type == 'validation':
            dataset = self.validation_dataset
        elif dataset_type == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError("dataset_type must be 'validation' or 'test'")

        if dataset is None:
             raise RuntimeError("Dataset not loaded or split correctly.")

        logging.info(f"Extracting texts for {dataset_type} set...")
        anchors = dataset['anchor']
        positives = dataset['positive']
        negatives = dataset['negative']
        logging.info(f"Extracted {len(anchors)} {dataset_type} triplets.")
        # Optional: Print samples
        # print(f"\n--- Sample {dataset_type.capitalize()} Triplets ---")
        # print(f"{dataset_type.capitalize()} Anchors:", anchors[:3])
        # print(f"{dataset_type.capitalize()} Positives:", positives[:3])
        # print(f"{dataset_type.capitalize()} Negatives:", negatives[:3])
        # print("-----------------------------------\n")
        return anchors, positives, negatives


class ModelTrainerEvaluator:
    """Handles model loading, training, and evaluation."""
    def __init__(self, model_name: str, output_path: str, eval_output_path: str, checkpoint_path: str, device: str, config: Dict):
        self.model_name = model_name
        self.output_path = output_path
        self.eval_output_path = eval_output_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.config = config # Store hyperparameters
        self.baseline_model = SentenceTransformer(model_name, device=device)
        self.finetuned_model: Optional[SentenceTransformer] = None
        self.baseline_accuracy: Optional[float] = None
        self.finetuned_accuracy: Optional[float] = None

    def _create_evaluator(self, anchors: List[str], positives: List[str], negatives: List[str], name: str, write_csv: bool = False) -> TripletEvaluator:
        """Helper to create a TripletEvaluator."""
        return TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            main_similarity_function='cosine', # Or other similarity
            margin=self.config['triplet_margin'],
            name=name,
            show_progress_bar=True,
            write_csv=write_csv,
            batch_size=self.config['batch_size'] * 2 # Often can use larger batch for eval
        )

    def evaluate_baseline(self, test_anchors: List[str], test_positives: List[str], test_negatives: List[str]) -> float:
        """Evaluates the baseline model performance on the test set."""
        logging.info("Evaluating baseline model...")
        baseline_evaluator = self._create_evaluator(test_anchors, test_positives, test_negatives, name='baseline-test')
        results = baseline_evaluator(self.baseline_model, output_path=self.eval_output_path)
        # The key might be 'accuracy_cosine' or similar depending on TripletEvaluator version/config
        primary_metric = baseline_evaluator.primary_metric
        self.baseline_accuracy = results.get(primary_metric, results.get('accuracy_cosine')) # Try common keys
        if self.baseline_accuracy is None:
             logging.error(f"Could not find primary metric '{primary_metric}' or 'accuracy_cosine' in baseline results: {results}")
             raise KeyError("Baseline accuracy metric not found in evaluation results.")

        logging.info(f"Baseline Test Accuracy ({primary_metric}): {self.baseline_accuracy:.4f}")
        return self.baseline_accuracy

    def train(self, train_dataset: Any, val_anchors: List[str], val_positives: List[str], val_negatives: List[str]) -> None:
        """Fine-tunes the model."""
        logging.info("Preparing for fine-tuning...")

        # Define Loss Function
        train_loss = losses.TripletLoss(
            model=self.baseline_model, # Start fine-tuning from the baseline
            distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
            triplet_margin=self.config['triplet_margin'],
        )

        # Create Data Loader
        train_examples = [InputExample(texts=[ex['anchor'], ex['positive'], ex['negative']]) for ex in train_dataset]
        logging.info(f"Created {len(train_examples)} training examples.")
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.config['batch_size'])

        # Define Validation Evaluator
        validation_evaluator = self._create_evaluator(val_anchors, val_positives, val_negatives, name='validation', write_csv=True)

        # Calculate Warmup Steps
        num_training_steps = len(train_dataloader) * self.config['epochs']
        warmup_steps = int(num_training_steps * self.config['warmup_steps_ratio'])
        logging.info(f"Total training steps: {num_training_steps}, Warmup steps: {warmup_steps}")

        # Fitting the Model
        logging.info("--- Starting Fine-tuning ---")
        self.baseline_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=validation_evaluator,
            epochs=self.config['epochs'],
            evaluation_steps=self.config['evaluation_steps'],
            warmup_steps=warmup_steps,
            output_path=self.output_path, # Saves the best model here
            save_best_model=True,
            optimizer_params={'lr': self.config['learning_rate']},
            checkpoint_path=self.checkpoint_path,
            checkpoint_save_steps=self.config['save_steps'],
            checkpoint_save_total_limit=self.config['checkpoint_limit'],
            output_path_ignore_not_empty=True # Allow overwriting if needed
        )
        logging.info("--- Fine-tuning Finished ---")

        # Load the best model saved by fit
        logging.info(f"Loading best fine-tuned model from: {self.output_path}")
        self.finetuned_model = SentenceTransformer(self.output_path, device=self.device)


    def evaluate_finetuned(self, test_anchors: List[str], test_positives: List[str], test_negatives: List[str]) -> float:
        """Evaluates the fine-tuned model performance on the test set."""
        if self.finetuned_model is None:
            # Try loading if not trained in this run
            if os.path.exists(self.output_path):
                 logging.info(f"Loading existing fine-tuned model from: {self.output_path}")
                 self.finetuned_model = SentenceTransformer(self.output_path, device=self.device)
            else:
                raise RuntimeError("Fine-tuned model not available. Train the model first or ensure the output path exists.")

        logging.info("Evaluating fine-tuned model on test set...")
        final_test_evaluator = self._create_evaluator(test_anchors, test_positives, test_negatives, name='finetuned-test')
        results = final_test_evaluator(self.finetuned_model, output_path=self.eval_output_path)

        primary_metric = final_test_evaluator.primary_metric
        self.finetuned_accuracy = results.get(primary_metric, results.get('accuracy_cosine'))
        if self.finetuned_accuracy is None:
             logging.error(f"Could not find primary metric '{primary_metric}' or 'accuracy_cosine' in fine-tuned results: {results}")
             raise KeyError("Fine-tuned accuracy metric not found in evaluation results.")

        logging.info(f"Fine-tuned Test Accuracy ({primary_metric}): {self.finetuned_accuracy:.4f}")
        return self.finetuned_accuracy

    def get_accuracies(self) -> Tuple[Optional[float], Optional[float]]:
        """Returns the baseline and fine-tuned accuracies."""
        return self.baseline_accuracy, self.finetuned_accuracy


class ResultVisualizer:
    """Plots the training validation performance and final comparison."""
    def __init__(self, eval_output_path: str, output_base_path: str):
        self.eval_output_path = eval_output_path
        self.output_base_path = output_base_path

    def plot_validation_accuracy(self) -> None:
        """Plots the validation accuracy recorded during training."""
        logging.info("--- Plotting Validation Performance ---")
        eval_filepath = os.path.join(self.eval_output_path, "triplet_evaluation_validation_results.csv")
        plot_save_path = os.path.join(self.output_base_path, 'validation_accuracy_plot.png')

        try:
            eval_results = pd.read_csv(eval_filepath)
            # Adjust column name based on actual output (might be accuracy_cosine, accuracy_dot, etc.)
            accuracy_col = None
            if 'accuracy_cosine' in eval_results.columns:
                accuracy_col = 'accuracy_cosine'
            elif 'accuracy_dot' in eval_results.columns: # Example alternative
                 accuracy_col = 'accuracy_dot'
            # Add more potential column names if needed

            if 'steps' in eval_results.columns and accuracy_col:
                plt.figure(figsize=(10, 5))
                plt.plot(eval_results['steps'], eval_results[accuracy_col], marker='o', linestyle='-')
                plt.title('Validation Accuracy during Fine-tuning')
                plt.xlabel('Training Steps')
                plt.ylabel(f'{accuracy_col.replace("_", " ").title()}')
                plt.grid(True)
                plt.savefig(plot_save_path)
                logging.info(f"Validation plot saved to: {plot_save_path}")
                # plt.show() # Optionally display plot immediately
                plt.close() # Close plot to free memory
            else:
                logging.warning(f"Columns 'steps' or a suitable accuracy column not found in {eval_filepath}. Cannot plot validation accuracy.")

        except FileNotFoundError:
            logging.warning(f"Evaluation results file not found at: {eval_filepath}. Plotting skipped.")
        except Exception as e:
            logging.error(f"An error occurred during validation plotting: {e}", exc_info=True)

    def plot_comparison(self, baseline_acc: float, finetuned_acc: float) -> None:
        """Plots the comparison bar chart of baseline vs fine-tuned accuracy."""
        logging.info("--- Plotting Results Comparison ---")
        plot_save_path = os.path.join(self.output_base_path, 'comparison_accuracy_plot.png')

        improvement = finetuned_acc - baseline_acc
        improvement_percent = (improvement / baseline_acc * 100) if baseline_acc != 0 else 0

        print("\n--- Results Comparison ---")
        print(f"Baseline Test Accuracy:   {baseline_acc:.4f}")
        print(f"Fine-tuned Test Accuracy: {finetuned_acc:.4f}")
        print(f"Improvement:              {improvement:.4f} ({improvement_percent:.1f}%)")
        print("--------------------------\n")

        labels = ['Baseline', 'Fine-tuned']
        accuracies = [baseline_acc, finetuned_acc]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, accuracies, color=['skyblue', 'lightgreen'])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

        plt.ylabel('Accuracy (Cosine)') # Assuming cosine, adjust if needed
        plt.title('Baseline vs. Fine-tuned Model Accuracy')
        plt.ylim(0, max(accuracies) * 1.15) # Adjust y-axis limit
        plt.savefig(plot_save_path)
        logging.info(f"Comparison plot saved to: {plot_save_path}")
        # plt.show() # Optionally display plot immediately
        plt.close() # Close plot


# --- Pipeline Orchestrator ---

class FineTuningPipeline:
    """Orchestrates the fine-tuning process."""
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device()

        # Initialize components
        self.loader_analyzer = DataLoaderAnalyzer(
            books_path=config['books_data_file'],
            authors_path=config['authors_data_file']
        )
        # TextPreprocessor needs loaded data, initialized later
        # TripletGenerator needs preprocessed data, initialized later
        self.dataset_splitter = DatasetSplitter(
            data_path=config['triplets_data_file'],
            test_size=config['test_split_size'],
            val_size=config['validation_split_size'],
            seed=config['random_state']
        )
        self.trainer_evaluator = ModelTrainerEvaluator(
            model_name=config['model_name'],
            output_path=config['output_base_path'],
            eval_output_path=config['eval_output_path'],
            checkpoint_path=config['checkpoint_path'],
            device=self.device,
            config={ # Pass relevant hyperparameters
                'batch_size': config['batch_size'],
                'epochs': config['epochs'],
                'learning_rate': config['learning_rate'],
                'triplet_margin': config['triplet_margin'],
                'warmup_steps_ratio': config['warmup_steps_ratio'],
                'evaluation_steps': config['evaluation_steps'],
                'save_steps': config['save_steps'],
                'checkpoint_limit': config['checkpoint_limit']
            }
        )
        self.visualizer = ResultVisualizer(
             eval_output_path=config['eval_output_path'],
             output_base_path=config['output_base_path']
        )

    def run(self):
        """Executes the pipeline steps."""
        logging.info("Starting Fine-Tuning Pipeline...")

        # 1. Load and Analyze Data
        self.loader_analyzer.run_analysis()
        books_df, authors_df = self.loader_analyzer.get_dataframes()

        # 2. Preprocess Text Data
        # Initialize now that data is loaded
        text_preprocessor = TextPreprocessor(
            books_df=books_df,
            authors_df=authors_df,
            output_path=self.config['book_texts_file']
        )
        book_texts_df = text_preprocessor.get_book_texts_df() # Generates/saves if needed

        # 3. Generate Triplets
        # Initialize now that text data is ready
        triplet_generator = TripletGenerator(
            books_df=books_df,
            book_texts_df=book_texts_df,
            output_path=self.config['triplets_data_file'],
            max_neg_attempts=self.config['max_negative_search_attempts']
        )
        triplet_generator.get_triplets_df() # Generates/saves if needed

        # 4. Split Data
        train_ds, val_ds, test_ds = self.dataset_splitter.get_datasets()
        val_anchors, val_positives, val_negatives = self.dataset_splitter.get_split_texts('validation')
        test_anchors, test_positives, test_negatives = self.dataset_splitter.get_split_texts('test')

        # 5. Evaluate Baseline Model
        baseline_acc = self.trainer_evaluator.evaluate_baseline(test_anchors, test_positives, test_negatives)

        # 6. Train Model
        self.trainer_evaluator.train(train_ds, val_anchors, val_positives, val_negatives)

        # 7. Evaluate Fine-tuned Model
        finetuned_acc = self.trainer_evaluator.evaluate_finetuned(test_anchors, test_positives, test_negatives)

        # 8. Visualize Results
        self.visualizer.plot_validation_accuracy()
        self.visualizer.plot_comparison(baseline_acc, finetuned_acc)

        logging.info("Fine-Tuning Pipeline Finished Successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    pipeline_config = {
        'random_state': RANDOM_STATE,
        'model_name': MODEL_NAME,
        'books_data_file': BOOKS_DATA_FILE,
        'authors_data_file': AUTHORS_DATA_FILE,
        'book_texts_file': BOOK_TEXTS_FILE,
        'triplets_data_file': TRIPLETS_DATA_FILE,
        'output_base_path': OUTPUT_BASE_PATH,
        'eval_output_path': EVAL_OUTPUT_PATH,
        'checkpoint_path': CHECKPOINT_PATH,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'triplet_margin': TRIPLET_MARGIN,
        'warmup_steps_ratio': WARMUP_STEPS_RATIO,
        'evaluation_steps': EVALUATION_STEPS,
        'save_steps': SAVE_STEPS,
        'checkpoint_limit': CHECKPOINT_LIMIT,
        'test_split_size': TEST_SPLIT_SIZE,
        'validation_split_size': VALIDATION_SPLIT_SIZE,
        'max_negative_search_attempts': MAX_NEGATIVE_SEARCH_ATTEMPTS
    }

    pipeline = FineTuningPipeline(pipeline_config)
    pipeline.run()