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
import datetime
from datasets import load_dataset
from typing import Dict, List, Tuple, Any, Optional
from .core import PipelineStep
from utils import get_device
from utils import send_discord_webhook


class SbertTrainerStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        # Default configuration - will be overridden by configure()
        self.random_state = 42
        self.model_name = 'all-MiniLM-L6-v2'
        self.books_data_file = "data/reduced_books.parquet"
        self.authors_data_file = "data/new_authors.parquet"
        self.book_texts_file = "data/book_texts.parquet"
        self.triplets_data_file = "data/books_triplets.parquet"
        self.output_base_path = "results/sbert"
        self.eval_output_path = os.path.join(self.output_base_path, 'eval')
        self.checkpoint_path = os.path.join(self.output_base_path, 'checkpoints')
        self.batch_size = 32
        self.epochs = 4
        self.learning_rate = 2e-5
        self.triplet_margin = 0.5
        self.warmup_steps_ratio = 0.1
        self.evaluation_steps = 500
        self.save_steps = 1000
        self.checkpoint_limit = 2
        self.test_split_size = 0.2
        self.validation_split_size = 0.5
        self.max_negative_search_attempts = 100
        self.device = get_device()
        self._set_seed()

    def _set_seed(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        random.seed(self.random_state)

    def _send_notification(self, title: str, description: str, color: int = 0x00FF00, fields: list = None, error: bool = False):
        """Send a Discord notification with consistent formatting."""
        try:
            embed = {
                "title": f"🧠 {title}" if not error else f"❌ {title}",
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

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        # Update configuration from config dict
        for key, value in self.config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Ensure directories exist
        os.makedirs(self.output_base_path, exist_ok=True)
        os.makedirs(self.eval_output_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.book_texts_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.triplets_data_file), exist_ok=True)
        
        self.device = get_device()
        self._set_seed()

    def analyze_dataframe(self, df: dd.DataFrame) -> pd.DataFrame:
        """Analyzes a Dask DataFrame for column info and nulls - matches finetune_sbert.py logic."""
        self.logger.info("Analyzing DataFrame...")
        cols = df.columns
        dtypes = df.dtypes
        total_rows = len(df.compute())
        
        results = []
        for col in cols:
            non_null_count = df[col].count().compute()
            null_count = total_rows - non_null_count
            null_percentage = (null_count / total_rows) * 100
            results.append({
                'Column': col,
                'Data Type': str(dtypes[col]),
                'Non-Null Count': non_null_count,
                'Null Count': null_count,
                'Null Percentage': f'{null_percentage:.2f}%'
            })
        
        results_df = pd.DataFrame(results)
        return results_df.sort_values('Null Percentage', ascending=False)

    def extract_genres(self, popular_shelves: Any) -> List[str]:
        """Extracts potential genres from popular shelves - exact logic from finetune_sbert.py."""
        try:
            if not isinstance(popular_shelves, np.ndarray) or len(popular_shelves) == 0:
                return []
            
            found_genres = set()
            
            genre_keywords = [
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
            ]
            # Sort keywords by length descending to match longer phrases first
            genre_keywords.sort(key=len, reverse=True)

            ignore_keywords = ['to-read', 'owned', 'hardcover', 'shelfari-favorites', 'series', 'might-read',
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

            for shelf in popular_shelves:
                if not isinstance(shelf, dict) or 'name' not in shelf:
                    continue
                
                shelf_name = shelf['name'].lower().strip()

                # Skip if shelf name contains any ignore keywords
                if any(ignore in shelf_name for ignore in ignore_keywords):
                    continue

                # Check if any genre keyword is present in the shelf name
                for keyword in genre_keywords:
                    if keyword in shelf_name: 
                        found_genres.add(keyword)

            return sorted(list(found_genres))
        except Exception as e:
            self.logger.error(f"Error in extract_genres function: {e}", exc_info=True)
            return []

    def create_book_texts(self, books_df: pd.DataFrame, authors_df: pd.DataFrame) -> pd.DataFrame:
        """Creates book text representations - exact logic from finetune_sbert.py."""
        try:
            self.logger.info("Creating book text representations...")
            
            book_texts = []
            total_books = len(books_df)
            valid_books = 0
            
            for index, row in books_df.iterrows():
                if row['description'] == '' or row['title'] == '':
                    continue
                if row['description'] is None or row['title'] is None:
                    continue
                    
                genres = self.extract_genres(row['popular_shelves'])
                authors = []
                for author_id in row['authors']:
                    author = authors_df.loc[authors_df['author_id'] == author_id]
                    if not author.empty:
                        authors.append(author.iloc[0]['name'])
                
                book_text = f"Title: {row['title']} | Genres: {', '.join(genres)} | Description: {row['description']} | Authors: {', '.join(authors)}"
                book_texts.append({'book_id': index, 'text': book_text})
                valid_books += 1

            book_texts_df = pd.DataFrame(book_texts)
            self.logger.info(f"Created {len(book_texts_df)} book text entries.")
            
            # Send success notification
            self._send_notification(
                "Book Text Generation Complete",
                f"Successfully created text representations for SBERT training",
                fields=[
                    {"name": "Total Books", "value": f"{total_books:,}", "inline": True},
                    {"name": "Valid Books", "value": f"{valid_books:,}", "inline": True},
                    {"name": "Success Rate", "value": f"{valid_books/total_books*100:.1f}%", "inline": True},
                    {"name": "Text Format", "value": "Title | Genres | Description | Authors", "inline": False},
                    {"name": "Output File", "value": f"`{os.path.basename(self.book_texts_file)}`", "inline": True}
                ]
            )
            
            return book_texts_df
        except Exception as e:
            error_msg = f"Failed to create book texts: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Book Text Generation Failed",
                error_msg,
                error=True
            )
            raise

    def generate_triplets(self, books_df: pd.DataFrame, book_texts_df: pd.DataFrame) -> pd.DataFrame:
        """Generates triplets for training - exact logic from finetune_sbert.py."""
        try:
            self.logger.info("Starting triplet generation...")
            
            # Create a dictionary for faster text lookup
            book_texts_map = book_texts_df.set_index('book_id')['text'].to_dict()
            all_book_ids_with_text = list(book_texts_map.keys())
            
            triplet_data = []
            anchor_count = 0
            positive_count = 0
            failed_negatives = 0
            
            # Iterate through books that have text representations
            for anchor_id, anchor_text in book_texts_map.items():
                try:
                    anchor_info = books_df.loc[anchor_id]
                    similar_books = anchor_info.get('similar_books', [])

                    if len(similar_books) == 0:
                        continue

                    anchor_count += 1
                    # Set of IDs that cannot be negative samples
                    forbidden_ids = {anchor_id} | set(similar_books)

                    # Iterate through positively related books
                    for positive_id in similar_books:
                        # Get positive text, skip if not found in our text map
                        positive_text = book_texts_map.get(int(positive_id))
                        if positive_text is None:
                            continue

                        positive_count += 1
                        
                        # Find a suitable negative sample
                        negative_id = None
                        negative_text = None
                        for _ in range(self.max_negative_search_attempts):
                            potential_neg_id = random.choice(all_book_ids_with_text)
                            
                            if potential_neg_id in forbidden_ids:
                                continue
                            
                            potential_neg_text = book_texts_map.get(potential_neg_id)
                            if potential_neg_text is not None:
                                negative_id = potential_neg_id
                                negative_text = potential_neg_text
                                break

                        # If a suitable negative was found, add the triplet
                        if negative_id and negative_text:
                            triplet_data.append({
                                'anchor': anchor_text,
                                'positive': positive_text,
                                'negative': negative_text
                            })
                        else:
                            failed_negatives += 1

                except KeyError:
                    self.logger.warning(f"Anchor book ID {anchor_id} not found in books_df.")
                    continue

            triplets_df = pd.DataFrame(triplet_data)
            self.logger.info(f"Generated {len(triplets_df)} triplets.")
            
            # Send success notification
            self._send_notification(
                "Triplet Generation Complete",
                f"Successfully generated training triplets for SBERT model",
                fields=[
                    {"name": "Books with Text", "value": f"{len(book_texts_map):,}", "inline": True},
                    {"name": "Anchors Processed", "value": f"{anchor_count:,}", "inline": True},
                    {"name": "Positive Pairs", "value": f"{positive_count:,}", "inline": True},
                    {"name": "Generated Triplets", "value": f"{len(triplets_df):,}", "inline": True},
                    {"name": "Failed Negatives", "value": f"{failed_negatives:,}", "inline": True},
                    {"name": "Success Rate", "value": f"{len(triplets_df)/positive_count*100:.1f}%" if positive_count > 0 else "0%", "inline": True},
                    {"name": "Output File", "value": f"`{os.path.basename(self.triplets_data_file)}`", "inline": False}
                ]
            )
            
            return triplets_df
        except Exception as e:
            error_msg = f"Failed to generate triplets: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "Triplet Generation Failed",
                error_msg,
                error=True
            )
            raise

    def run(self) -> Dict[str, Any]:
        """Execute the SBERT training pipeline - matches finetune_sbert.py exactly."""
        self.logger.info("Starting SBERT fine-tuning step...")
        
        # Send pipeline start notification
        self._send_notification(
            "SBERT Training Started",
            f"Beginning SBERT fine-tuning pipeline: **{self.name}**",
            color=0x0099FF,  # Blue for start
            fields=[
                {"name": "Base Model", "value": f"`{self.model_name}`", "inline": True},
                {"name": "Device", "value": f"{self.device}", "inline": True},
                {"name": "Epochs", "value": f"{self.epochs}", "inline": True},
                {"name": "Batch Size", "value": f"{self.batch_size}", "inline": True},
                {"name": "Learning Rate", "value": f"{self.learning_rate}", "inline": True},
                {"name": "Triplet Margin", "value": f"{self.triplet_margin}", "inline": True}
            ]
        )
        
        outputs = {}

        try:
            # 1. Load and analyze data
            self.logger.info("Loading and analyzing data...")
            books_df = dd.read_parquet(self.books_data_file)
            
            # Display analysis like in finetune_sbert.py
            analysis_results = self.analyze_dataframe(books_df)
            print("DataFrame Analysis:")
            print(analysis_results)
            
            books_df = books_df.compute()
            books_df.set_index('book_id', inplace=True)
            
            authors_df = dd.read_parquet(self.authors_data_file).compute()
            
            # Send data loading notification
            self._send_notification(
                "Data Loading Complete",
                f"Successfully loaded and analyzed datasets",
                fields=[
                    {"name": "Books Loaded", "value": f"{len(books_df):,}", "inline": True},
                    {"name": "Authors Loaded", "value": f"{len(authors_df):,}", "inline": True},
                    {"name": "Books File", "value": f"`{os.path.basename(self.books_data_file)}`", "inline": True},
                    {"name": "Authors File", "value": f"`{os.path.basename(self.authors_data_file)}`", "inline": True}
                ]
            )

            # 2. Create book texts
            book_texts_df = self.create_book_texts(books_df, authors_df)
            book_texts_df.to_parquet(self.book_texts_file, index=False)
            outputs["book_texts_parquet"] = self.book_texts_file

            # 3. Generate triplets
            triplets_df = self.generate_triplets(books_df, book_texts_df)
            
            if not triplets_df.empty:
                triplets_df.to_parquet(self.triplets_data_file, index=False)
                print(f"Successfully saved triplets to {self.triplets_data_file}")
                print("Sample triplets:")
                print(triplets_df.head())
            else:
                error_msg = "No triplets were generated."
                self.logger.error(error_msg)
                self._send_notification(
                    "SBERT Training Failed",
                    error_msg,
                    error=True
                )
                return outputs
            
            outputs["triplets_parquet"] = self.triplets_data_file

            # 4. Data loading & splitting
            self.logger.info("Loading and splitting dataset...")
            full_dataset = load_dataset('parquet', data_files=self.triplets_data_file, split='train')
            
            # Split exactly like finetune_sbert.py
            train_testvalid_split = full_dataset.train_test_split(test_size=self.test_split_size, seed=self.random_state)
            train_dataset = train_testvalid_split['train']
            test_valid_dataset = train_testvalid_split['test']
            
            test_validation_split = test_valid_dataset.train_test_split(test_size=self.validation_split_size, seed=self.random_state)
            validation_dataset = test_validation_split['train']
            test_dataset = test_validation_split['test']
            
            val_anchors = validation_dataset['anchor']
            val_positives = validation_dataset['positive']
            val_negatives = validation_dataset['negative']
            
            test_anchors = test_dataset['anchor']
            test_positives = test_dataset['positive']
            test_negatives = test_dataset['negative']
            
            self.logger.info(f"Train size: {len(train_dataset)}")
            self.logger.info(f"Validation size: {len(validation_dataset)}")
            self.logger.info(f"Test size: {len(test_dataset)}")
            
            # Send data splitting notification
            self._send_notification(
                "Dataset Splitting Complete",
                f"Successfully split dataset for training and evaluation",
                fields=[
                    {"name": "Total Triplets", "value": f"{len(full_dataset):,}", "inline": True},
                    {"name": "Training Set", "value": f"{len(train_dataset):,}", "inline": True},
                    {"name": "Validation Set", "value": f"{len(validation_dataset):,}", "inline": True},
                    {"name": "Test Set", "value": f"{len(test_dataset):,}", "inline": True},
                    {"name": "Test Split", "value": f"{self.test_split_size*100:.0f}%", "inline": True},
                    {"name": "Val Split", "value": f"{self.validation_split_size*100:.0f}%", "inline": True}
                ]
            )

            # 5. Baseline performance
            self.logger.info("Evaluating baseline model...")
            baseline_model = SentenceTransformer(self.model_name, device=self.device)
            
            baseline_evaluator = TripletEvaluator(
                anchors=test_anchors,
                positives=test_positives,
                negatives=test_negatives,
                main_similarity_function='cosine',
                margin=self.triplet_margin,
            )
            
            baseline_results = baseline_evaluator(baseline_model)
            baseline_accuracy = baseline_results[baseline_evaluator.primary_metric]
            self.logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
            outputs["baseline_accuracy"] = baseline_accuracy
            
            # Send baseline evaluation notification
            self._send_notification(
                "Baseline Model Evaluation Complete",
                f"Evaluated pre-trained model performance",
                fields=[
                    {"name": "Model", "value": f"`{self.model_name}`", "inline": True},
                    {"name": "Test Accuracy", "value": f"{baseline_accuracy:.4f}", "inline": True},
                    {"name": "Similarity Function", "value": "Cosine", "inline": True},
                    {"name": "Evaluation Set", "value": f"{len(test_dataset):,} triplets", "inline": True}
                ]
            )

            # 6. Fine-tuning
            self.logger.info("Starting fine-tuning...")
            
            # Send fine-tuning start notification
            self._send_notification(
                "Fine-tuning Started",
                f"Beginning SBERT model fine-tuning process",
                color=0xFFA500,  # Orange for progress
                fields=[
                    {"name": "Training Examples", "value": f"{len(train_dataset):,}", "inline": True},
                    {"name": "Epochs", "value": f"{self.epochs}", "inline": True},
                    {"name": "Batch Size", "value": f"{self.batch_size}", "inline": True},
                    {"name": "Learning Rate", "value": f"{self.learning_rate}", "inline": True}
                ]
            )
            
            # Define loss function
            train_loss = losses.TripletLoss(
                model=baseline_model,
                distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
                triplet_margin=self.triplet_margin,
            )
            
            # Create data loader
            train_examples = []
            for i in range(len(train_dataset)):
                example = train_dataset[i]
                train_examples.append(InputExample(texts=[example['anchor'], example['positive'], example['negative']]))
            
            self.logger.info(f"Created {len(train_examples)} training examples.")
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)
            
            # Define validation evaluator
            validation_evaluator = TripletEvaluator(
                anchors=val_anchors,
                positives=val_positives,
                negatives=val_negatives,
                name='validation',
                show_progress_bar=True,
                write_csv=True 
            )
            
            # Calculate warmup steps
            num_training_steps = len(train_dataloader) * self.epochs
            warmup_steps = int(num_training_steps * self.warmup_steps_ratio)
            self.logger.info(f"Total training steps: {num_training_steps}")
            self.logger.info(f"Warmup steps: {warmup_steps}")
            
            # Fit the model
            print("\n--- Starting Fine-tuning ---")
            baseline_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=validation_evaluator,
                epochs=self.epochs,
                evaluation_steps=self.evaluation_steps,
                warmup_steps=warmup_steps,
                output_path=self.output_base_path,
                save_best_model=True,
                optimizer_params={'lr': self.learning_rate},
                checkpoint_path=self.checkpoint_path,
                checkpoint_save_steps=self.save_steps,
                checkpoint_save_total_limit=self.checkpoint_limit
            )
            print("--- Fine-tuning Finished ---")
            
            outputs["best_model_path"] = self.output_base_path
            outputs["checkpoint_path"] = self.checkpoint_path
            
            # Send fine-tuning completion notification
            self._send_notification(
                "Fine-tuning Complete",
                f"Successfully completed SBERT model fine-tuning",
                fields=[
                    {"name": "Training Steps", "value": f"{num_training_steps:,}", "inline": True},
                    {"name": "Warmup Steps", "value": f"{warmup_steps:,}", "inline": True},
                    {"name": "Model Path", "value": f"`{os.path.basename(self.output_base_path)}`", "inline": True},
                    {"name": "Checkpoints", "value": f"`{os.path.basename(self.checkpoint_path)}`", "inline": True}
                ]
            )

            # 7. Plot validation performance
            self._plot_validation_performance()
            outputs["validation_plot"] = os.path.join(self.output_base_path, 'validation_accuracy_plot.png')

            # 8. Evaluate fine-tuned model
            self.logger.info("Evaluating fine-tuned model...")
            model_finetuned = SentenceTransformer(self.output_base_path, device=self.device)
            
            final_test_evaluator = TripletEvaluator(
                anchors=test_anchors,
                positives=test_positives,
                negatives=test_negatives,
                name='finetuned-test',
                show_progress_bar=True
            )
            
            finetuned_results = final_test_evaluator(model_finetuned, output_path=self.eval_output_path)
            finetuned_accuracy = finetuned_results[final_test_evaluator.primary_metric]
            self.logger.info(f"Fine-tuned accuracy: {finetuned_accuracy:.4f}")
            outputs["finetuned_accuracy"] = finetuned_accuracy

            # 9. Plot comparison
            self._plot_comparison(baseline_accuracy, finetuned_accuracy)
            outputs["comparison_plot"] = os.path.join(self.output_base_path, 'comparison_accuracy_plot.png')

            # Calculate improvement
            improvement = finetuned_accuracy - baseline_accuracy
            improvement_pct = (improvement / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0

            self.logger.info("SBERT fine-tuning step finished.")
            
            # Send final completion notification
            self._send_notification(
                "SBERT Training Complete! 🎉",
                f"Successfully completed entire SBERT fine-tuning pipeline: **{self.name}**",
                color=0x00FF00,  # Green for success
                fields=[
                    {"name": "Baseline Accuracy", "value": f"{baseline_accuracy:.4f}", "inline": True},
                    {"name": "Fine-tuned Accuracy", "value": f"{finetuned_accuracy:.4f}", "inline": True},
                    {"name": "Improvement", "value": f"+{improvement:.4f} ({improvement_pct:+.1f}%)", "inline": True},
                    {"name": "Training Data", "value": f"{len(train_dataset):,} triplets", "inline": True},
                    {"name": "Model Type", "value": "Sentence-BERT", "inline": True},
                    {"name": "Device Used", "value": f"{self.device}", "inline": True},
                    {"name": "Model Output", "value": f"`{os.path.basename(self.output_base_path)}`", "inline": True},
                    {"name": "Plots Generated", "value": "✅ Validation & Comparison", "inline": True}
                ]
            )
            
            self.output_data = outputs
            return outputs
            
        except Exception as e:
            error_msg = f"SBERT training pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self._send_notification(
                "SBERT Training Pipeline Failed",
                error_msg,
                error=True
            )
            raise

    def _plot_validation_performance(self):
        """Plot validation performance during training."""
        print("\n--- Plotting Validation Performance ---")
        eval_filepath = os.path.join(self.eval_output_path, "triplet_evaluation_validation_results.csv")
        
        try:
            eval_results = pd.read_csv(eval_filepath)
            if 'steps' in eval_results.columns and 'accuracy_cosine' in eval_results.columns:
                plt.figure(figsize=(10, 5))
                plt.plot(eval_results['steps'], eval_results['accuracy_cosine'], marker='o', linestyle='-')
                plt.title('Validation Accuracy during Fine-tuning')
                plt.xlabel('Training Steps')
                plt.ylabel('Cosine Accuracy')
                plt.grid(True)
                plot_save_path = os.path.join(self.output_base_path, 'validation_accuracy_plot.png')
                plt.savefig(plot_save_path)
                print(f"Validation plot saved to: {plot_save_path}")
                plt.close()
            else:
                print(f"Columns 'steps' or 'accuracy_cosine' not found in {eval_filepath}. Cannot plot.")
        except FileNotFoundError:
            print(f"Evaluation results file not found at: {eval_filepath}")
        except Exception as e:
            print(f"An error occurred during plotting: {e}")

    def _plot_comparison(self, baseline_acc: float, finetuned_acc: float):
        """Plot comparison between baseline and fine-tuned model."""
        print("\n--- Results Comparison ---")
        print(f"Baseline Test Accuracy:   {baseline_acc:.4f}")
        print(f"Fine-tuned Test Accuracy: {finetuned_acc:.4f}")
        improvement = finetuned_acc - baseline_acc
        print(f"Improvement:              {improvement:.4f} ({improvement/baseline_acc:.1%})")

        labels = ['Baseline', 'Fine-tuned']
        accuracies = [baseline_acc, finetuned_acc]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, accuracies, color=['skyblue', 'lightgreen'])

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

        plt.ylabel('Accuracy (Cosine)')
        plt.title('Baseline vs. Fine-tuned Model Accuracy')
        plt.ylim(0, max(accuracies) * 1.1)
        plot_save_path = os.path.join(self.output_base_path, 'comparison_accuracy_plot.png')
        plt.savefig(plot_save_path)
        print(f"Comparison plot saved to: {plot_save_path}")
        plt.close()