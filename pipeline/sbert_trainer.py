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
from .core import PipelineStep
from ..utils import get_device


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
        self.logger.info("Creating book text representations...")
        
        book_texts = []
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

        book_texts_df = pd.DataFrame(book_texts)
        self.logger.info(f"Created {len(book_texts_df)} book text entries.")
        return book_texts_df

    def generate_triplets(self, books_df: pd.DataFrame, book_texts_df: pd.DataFrame) -> pd.DataFrame:
        """Generates triplets for training - exact logic from finetune_sbert.py."""
        self.logger.info("Starting triplet generation...")
        
        # Create a dictionary for faster text lookup
        book_texts_map = book_texts_df.set_index('book_id')['text'].to_dict()
        all_book_ids_with_text = list(book_texts_map.keys())
        
        triplet_data = []
        
        # Iterate through books that have text representations
        for anchor_id, anchor_text in book_texts_map.items():
            try:
                anchor_info = books_df.loc[anchor_id]
                similar_books = anchor_info.get('similar_books', [])

                if len(similar_books) == 0:
                    continue

                # Set of IDs that cannot be negative samples
                forbidden_ids = {anchor_id} | set(similar_books)

                # Iterate through positively related books
                for positive_id in similar_books:
                    # Get positive text, skip if not found in our text map
                    positive_text = book_texts_map.get(int(positive_id))
                    if positive_text is None:
                        continue

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

            except KeyError:
                self.logger.warning(f"Anchor book ID {anchor_id} not found in books_df.")
                continue

        triplets_df = pd.DataFrame(triplet_data)
        self.logger.info(f"Generated {len(triplets_df)} triplets.")
        return triplets_df

    def run(self) -> Dict[str, Any]:
        """Execute the SBERT training pipeline - matches finetune_sbert.py exactly."""
        self.logger.info("Starting SBERT fine-tuning step...")
        outputs = {}

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
            self.logger.error("No triplets were generated.")
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

        # 6. Fine-tuning
        self.logger.info("Starting fine-tuning...")
        
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

        self.logger.info("SBERT fine-tuning step finished.")
        self.output_data = outputs
        return outputs

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