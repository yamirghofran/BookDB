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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FILE = "../data/books_triplets.parquet"
OUTPUT_PATH = f'sbert-output/finetuning-{MODEL_NAME}-books'
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'eval'), exist_ok=True) # For evaluator output

# 1. Load Dataset
books_df = dd.read_parquet('../data/reduced_books.parquet')

def analyze_dataframe(df):
    # Get column info
    cols = df.columns
    dtypes = df.dtypes
    
    # Calculate total rows
    total_rows = len(df.compute())
    
    # Initialize lists to store results
    results = []
    
    # Analyze each column
    for col in cols:
        # Count non-null values
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
    
    # Convert results to pandas DataFrame for better display
    results_df = pd.DataFrame(results)
    return results_df.sort_values('Null Percentage', ascending=False)

# Display the analysis
print("DataFrame Analysis:")
print(analyze_dataframe(books_df))

books_df = books_df.compute()
books_df.set_index('book_id', inplace=True)
books_df.head()

def extract_genres(popular_shelves):
    """
    Extracts potential genres from a list of popular shelves dictionaries,
    adding only the base genre keyword found.

    Args:
        popular_shelves: A list of dictionaries, where each dictionary has
                         'count' and 'name' keys.

    Returns:
        A list of unique base genre names found, or an empty list on error.
    """
    try:
        if not isinstance(popular_shelves, np.ndarray) or len(popular_shelves) == 0:
            return []
        
        # Use a set to store unique base genres found
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
        # Sort keywords by length descending to match longer phrases first (e.g., "science fiction" before "science")
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
            
            shelf_name = shelf['name'].lower().strip() # Normalize shelf name

            # Skip if shelf name contains any ignore keywords
            if any(ignore in shelf_name for ignore in ignore_keywords):
                continue

            # Check if any genre keyword is present in the shelf name
            for keyword in genre_keywords:
                # Use word boundaries or careful checks to avoid partial matches (e.g., 'art' in 'heart')
                # Simple substring check for now, might need refinement depending on data
                if keyword in shelf_name: 
                    found_genres.add(keyword) # Add the base keyword
                    # Optional: break here if you only want the first/longest match per shelf
                    # break 

        return sorted(list(found_genres))
    except Exception as e:
        print(f"Error in extract_genres function: {e}")
        # Log the error message
        logging.error("Error in extract_genres function", exc_info=True)
        return []

# 2. Create Item Texts
# In this step, we will create a combined text representing each book in the format of "Title: {title} | Genres: {genres} | Description: {description} | Publisher: {publisher} | Authors: {authors}
authors_df = dd.read_parquet("../data/new_authors.parquet")
authors_df = authors_df.compute()

book_texts = []
for index, row in books_df.iterrows():
    if row['description'] == '' or row['title'] == '':
        continue
    if row['description'] is None or row['title'] is None:
        continue
    genres = extract_genres(row['popular_shelves'])
    authors = []
    for author_id in row['authors']:
        author = authors_df.loc[authors_df['author_id'] == author_id]
        if not author.empty:
            authors.append(author.iloc[0]['name'])
    book_text = f"Title: {row['title']} | Genres: {', '.join(genres)} | Description: {row['description']} | Authors: {', '.join(authors)}"
    book_texts.append({'book_id': index, 'text': book_text})


book_texts_df = pd.DataFrame(book_texts)
book_texts_df.to_parquet('data/book_texts.parquet', index=False)

# 3. Creating Finetuning Data (Triplet Loss Approach)
# In this step, we will create records in the format of (anchor_text, positive_text, negative_text). The goal is to teach the model that the anchor_text should be closer to positive_text than negative_text.

# For positive text, we will use the text of books in the `similar_books` attribute for each book. For negative texts, we will get the text from a randomly selected book that doesn't share a genre with the book.

book_texts_df = dd.read_parquet('../data/book_texts.parquet')
book_texts_df = book_texts_df.compute()


book_texts_df.head()

if books_df.index.name != 'book_id':
    books_df.set_index('book_id', inplace=True)

# Create a dictionary for faster text lookup {book_id: text}
book_texts_map = book_texts_df.set_index('book_id')['text'].to_dict()

# Get a list of all book IDs that have text representations
all_book_ids_with_text = list(book_texts_map.keys())
print(f"Total books with text representation: {len(all_book_ids_with_text)}")

triplet_data = []
MAX_NEGATIVE_SEARCH_ATTEMPTS = 100 # Limit attempts to find a negative sample

print(f"Starting triplet generation for {len(book_texts_map)} books...")

# Iterate through books that have text representations
for anchor_id, anchor_text in book_texts_map.items():
    try:
        # Get anchor book's details from the original dataframe
        anchor_info = books_df.loc[anchor_id]
        similar_books = anchor_info.get('similar_books', [])

        #print(f"Processing anchor book ID: {anchor_id} with similar books: {similar_books}")

        if len(similar_books)==0: # Skip if no similar books listed
            continue

        # Set of IDs that cannot be negative samples
        forbidden_ids = {anchor_id} | set(similar_books)

        # Iterate through positively related books
        for positive_id in similar_books:
            # Get positive text, skip if not found in our text map
            positive_text = book_texts_map.get(int(positive_id))
            #print(f"Positive book ID: {positive_id} with text: {positive_text}")
            if positive_text is None:
                continue

            # Find a suitable negative sample
            negative_id = None
            negative_text = None
            for _ in range(MAX_NEGATIVE_SEARCH_ATTEMPTS):
                # Choose a random book ID
                potential_neg_id = random.choice(all_book_ids_with_text)
                #print(f"Potential negative book ID: {potential_neg_id}")
            
                # Check if it's a forbidden ID
                if potential_neg_id in forbidden_ids:
                    print(f"Skipping forbidden ID: {potential_neg_id}")
                    continue
            
                # Get the text for this potential negative
                potential_neg_text = book_texts_map.get(potential_neg_id)
                #print(f"Potential negative text: {potential_neg_text}")
                if potential_neg_text is not None:
                    #print(f"Found suitable negative ID: {potential_neg_id} with text: {potential_neg_text}")
                    negative_id = potential_neg_id
                    negative_text = potential_neg_text
                    break

            # If a suitable negative was found, add the triplet
            if negative_id and negative_text:
                #print(f"Adding triplet: {anchor_id}, {positive_id}, {negative_id}")
                # Using InputExample format directly if needed for SentenceTransformer later
                # triplet_examples.append(InputExample(texts=[anchor_text, positive_text, negative_text]))
                # Or using a dictionary for DataFrame creation
                triplet_data.append({
                    'anchor': anchor_text,
                    'positive': positive_text,
                    'negative': negative_text
                })
            else:
                # Optional: Log if a negative couldn't be found for a pair
                print(f"Warning: Could not find suitable negative for anchor {anchor_id}, positive {positive_id}")

    except KeyError:
        print(f"Warning: Anchor book ID {anchor_id} not found in books_df.")
        continue # Skip if anchor_id is not in the original books_df

print(f"Generated {len(triplet_data)} triplets.")

# Create DataFrame from the collected triplets
triplets_df = pd.DataFrame(triplet_data)

# Save the triplets DataFrame to a Parquet file
if not triplets_df.empty:
    output_path = '../data/books_triplets.parquet'
    triplets_df.to_parquet(output_path, index=False)
    print(f"Successfully saved triplets to {output_path}")
else:
    print("No triplets were generated.")

# Display the head of the new DataFrame (optional)
print(triplets_df.head())
triplets_df.iloc[78]['anchor']

# 4. Data Loading & Splitting
# Load directly from parquet
full_dataset = load_dataset('parquet', data_files=DATA_FILE, split='train')

# First split: 80% train, 20% temporary (for validation + test)
train_testvalid_split = full_dataset.train_test_split(test_size=0.2, seed=RANDOM_STATE)
train_dataset = train_testvalid_split['train']
test_valid_dataset = train_testvalid_split['test'] # This is 20% of the original

# Second split: Split the 20% into 50% validation (10% of original) and 50% test (10% of original)
# test_size=0.5 means 50% of the test_valid_dataset (which is 20% of total) goes to the test set
test_validation_split = test_valid_dataset.train_test_split(test_size=0.5, seed=RANDOM_STATE)
validation_dataset = test_validation_split['train'] # This is 10% of the original
test_dataset = test_validation_split['test']       # This is 10% of the original

val_anchors = validation_dataset['anchor']
val_positives = validation_dataset['positive']
val_negatives = validation_dataset['negative']
print("Validation Anchors:", val_anchors[:5])
print("Validation Positives:", val_positives[:5])
print("Validation Negatives:", val_negatives[:5])

test_anchors = test_dataset['anchor']
test_positives = test_dataset['positive']
test_negatives = test_dataset['negative']


print("Test Anchors:", test_anchors[:5])
print("Test Positives:", test_positives[:5])
print("Test Negatives:", test_negatives[:5])

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(validation_dataset)}")
print(f"Test size: {len(test_dataset)}")

# ## 5. Baseline Performence
# In this step, we pick `all-MiniLM-L6-v2` as our base model and evaluate its performance on our test dataset.

# We will later compare the performance of our finetuned model against this baseline performance.

baseline_model = SentenceTransformer(MODEL_NAME)

baseline_evaluator = TripletEvaluator(
    anchors=test_anchors,
    positives=test_positives,
    negatives=test_negatives,
    main_similarity_function='cosine',
    margin=0.5,
)

baseline_results = baseline_evaluator(baseline_model)
baseline_accuracy = baseline_results[baseline_evaluator.primary_metric]
print(baseline_evaluator.primary_metric)
print(f"Accuracy: {baseline_results[baseline_evaluator.primary_metric]:.4f}")

# 6. Fine-tuning
# In this step, we pick `all-MiniLM-L6-v2` as our base model and fine-tune it with our data using the `TripletLoss` loss function. We use Optuna for hyperparameter tuning.

# We choose `all-MiniLM-L6-v2` because it is a small but performant model compared to the [other models](https://sbert.net/docs/sentence_transformer/pretrained_models.html#original-models).


# Check for GPU availability (Prioritize MPS on Mac, then CUDA, then CPU)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")
if device == "cpu":
    print("Warning: Training on CPU will be very slow.")
elif device == "mps":
    print("Using Apple Metal Performance Shaders (MPS).")

# Set Hyperparameters
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
TRIPLET_MARGIN = 0.5
WARMUP_STEPS_RATIO = 0.1 # 10% of total steps for warmup
EVALUATION_STEPS = 500   # Evaluate every N steps
SAVE_STEPS = 1000        # Save checkpoint every N steps (optional)

# Define Loss Function
train_loss = losses.TripletLoss(
    model=baseline_model,
    distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
    triplet_margin=TRIPLET_MARGIN,
)

# Create Data Loader
# For Training DataLoader (needs list of InputExample)
train_examples = []
for i in range(len(train_dataset)):
    example = train_dataset[i]
    train_examples.append(InputExample(texts=[example['anchor'], example['positive'], example['negative']]))
print(f"Created {len(train_examples)} training examples.")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# Define Validation Evaluator
validation_evaluator = TripletEvaluator(
    anchors=val_anchors,
    positives=val_positives,
    negatives=val_negatives,
    name='validation',
    show_progress_bar=True,
    write_csv=True 
)

# Calculate Warmup Steps
num_training_steps = len(train_dataloader) * EPOCHS
warmup_steps = int(num_training_steps * WARMUP_STEPS_RATIO)
print(f"Total training steps: {num_training_steps}")
print(f"Warmup steps: {warmup_steps}")

# Fitting the Model
print("\n--- Starting Fine-tuning ---")
# Use model.fit for training
baseline_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=validation_evaluator,
    epochs=EPOCHS,
    evaluation_steps=EVALUATION_STEPS,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_PATH,
    save_best_model=True,  # Saves the best model based on validation evaluator
    optimizer_params={'lr': LEARNING_RATE},
    checkpoint_path=os.path.join(OUTPUT_PATH, 'checkpoints'), # Optional: Save checkpoints
    checkpoint_save_steps=SAVE_STEPS,                          # Optional: Frequency
    checkpoint_save_total_limit=2                              # Optional: Limit checkpoints
)
print("--- Fine-tuning Finished ---")


print("\n--- Plotting Validation Performance ---")
eval_filepath = os.path.join(OUTPUT_PATH, "eval", "triplet_evaluation_validation_results.csv")

try:
    eval_results = pd.read_csv(eval_filepath)
    # Check if 'steps' and 'cosine_accuracy' columns exist
    if 'steps' in eval_results.columns and 'accuracy_cosine' in eval_results.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(eval_results['steps'], eval_results['accuracy_cosine'], marker='o', linestyle='-')
        plt.title('Validation Accuracy during Fine-tuning')
        plt.xlabel('Training Steps')
        plt.ylabel('Cosine Accuracy')
        plt.grid(True)
        plot_save_path = os.path.join(OUTPUT_PATH, 'validation_accuracy_plot.png')
        plt.savefig(plot_save_path)
        print(f"Validation plot saved to: {plot_save_path}")
        plt.show()
    else:
        print(f"Columns 'steps' or 'accuracy_cosine' not found in {eval_filepath}. Cannot plot.")

except FileNotFoundError:
    print(f"Evaluation results file not found at: {eval_filepath}")
    print("Plotting skipped. Ensure 'write_csv=True' was set for the validation evaluator.")
except Exception as e:
    print(f"An error occurred during plotting: {e}")

# Evaluate Finetuned Model on Test
print(f"Loading best model from: {OUTPUT_PATH}")
model_finetuned = SentenceTransformer(OUTPUT_PATH)

# Create evaluator for the test set (identical to baseline setup)
final_test_evaluator = TripletEvaluator(
    anchors=test_anchors,
    positives=test_positives,
    negatives=test_negatives,
    name='finetuned-test',
    show_progress_bar=True
)

# Evaluate the fine-tuned model
print("Evaluating fine-tuned model on test set...")
finetuned_results = final_test_evaluator(model_finetuned)
finetuned_accuracy = finetuned_results[final_test_evaluator.primary_metric]
print(f"Fine-tuned Test Set Accuracy (Cosine): {finetuned_accuracy:.4f}")


# Comparison
print("\n--- Results Comparison ---")
print(f"Baseline Test Accuracy:   {baseline_accuracy:.4f}")
print(f"Fine-tuned Test Accuracy: {finetuned_accuracy:.4f}")
improvement = finetuned_accuracy - baseline_accuracy
print(f"Improvement:              {improvement:.4f} ({improvement/baseline_accuracy:.1%})")

# Data for plotting
labels = ['Baseline', 'Fine-tuned']
accuracies = [baseline_accuracy, finetuned_accuracy]

# Create bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, accuracies, color=['skyblue', 'lightgreen'])

# Add accuracy values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center') # Add text labels

plt.ylabel('Accuracy (Cosine)')
plt.title('Baseline vs. Fine-tuned Model Accuracy')
plt.ylim(0, max(accuracies) * 1.1) # Adjust y-axis limit for better visualization
plt.show()


