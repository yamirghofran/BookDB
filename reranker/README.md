# Reranker Data Engineering and Model Fine-tuning Process

## 1. Introduction

This document outlines the data engineering pipeline designed to prepare data for training a cross-encoder reranker model, and the subsequent process for fine-tuning that model. The goal is to transform raw book metadata and user interaction data into a structured format of (user_context, book_text, label) triplets suitable for training, and then to train a model that can effectively rerank book recommendations.

The process involves:
1.  Initial data loading, cleaning, and feature engineering using the `reranker_data_engineering.ipynb` Jupyter notebook.
2.  Generation of positive and negative training pairs using the `generate_training_data.py` script (which leverages `cross_encoder_data_prep.py`).
3.  Post-processing and sub-sampling of these training pairs using the `process_parquet_chunks.py` script.
4.  Verification and analysis of the generated datasets within the `reranker_data_engineering.ipynb` notebook.
5.  Fine-tuning a cross-encoder model using the prepared data, primarily with the `reranker_model_tuning_colab.ipynb` notebook.

## 2. Prerequisites

### 2.1. Software
*   Python 3.x
*   Jupyter Notebook or JupyterLab
*   Google Colab (for the recommended model fine-tuning process)
*   Required Python libraries (install via `pip install -r requirements.txt` if a `requirements.txt` is provided, otherwise install individually):
    *   `pandas`
    *   `numpy`
    *   `dask`
    *   `dask[diagnostics]` (for ProgressBar)
    *   `boto3` (for AWS S3 / R2 interaction)
    *   `pyarrow` (for reading/writing Parquet files efficiently with Dask/Pandas)
    *   `matplotlib` (for any plotting in the notebook)
    *   `psutil` (for memory monitoring in the notebook)
    *   `sentence-transformers` (for model fine-tuning)
    *   `torch` (as a dependency for sentence-transformers)
    *   `scikit-learn` (for evaluation metrics)
    *   `huggingface_hub`

### 2.2. Input Data
The pipeline expects the following initial datasets, typically stored in a cloud object storage like Cloudflare R2 and downloaded locally to the `data/` directory within `BookDB/reranker/`:

*   `data/reduced_books.parquet`: Contains book metadata. Expected columns include `book_id`, `title`, `description`, `popular_shelves` (list of shelf names/tags), `authors` (list of author IDs).
*   `data/reduced_interactions.parquet`: Contains user-item interaction data. Expected columns include `user_id`, `book_id`, `rating`.
*   `data/new_authors.parquet`: Maps author IDs to author names. Expected columns: `author_id`, `name`.
*   `data/book_texts_reduced.parquet`: Contains processed textual content for books. Expected columns: `book_id`, `text` (where text could be a concatenation of title, description, genres, etc.).

The model fine-tuning step expects the output of the data engineering process:
*   `data/processed_training_pairs_parts_0_to_12.parquet` (or similarly named file from `process_parquet_chunks.py`): Contains `user_id`, `user_ctx`, `book_text`, and `label`.

### 2.3. Cloud Storage Credentials
*   If using Cloudflare R2 (as configured in the notebook's helper functions), ensure you have:
    *   R2 Endpoint URL
    *   R2 Access Key ID
    *   R2 Secret Access Key
*   **Security Best Practice:** Store these credentials securely, for example, as environment variables or through a secure credentials management system. The notebook examples might show placeholder credentials that need to be replaced.

## 3. Step-by-Step Data Engineering Process

The data engineering pipeline is executed in the following sequence:

### Step 1: Initial Data Loading & Feature Engineering (`reranker_data_engineering.ipynb` - Part 1)

**Objective:** Load raw data from cloud storage, perform initial cleaning, extensive feature engineering, and save the processed DataFrames as Parquet files that will serve as inputs to the first script.

**Location:** `BookDB/reranker/reranker_data_engineering.ipynb`

**Key Operations:**

1.  **Setup & Imports:**
    *   Import necessary libraries (pandas, dask, boto3, etc.).
    *   Define helper functions: `download_from_r2`, `list_bucket_contents`, `upload_to_r2` for cloud storage interaction.
2.  **Load Raw Data from Cloud:**
    *   Download `reduced_books.parquet`, `reduced_interactions.parquet`, `new_authors.parquet`, and `book_texts_reduced.parquet` from R2 to the local `data/` directory.
    *   Load these into Dask DataFrames.
3.  **Book Metadata Engineering (`books_df` -> `data/reduce_books_df.parquet`):**
    *   Analysis of `books_df` (structure, nulls).
    *   Genre extraction from `popular_shelves`.
    *   Author name mapping using `authors_df`.
    *   Transformation to select relevant columns and create comma-separated string columns for `genre` and `authors`.
    *   **Output:** Processed book metadata saved as `data/reduce_books_df.parquet`.
4.  **User Interaction Data Engineering (`interactions_df` -> `data/sampled_users_book.parquet`):**
    *   Creation of `user_books_df`: For each user, a list of `book_id`s they interacted with, sorted by `rating`.
    *   Sampling of users (e.g., 50,000) from `user_books_df` to create `sampled_users_book`.
    *   **Output:** Sampled user-book interaction lists saved as `data/sampled_users_book.parquet`.
5. **Cloud Upload (Optional but Recommended):**
    *   The generated files (`reduce_books_df.parquet`, `sampled_users_book.parquet`) and the downloaded `book_texts_reduced.parquet` are uploaded back to R2 to ensure they are accessible for the next script, especially if running in a different environment.

**Key Local Outputs from this Notebook Phase (typically in `BookDB/reranker/data/`):**
*   `data/reduce_books_df.parquet` (processed book metadata)
*   `data/sampled_users_book.parquet` (sampled user-book interaction lists)
*   `data/book_texts_reduced.parquet` (downloaded book textual content, verified)

### Step 2: Generating Training Pairs (`generate_training_data.py`)

**Objective:** Create positive and negative training examples (user context, book text, label) for the reranker model.

**Script:** `BookDB/reranker/generate_training_data.py`
**Helper Module:** `BookDB/reranker/cross_encoder_data_prep.py`

**Inputs (read by the script from the `data/` directory relative to its location):**
*   `data/sampled_users_book.parquet`
*   `data/reduce_books_df.parquet`
*   `data/book_texts_reduced.parquet`

**Process Overview (handled by `generate_training_data_dask` in `cross_encoder_data_prep.py`):**
1.  Loads the three input Parquet files using Dask.
2.  For each user in `sampled_users_book.parquet`:
    a.  **Positive Pairs:** For each book in the user's `books_read` list:
        i.  A "user context" (`user_ctx`) is generated using `make_user_context`. This context is typically a string describing other books the user liked and their preferred genres, excluding the current positive book (leave-one-out).
        ii. The text for the current positive book (`book_text`) is retrieved.
        iii. A positive training pair (`user_id`, `user_ctx`, `book_id`, `book_text`, `label=1`) is created.
    b.  **Negative Pairs:**
        i.  A pool of books the user has *not* interacted with is identified.
        ii. For each positive pair, `neg_ratio` negative book samples are drawn.
        iii. For each negative book, using the same `user_ctx`, a negative training pair (`user_id`, `user_ctx`, `book_id`, `book_text`, `label=0`) is created.
3.  Uses Dask for distributed processing.

**Output:**
*   A directory named `data/training_pairs.parquet/` containing multiple Parquet part-files.

### Step 3: Post-Processing and Fine-grained Sampling of Training Pairs (`process_parquet_chunks.py`)

**Objective:** Apply more refined sampling logic to balance the dataset and control samples per user.

**Script:** `BookDB/reranker/process_parquet_chunks.py`

**Input:**
*   The `data/training_pairs.parquet/` directory. The script is configured to read specific part-files.

**Process Overview:**
1.  Iterates through specified Parquet part-files (chunks).
2.  `process_individual_chunk` function:
    a.  Identifies positive and negative interactions per user in the chunk.
    b.  **Positive Samples:** If positives >= `MAX_POSITIVES_TO_SAMPLE` (3), samples 3. If between `MIN_POSITIVES_TO_KEEP_USER` (3) and 3, keeps all. If < 3, positives are dropped.
    c.  **Negative Samples:** For each selected positive, samples `NEGATIVES_PER_POSITIVE` (3) negatives for that user from that chunk.
3.  Concatenates processed data from all chunks.

**Output:**
*   A single Parquet file (e.g., `data/processed_training_pairs_parts_0_to_12.parquet`), containing the refined training pairs.

### Step 4: Verification and Analysis of Generated/Processed Data (`reranker_data_engineering.ipynb` - Part 2)

**Objective:** Load and inspect datasets produced by scripts for integrity and characteristics.

**Location:** `BookDB/reranker/reranker_data_engineering.ipynb` (cells after script execution points)

**Key Operations:**
1.  **Analyze `generate_training_data.py` output:**
    *   Load `data/training_pairs.parquet/` into Dask.
    *   Display `head()`, `shape`.
    *   Analyze label distribution, example contexts/texts, positives per user on a sample.
2.  **Analyze `process_parquet_chunks.py` output (Implied):**
    *   Perform similar analysis on the final processed file (e.g., `data/processed_training_pairs_parts_0_to_12.parquet`).

## 4. Data Schema Overview (Final Training Data)

The final processed training data (output of `process_parquet_chunks.py`) typically has:
*   `user_id`: Identifier for the user.
*   `user_ctx`: Textual representation of the user's context.
*   `book_id`: Identifier for the candidate book.
*   `book_text`: Textual representation of the candidate book.
*   `label`: Target label (1 for positive, 0 for negative).

## 5. Running the Full Data Engineering Pipeline

1.  **Setup:** Prerequisites (software, data, credentials).
2.  **Notebook - Part 1 (`reranker_data_engineering.ipynb`):**
    *   Run cells sequentially to generate `data/reduce_books_df.parquet`, `data/sampled_users_book.parquet`, and download `data/book_texts_reduced.parquet`.
3.  **Script 1 (`generate_training_data.py`):**
    *   In `BookDB/reranker/` terminal, run: `python generate_training_data.py`.
    *   Creates `data/training_pairs.parquet/`.
4.  **Script 2 (`process_parquet_chunks.py`):**
    *   Run: `python process_parquet_chunks.py`.
    *   Creates final processed file (e.g., `data/processed_training_pairs_parts_0_to_12.parquet`).
    *   *Note:* May need to adjust `parts_to_load_indices` in the script.
5.  **Notebook - Part 2 (Analysis in `reranker_data_engineering.ipynb`):**
    *   Run cells under "Inspecting and Post-processing Generated Training Pairs" (or similar title) to analyze script outputs.

## 6. Model Fine-tuning (`reranker_model_tuning_colab.ipynb`)

Once the training data has been generated (e.g., `data/processed_training_pairs_parts_0_to_12.parquet`), the next stage is to fine-tune a cross-encoder model.

### 6.1. Notebook Versions for Fine-tuning

Two Jupyter notebooks are available for model fine-tuning within the `BookDB/reranker/` directory:

*   **`reranker_model_tuning.ipynb`**: This was an initial version intended for local model tuning. However, due to computational limitations (e.g., lack of a local GPU, insufficient RAM/CPU for larger models and datasets), this notebook is **incomplete and was not used for the final model training.**
*   **`reranker_model_tuning_colab.ipynb`**: This notebook is specifically designed and adapted for fine-tuning on **Google Colab**, leveraging its free GPU resources. **This is the version that was used for the actual model fine-tuning process and is recommended for reproducible performance.**

The following documentation details the steps within `reranker_model_tuning_colab.ipynb`.

### 6.2. Fine-tuning Process using `reranker_model_tuning_colab.ipynb`

**Objective:** Fine-tune a pre-trained cross-encoder model on the custom-prepared book ranking dataset to improve its ability to distinguish relevant book recommendations for a given user context.

**Location:** `BookDB/reranker/reranker_model_tuning_colab.ipynb` (to be run in a Google Colab environment)

**Key Operations:**

1.  **Setup and Environment (Colab):**
    *   **Install Dependencies:** Installs/updates `sentence-transformers`, `pandas`, `pyarrow`, and `huggingface_hub`.
    *   **Mount Google Drive:** Mounts Google Drive (`/content/drive`) to access datasets and save the fine-tuned model.

2.  **Configuration:**
    *   **Paths:**
        *   `data_path`: Path on Google Drive to the processed training data Parquet file (e.g., `/content/drive/MyDrive/ML_Reranker/data/processed_training_pairs_parts_0_to_12.parquet`).
        *   `output_model_dir`: Path on Google Drive to save the fine-tuned model (e.g., `/content/drive/MyDrive/ML_Reranker/models/reranker_finetuned_model`).
    *   **Model & Training Parameters:**
        *   `model_name`: Pre-trained cross-encoder (e.g., `'cross-encoder/ms-marco-MiniLM-L-6-v2'`).
        *   `num_train_epochs`: Number of training epochs (e.g., `1`).
        *   `train_batch_size`: Batch size for training (e.g., `16`).
        *   `max_length`: Maximum sequence length for tokenization.
        *   `default_activation_function`: Activation function for the CrossEncoder (e.g., `torch.nn.SiLU()`).
        *   `learning_rate`: Optimizer learning rate.
        *   `weight_decay`: Optimizer weight decay.
        *   `warmup_steps`: Learning rate scheduler warmup steps.
    *   **Data Handling:**
        *   `random_seed`: For reproducibility.
        *   `val_size`: Proportion of data for the validation set (e.g., `0.1`).

3.  **Load and Prepare Data:**
    *   **Load Dataset:** Loads training data from the Parquet file using pandas.
    *   **Create `InputExample`s:** Converts DataFrame rows (`user_ctx`, `book_text`, `label`) into `sentence_transformers.InputExample` objects.
    *   **Train/Validation Split:** Splits `InputExample`s into training and validation sets.

4.  **Model Initialization:**
    *   Initializes `CrossEncoder` with `model_name`, `num_labels=1` (for ranking score), `max_length`, and `default_activation_function`.

5.  **Model Training:**
    *   Creates a `DataLoader` for the training set.
    *   Calls `model.fit()` with training parameters: `train_dataloader`, `epochs`, `warmup_steps`, `output_path` (set to `output_model_dir`), `optimizer_params`, etc.
    *   An evaluator `CESoftmaxAccuracyEvaluator` used with `evaluation_steps` to monitor performance on the validation set during training.

6.  **Saving the Model:**
    *   The `model.fit()` method saves the best model (if an evaluator is used) or the final model to `output_path`.
    *   An explicit `model.save(output_model_dir)` can also be called.

7.  **Evaluation (Post-Training):**
    *   Loads the validation set examples (`val_examples`).
    *   Model predicts scores on the validation text pairs.
    *   Calculates and prints metrics. The primary evaluation block in the notebook, when run on a test set (`test_df`), performs the following:
        *   **Global Metrics (ROC AUC, AP):** `roc_auc_score` and `average_precision_score` are calculated across all test items.
        *   **Per-User Metrics (MAP, Mean NDCG@k):**
            *   For Mean Average Precision (MAP), `average_precision_score` is calculated for each user, and then the mean of these scores is taken.
            *   For Mean NDCG@k (e.g., k=3, 5, 10), `ndcg_score` from `sklearn.metrics` is calculated for each user's ranked list, and then the mean of these per-user NDCG scores is taken. This provides a more robust measure of ranking quality per user.

**Running the Colab Notebook:**
1.  Upload `reranker_model_tuning_colab.ipynb` to Google Colab.
2.  Ensure processed training data (e.g., `processed_training_pairs_parts_0_to_12.parquet`) is in the Google Drive path specified by `data_path`.
3.  Create the model output directory on Google Drive specified by `output_model_dir`.
4.  Run cells sequentially, authorizing Google Drive access.
5.  The fine-tuned model will be saved to Google Drive.

## 7. Using the Fine-tuned Model

This section would detail how to load the saved fine-tuned model from `output_model_dir` and use it for inference to rerank book recommendations for new user contexts. This typically involves:
1.  Loading the `CrossEncoder` model from the saved path.
2.  For a given user context and a list of candidate book texts, preparing pairs `(user_context, candidate_book_text_i)`.
3.  Using `model.predict()` on these pairs to get scores.
4.  Sorting the candidate books based on these scores in descending order to get the reranked list.
