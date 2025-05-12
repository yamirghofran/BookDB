import shutil
import subprocess
import os

# Paths
prepared_data = "../data/processed_ncf/interactions_prepared_ncf_reduced.parquet"
target_dir = "neural-collaborative-filtering/data"
target_file = os.path.join(target_dir, "interactions.parquet")
train_script = "neural-collaborative-filtering/src/train.py"

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# Copy the file
shutil.copy(prepared_data, target_file)

# Run train.py
subprocess.run(["python", train_script], check=True)

export_script = "neural-collaborative-filtering/src/export_gmf_embeddings.py"
subprocess.run(["python", export_script], check=True)