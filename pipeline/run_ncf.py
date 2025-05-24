import shutil
import subprocess
import os

# Create results directory if it doesn't exist
os.makedirs("results/ncf", exist_ok=True)

train_script = "neural-collaborative-filtering/src/train.py"

# Run train.py
subprocess.run(["python", train_script], check=True)

export_script = "neural-collaborative-filtering/src/export_gmf_embeddings.py"
subprocess.run(["python", export_script], check=True)

print("NCF pipeline completed. Results saved to results/ncf directory.")