import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the CSV files
RES_DIR = os.path.join(os.path.dirname(__file__), 'res')

# Map file names to model names for legend clarity
model_files = {
    'gmf_metrics_history.csv': 'GMF',
    'mlp_metrics_history.csv': 'MLP',
    'neumf_metrics_history.csv': 'NeuMF',
}

plt.figure(figsize=(10, 6))

# Plot hit_ratio
for csv_file, model_name in model_files.items():
    path = os.path.join(RES_DIR, csv_file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.head(20)
        df['epoch'] = range(1, len(df) + 1)
        plt.plot(df['epoch'], df['hit_ratio'], marker='o', label=f'{model_name} Hit Ratio')

plt.xlabel('Epoch')
plt.ylabel('Hit Ratio')
plt.title('Model Comparison: Hit Ratio over Epochs')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 21))  # Show only whole numbers 1 to 20
plt.tight_layout()
plt.savefig('hit_ratio_comparison.png')
plt.show()

plt.figure(figsize=(10, 6))
# Plot ndcg
for csv_file, model_name in model_files.items():
    path = os.path.join(RES_DIR, csv_file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.head(20)
        df['epoch'] = range(1, len(df) + 1)
        plt.plot(df['epoch'], df['ndcg'], marker='o', label=f'{model_name} NDCG')

plt.xlabel('Epoch')
plt.ylabel('NDCG')
plt.title('Model Comparison: NDCG over Epochs')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 21))  # Show only whole numbers 1 to 20
plt.tight_layout()
plt.savefig('ndcg_comparison.png')
plt.show()
