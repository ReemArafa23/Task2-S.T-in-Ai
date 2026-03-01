import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Load public dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Split data
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
print("Preprocessing complete. Train and test sets saved.")