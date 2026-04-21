import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "merged_final_transformed.csv")

df = pd.read_csv(CSV_PATH)

# Shape of the dataset
print("Shape:", df.shape)

# All column names
print("\nColumns:")
for col in df.columns:
    print(" -", col)

# Sample of the data
print("\nFirst 2 rows:")
print(df.head(2))

# Check key columns exist
print("\nUnique years:", sorted(df['year'].unique().tolist()))
print("Sample counties:", df['County name'].head(5).tolist())