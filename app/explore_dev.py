import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "merged_final_transformed.csv")
df = pd.read_csv(CSV_PATH)

print("=== MEDIAN AGE ===")
print(df["median_age"].describe().round(1).to_frame().T.to_string())
print()
print("Top 5 abnormal values (above 100):")
print(df[df["median_age"] > 100][["County name", "StateAbbr", "year", "median_age"]].head(5).to_string())

print()
print("=== PCT GRADUATE DEGREE ===")
print(df["pct_graduate_degree"].describe().round(1).to_frame().T.to_string())
print()
print("Top 5 abnormal values (above 50%):")
print(df[df["pct_graduate_degree"] > 50][["County name", "StateAbbr", "year", "pct_graduate_degree"]].head(5).to_string())
 