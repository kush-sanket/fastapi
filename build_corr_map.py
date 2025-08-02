
import pandas as pd
import json

# --- Paths ---
INPUT_CSV  = "list_attr_celeba.csv"   # your CSV file
OUTPUT_JSON = "corr_map.json"

# 1) Read the CSV normally
df = pd.read_csv(INPUT_CSV)

# 2) Drop any non‐attribute column (e.g. image filename column)
if "image_id" in df.columns:
    df = df.drop("image_id", axis=1)
elif "ImageId" in df.columns:
    df = df.drop("ImageId", axis=1)

# 3) If your labels are still -1/+1, convert them to 0/1
#    (skip this step if they’re already 0/1)
df = df.replace({-1: 0, 1: 1})

# 4) Make sure 'Attractive' is in the columns
if "Attractive" not in df.columns:
    raise KeyError(f"'Attractive' column not found—got: {df.columns.tolist()}")

# 5) Compute Pearson correlations versus 'Attractive'
corr_series = df.corrwith(df["Attractive"]).drop("Attractive")
corr_map = corr_series.round(3).to_dict()

# 6) Save out to JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(corr_map, f, indent=2)

print(f"Saved correlation map with {len(corr_map)} entries → {OUTPUT_JSON}")
