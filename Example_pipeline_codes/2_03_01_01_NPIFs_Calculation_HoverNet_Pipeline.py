#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Feb 9, 2025
#### Compute 12NPIFs based on HoverNet prediction, using MPP = 0.248 for unit conversion
#### Removes outliers for Major Axis and Minor Axis using IQR
#### Filters top 25% tiles based on majority of cancer nuclei
#### -------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.makedirs(_wpath_, exist_ok=True)
os.chdir(_wpath_)

print("Working directory:", _wpath_)

# Define dataset name
dataset_name = "TCGA_BRCA_FFPE"

# NEW: accept Morph root as input and NPIFs CSV as output
parser = ArgumentParser(description="Compute 12 NPIFs (Top-25% tiles) from Morph outputs")
parser.add_argument(
    "--morph_root",
    type=str,
    default=os.path.join(_wpath_, dataset_name, "outputs", "Morph") + "/",
    help="Root folder containing per-slide morphology CSVs: Morph/<SLIDE>/tumor_nuclei.csv"
)
parser.add_argument(
    "--output_csv",
    type=str,
    default=os.path.join(_wpath_, dataset_name, "outputs", "NPIFs", "TCGA_BRCA_HoverNet_NPIFs_25Q.csv"),
    help="Output CSV path (Top-25% NPIFs)"
)
args = parser.parse_args()

# UPDATED: use Morph as input and NPIFs as output
input_folder     = args.morph_root
output_file_path = args.output_csv

# Define column names for computation
columns_to_compute = ["Area", "Major Axis", "Minor Axis", "Perimeter", "Eccentricity", "Circularity"]

# Define MPP for conversion
MPP = 0.248

# Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Store results in a list
results = []

# Get all folders matching TCGA pattern under Morph root
tcga_folders = [
    f for f in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, f)) and f.startswith("TCGA")
]

# Process each detected TCGA folder
for slide_name in tcga_folders:
    # UPDATED: file path now points to Morph/<SLIDE>/tumor_nuclei.csv
    file_path = os.path.join(input_folder, slide_name, "tumor_nuclei.csv")

    # Check if file exists before proceeding
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping...")
        continue

    # Load CSV file
    df = pd.read_csv(file_path)

    # Unit conversion using MPP
    df["Area"]        = df["Area"] * (MPP ** 2)  # µm²
    df["Major Axis"]  = df["Major Axis"] * MPP   # µm
    df["Minor Axis"]  = df["Minor Axis"] * MPP   # µm
    df["Perimeter"]   = df["Perimeter"] * MPP    # µm

    # Remove outliers for Major/Minor Axis
    df = remove_outliers(df, "Major Axis")
    df = remove_outliers(df, "Minor Axis")

    # Count total nuclei per tile
    tile_nucleus_counts = df.groupby("Tile")["Nucleus ID"].count().reset_index()
    tile_nucleus_counts.rename(columns={"Nucleus ID": "Nucleus_Count"}, inplace=True)

    total_tiles_with_nuclei = tile_nucleus_counts.shape[0]
    percentile_75 = tile_nucleus_counts["Nucleus_Count"].quantile(0.75)
    top_25_percent_tiles = tile_nucleus_counts[tile_nucleus_counts["Nucleus_Count"] >= percentile_75]
    num_filtered_tiles = top_25_percent_tiles.shape[0]

    print(f"Slide: {slide_name}")
    print(f"  - Total tiles with cancer nuclei: {total_tiles_with_nuclei}")
    print(f"  - 75th percentile nucleus count threshold: {percentile_75}")
    print(f"  - Number of tiles selected (top 25%): {num_filtered_tiles}")

    # Filter original dataframe to top 25% tiles
    df_filtered = df[df["Tile"].isin(top_25_percent_tiles["Tile"])]

    if df_filtered.empty:
        print(f"  - No tiles with nucleus count in the top 25% for {slide_name}, skipping...\n")
        continue

    # Compute mean and std for required columns
    mean_values = df_filtered[columns_to_compute].mean()
    std_values  = df_filtered[columns_to_compute].std()

    # Append results
    results.append(
        [slide_name, total_tiles_with_nuclei, num_filtered_tiles] +
        mean_values.tolist() + std_values.tolist()
    )

# Create DataFrame for results
result_df = pd.DataFrame(
    results,
    columns=["Slide_Name", "Total_Tiles", "Filtered_Tiles"] +
            [f"Mean {col}" for col in columns_to_compute] +
            [f"Std {col}" for col in columns_to_compute]
)

# Handle NaN and infinite values: Replace them with column mean
result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
result_df.fillna(result_df.mean(numeric_only=True), inplace=True)
result_df.fillna(0, inplace=True)  # fallback for entirely-NaN columns

# Save results to CSV (ensure NPIFs dir exists)
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
result_df.to_csv(output_file_path, index=False)

print(f"\nFiltered results saved to: {output_file_path}")


# In[2]:


result_df


# In[ ]:




