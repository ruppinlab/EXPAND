#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Apr 8, 2025
#### Compute 12NPIFs based on HoverNet prediction, using MPP = 0.248 for unit conversion
#### Removes outliers for Major Axis and Minor Axis using IQR
#### Computes across **all tiles** with tumor nuclei
#### ------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.makedirs(_wpath_, exist_ok=True)
os.chdir(_wpath_)
print("Working directory:", _wpath_)

# Define dataset name and output file path
dataset_name = "TCGA_BRCA_FFPE"
input_folder = f"{dataset_name}/outputs/HoverNet/"
output_file_path = f"{dataset_name}/outputs/HoverNet/HoverNet_NPIFs_TCGA_BRCA_1106_AllTiles.csv"

# Define column names for NPIF computation
columns_to_compute = ["Area", "Major Axis", "Minor Axis", "Perimeter", "Eccentricity", "Circularity"]

# Microns-per-pixel for unit conversion
MPP = 0.248  

# IQR-based outlier removal
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Store results
results = []

# Get all TCGA folders
tcga_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f)) and f.startswith("TCGA")]

# Process each slide
for slide_name in tcga_folders:
    file_path = os.path.join(input_folder, slide_name, "features", f"{slide_name}.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping...")
        continue

    df = pd.read_csv(file_path)

    # Unit conversion
    df["Area"] = df["Area"] * (MPP ** 2)  
    df["Major Axis"] = df["Major Axis"] * MPP  
    df["Minor Axis"] = df["Minor Axis"] * MPP  
    df["Perimeter"] = df["Perimeter"] * MPP  

    # Remove outliers
    df = remove_outliers(df, "Major Axis")
    df = remove_outliers(df, "Minor Axis")

    # Count number of unique tiles with at least one nucleus
    total_tiles_with_nuclei = df["Tile"].nunique()

    if df.empty:
        print(f"  - No valid nuclei found for {slide_name}, skipping...\n")
        continue

    # Compute statistics across all tiles
    mean_values = df[columns_to_compute].mean()
    std_values = df[columns_to_compute].std()

    # Append result
    results.append([slide_name, total_tiles_with_nuclei] + mean_values.tolist() + std_values.tolist())

# Create final dataframe
result_df = pd.DataFrame(
    results,
    columns=["Slide_Name", "Total_Tiles"] + 
            [f"Mean {col}" for col in columns_to_compute] + 
            [f"Std {col}" for col in columns_to_compute]
)

# Clean NaNs and infs
result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
result_df.fillna(result_df.mean(numeric_only=True), inplace=True)
result_df.fillna(0, inplace=True)

# Save to CSV
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
result_df.to_csv(output_file_path, index=False)

print(f"\nNPIFs computed from all tiles saved to: {output_file_path}")


# In[2]:


result_df


# In[ ]:




