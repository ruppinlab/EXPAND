#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Mar 6, 2025
#### Compute NPIFs based on HoverNet prediction, using MPP = 0.248 for unit conversion
#### Removes outliers for Major Axis and Minor Axis using IQR
#### Filters top X% tiles based on majority of cancer nuclei
#### ------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--percentile", type=int, required=True, help="Top X percentile for filtering tiles")
args = parser.parse_args()

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.makedirs(_wpath_, exist_ok=True)
os.chdir(_wpath_)

print(f"Working directory: {_wpath_}")

# Define dataset paths
dataset_name = "TCGA_BRCA_FFPE"
input_folder = f"{dataset_name}/outputs/HoverNet/"
output_file_path = f"{dataset_name}/outputs/HoverNet/HoverNet_NPIFs_TCGA_BRCA_1106_Filtered_Top{args.percentile}Q.csv"

# Define computation settings
columns_to_compute = ["Area", "Major Axis", "Minor Axis", "Perimeter", "Eccentricity", "Circularity"]
MPP = 0.248  

def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 3 * IQR) & (df[column] <= Q3 + 3 * IQR)]

results = []
tcga_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f)) and f.startswith("TCGA")]

for slide_name in tcga_folders:
    file_path = os.path.join(input_folder, slide_name, "features", f"{slide_name}.csv")
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)
    df[["Area", "Major Axis", "Minor Axis", "Perimeter"]] *= MPP  

    df = remove_outliers(df, "Major Axis")
    df = remove_outliers(df, "Minor Axis")

    tile_nucleus_counts = df.groupby("Tile")["Nucleus ID"].count().reset_index()
    tile_nucleus_counts.rename(columns={"Nucleus ID": "Nucleus_Count"}, inplace=True)

    threshold_value = tile_nucleus_counts["Nucleus_Count"].quantile(args.percentile / 100)
    top_tiles = tile_nucleus_counts[tile_nucleus_counts["Nucleus_Count"] >= threshold_value]

    df_filtered = df[df["Tile"].isin(top_tiles["Tile"])]
    if df_filtered.empty:
        continue

    mean_values = df_filtered[columns_to_compute].mean()
    std_values = df_filtered[columns_to_compute].std()
    results.append([slide_name, len(tile_nucleus_counts), len(top_tiles)] + mean_values.tolist() + std_values.tolist())

result_df = pd.DataFrame(results, columns=["Slide_Name", "Total_Tiles", "Filtered_Tiles"] +
                          [f"Mean {col}" for col in columns_to_compute] + 
                          [f"Std {col}" for col in columns_to_compute])

result_df.to_csv(output_file_path, index=False)
print(f"Filtered results saved to: {output_file_path}")


# In[2]:


# result_df


# In[3]:


# result_df.describe()


# In[ ]:




