#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Feb 9, 2025
#### Compute 12NPIFs based on HoverNet prediction, using MPP = 0.248 for unit conversion
#### Removes outliers for Major Axis and Minor Axis using IQR
#### Filters top 25% tiles based on majority of cancer nuclei
#### ------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.makedirs(_wpath_, exist_ok=True)
os.chdir(_wpath_)

print("Working directory:", _wpath_)

# Define dataset name and output file path dynamically
dataset_name = "TCGA_BRCA_FFPE"
input_folder = f"{dataset_name}/outputs/HoverNet/"
output_file_path = f"{dataset_name}/outputs/HoverNet/HoverNet_NPIFs_TCGA_BRCA_1106_Filter_Top_25Q.csv"

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

# Get all folders matching TCGA pattern
tcga_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f)) and f.startswith("TCGA")]

# Process each detected TCGA folder
for slide_name in tcga_folders:
    file_path = os.path.join(input_folder, slide_name, "features", f"{slide_name}.csv")
    
    # Check if file exists before proceeding
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping...")
        continue

    # Load CSV file
    df = pd.read_csv(file_path)

    # Convert units using MPP where applicable
    df["Area"] = df["Area"] * (MPP ** 2)  # Convert area to µm²
    df["Major Axis"] = df["Major Axis"] * MPP  # Convert major axis length to µm
    df["Minor Axis"] = df["Minor Axis"] * MPP  # Convert minor axis length to µm
    df["Perimeter"] = df["Perimeter"] * MPP  # Convert perimeter to µm

    # Remove outliers for Major Axis and Minor Axis
    df = remove_outliers(df, "Major Axis")
    df = remove_outliers(df, "Minor Axis")

    # Count total nuclei per tile
    tile_nucleus_counts = df.groupby("Tile")["Nucleus ID"].count().reset_index()
    tile_nucleus_counts.rename(columns={"Nucleus ID": "Nucleus_Count"}, inplace=True)

    # Get total number of tiles with cancer nuclei
    total_tiles_with_nuclei = tile_nucleus_counts.shape[0]

    # Compute the 75th percentile (top 25%) nucleus count for filtering
    percentile_75 = tile_nucleus_counts["Nucleus_Count"].quantile(0.75)

    # Filter top 25% tiles with highest nucleus count
    top_25_percent_tiles = tile_nucleus_counts[tile_nucleus_counts["Nucleus_Count"] >= percentile_75]

    # Get the number of tiles that passed the filtering
    num_filtered_tiles = top_25_percent_tiles.shape[0]

    print(f"Slide: {slide_name}")
    print(f"  - Total tiles with cancer nuclei: {total_tiles_with_nuclei}")
    print(f"  - 75th percentile nucleus count threshold: {percentile_75}")
    print(f"  - Number of tiles selected (top 25%): {num_filtered_tiles}")

    # Merge filtered tiles with original dataframe
    df_filtered = df[df["Tile"].isin(top_25_percent_tiles["Tile"])]

    if df_filtered.empty:
        print(f"  - No tiles with nucleus count in the top 25% for {slide_name}, skipping...\n")
        continue

    # Compute mean and std for required columns
    mean_values = df_filtered[columns_to_compute].mean()
    std_values = df_filtered[columns_to_compute].std()

    # Append results
    results.append([slide_name, total_tiles_with_nuclei, num_filtered_tiles] + mean_values.tolist() + std_values.tolist())

# Create DataFrame for results
result_df = pd.DataFrame(
    results,
    columns=["Slide_Name", "Total_Tiles", "Filtered_Tiles"] + 
            [f"Mean {col}" for col in columns_to_compute] + 
            [f"Std {col}" for col in columns_to_compute]
)

# Handle NaN and infinite values: Replace them with column mean
result_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert inf to NaN
result_df.fillna(result_df.mean(numeric_only=True), inplace=True)  # Replace NaN with column mean

# If a column is entirely NaN, replace with zero
result_df.fillna(0, inplace=True)

# Save results to CSV
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Ensure directory exists
result_df.to_csv(output_file_path, index=False)

print(f"\nFiltered results saved to: {output_file_path}")


# In[2]:


result_df


# In[ ]:




