#!/usr/bin/env python
# coding: utf-8

# In[11]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: June 22, 2025
#### Compute 12NPIFs based on HoverNet prediction, using MPP = 0.248 for unit conversion
#### Removes outliers for Major Axis and Minor Axis using IQR
#### Filters top 25% tiles based on majority of cancer nuclei
#### Dataset: POST_NAT_BRCA
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
dataset_name = "POST_NAT_BRCA"
input_folder = f"{dataset_name}/HoverNet/outputs/"
output_file_path = f"{dataset_name}/HoverNet/outputs/POST_NAT_BRCA_HoverNet_NPIFs_Filtered_Tiles_Top25Q.csv"

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

# Get all folders matching POST_NAT pattern
slide_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f)) and f.endswith("_tiles")]

# Process each detected slide
for slide_name in slide_folders:
    file_path = os.path.join(input_folder, slide_name, "features", f"{slide_name}.csv")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping...")
        continue

    df = pd.read_csv(file_path)

    # Convert units using MPP
    df["Area"] = df["Area"] * (MPP ** 2)
    df["Major Axis"] = df["Major Axis"] * MPP
    df["Minor Axis"] = df["Minor Axis"] * MPP
    df["Perimeter"] = df["Perimeter"] * MPP

    # Remove outliers for Major Axis and Minor Axis
    df = remove_outliers(df, "Major Axis")
    df = remove_outliers(df, "Minor Axis")

    # Count total nuclei per tile
    tile_nucleus_counts = df.groupby("Tile")["Nucleus ID"].count().reset_index()
    tile_nucleus_counts.rename(columns={"Nucleus ID": "Nucleus_Count"}, inplace=True)

    # Get total number of tiles with cancer nuclei
    total_tiles_with_nuclei = tile_nucleus_counts.shape[0]

    # Compute 75th percentile threshold
    percentile_75 = tile_nucleus_counts["Nucleus_Count"].quantile(0.75)

    # Filter top 25% tiles
    top_25_percent_tiles = tile_nucleus_counts[tile_nucleus_counts["Nucleus_Count"] >= percentile_75]
    num_filtered_tiles = top_25_percent_tiles.shape[0]

    print(f"Slide: {slide_name}")
    print(f"  - Total tiles with cancer nuclei: {total_tiles_with_nuclei}")
    print(f"  - 75th percentile nucleus count threshold: {percentile_75}")
    print(f"  - Number of tiles selected (top 25%): {num_filtered_tiles}")

    # Filter dataframe
    df_filtered = df[df["Tile"].isin(top_25_percent_tiles["Tile"])]
    if df_filtered.empty:
        print(f"  - No tiles with nucleus count in top 25% for {slide_name}, skipping...\n")
        continue

    # Compute statistics
    mean_values = df_filtered[columns_to_compute].mean()
    std_values = df_filtered[columns_to_compute].std()

    # Append result row
    results.append([slide_name, total_tiles_with_nuclei, num_filtered_tiles] + mean_values.tolist() + std_values.tolist())

# Convert result list to DataFrame
result_df = pd.DataFrame(
    results,
    columns=["Slide_Name", "Total_Tiles", "Filtered_Tiles"] + 
            [f"Mean {col}" for col in columns_to_compute] + 
            [f"Std {col}" for col in columns_to_compute]
)

# Handle NaN/infs
result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
result_df.fillna(result_df.mean(numeric_only=True), inplace=True)
result_df.fillna(0, inplace=True)

result_df

# Save slide-level results
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
result_df.to_csv(output_file_path, index=False)
print(f"\nFiltered results saved to: {output_file_path}")


# In[12]:


result_df


# In[13]:


# Extract Slide_ID as the part before the first underscore in Slide_Name
result_df['Slide_ID'] = result_df['Slide_Name'].str.split('_').str[0]
result_df

# Aggregate patient-level features
patient_avg_df = result_df.groupby('Slide_ID').mean(numeric_only=True)

patient_avg_df

# Save patient-level results
patient_avg_df.to_csv(output_file_path)
print(f"Converted results saved to: {output_file_path}")

patient_avg_df


# In[14]:


patient_avg_df.describe()


# In[15]:


# Plotting
import matplotlib.pyplot as plt

# Select only NPIF features
bcnp_features = patient_avg_df.iloc[:, 2:]
n_bcnp = bcnp_features.shape[0]

# Boxplot
plt.figure(figsize=(30, 10))
plt.boxplot(bcnp_features.values, labels=bcnp_features.columns, vert=True, patch_artist=True)
plt.xticks(rotation=90)
plt.title(f"BCNB NPIFs Original Values (n = {n_bcnp})")
plt.ylabel("Feature Values")
plt.grid(True)
plt.show()


# In[ ]:




