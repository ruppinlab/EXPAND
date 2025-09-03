#!/usr/bin/env python
# coding: utf-8

# In[4]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Feb 18, 2025
#### Mapped HoverNet NPIFs to TCGA_BRCA subtypes status (all subtype status) 
#### --------------------------------------------------------------------------------------------

import os
import pandas as pd

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)

print(f"Working directory: {_wpath_}\n")

# Define dataset and file paths
dataset_name = "TCGA_BRCA_FFPE"

# File paths
hovernet_Predicted_NPIFs_TCGA_BRCA_file = f"{dataset_name}/outputs/HoverNet/HoverNet_NPIFs_TCGA_BRCA_1106.csv"
tcga_subtypes_file = "PA_HIF_BRCA/outputs_clinical/TCGA_BRCA_Subtypes_clinical.tsv"

# Load the original NPIFs values data
hovernet_Predicted_NPIFs_TCGA_BRCA = pd.read_csv(hovernet_Predicted_NPIFs_TCGA_BRCA_file)

# Remove trailing spaces from column names
hovernet_Predicted_NPIFs_TCGA_BRCA.columns = hovernet_Predicted_NPIFs_TCGA_BRCA.columns.str.strip()

# Extract the first 12 characters of Slide_Name to create a new Sample_ID column
hovernet_Predicted_NPIFs_TCGA_BRCA["Slide_Name"] = hovernet_Predicted_NPIFs_TCGA_BRCA["Slide_Name"].str[:12]

# Drop duplicate Slide_Name in hovernet_Predicted_NPIFs_TCGA_BRCA (keep the first occurrence)
hovernet_Predicted_NPIFs_TCGA_BRCA = hovernet_Predicted_NPIFs_TCGA_BRCA.drop_duplicates(subset=['Slide_Name'], keep='first')


# Load the TCGA BRCA subtype data
tcga_subtypes_data = pd.read_table(tcga_subtypes_file, sep="\t")

# Keep only the relevant columns from TCGA subtypes
tcga_subtypes_data = tcga_subtypes_data[["sampleID", "HER2_Final_Status_nature2012", "PR_Status_nature2012", "ER_Status_nature2012"]]

# Rename columns to match the expected format
tcga_subtypes_data = tcga_subtypes_data.rename(columns={
    "HER2_Final_Status_nature2012": "HER2_Status",
    "PR_Status_nature2012": "PR_Status",
    "ER_Status_nature2012": "ER_Status"
})

# Remove '-01' suffix from Sample_ID to match NPIFs data
tcga_subtypes_data["sampleID"] = tcga_subtypes_data["sampleID"].str.replace('-01', '', regex=False)

hovernet_Predicted_NPIFs_TCGA_BRCA
# tcga_subtypes_data


# In[5]:


# Merge NPIFs with TCGA subtypes data on Sample_ID
merged_df = pd.merge(tcga_subtypes_data, hovernet_Predicted_NPIFs_TCGA_BRCA, left_on='sampleID', right_on='Slide_Name', how="inner")

# Drop the redundant Slide_Name column
merged_df.drop(columns=["Slide_Name"], inplace=True)

# Define output file path
output_path = f"{dataset_name}/outputs/HoverNet/Subtypes/"

# Ensure the directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

# file_name for output file
file_name = "HoverNet_Original_NPIFs_Values_TCGA_BRCA_Mapped_BRCA_Status.csv"

output_file = os.path.join(output_path, file_name)

output_file


# Save merged data
merged_df.to_csv(output_file, index=False)

print(f"Mapped data saved to: {output_file}")
print("Done!")

merged_df


# In[ ]:




