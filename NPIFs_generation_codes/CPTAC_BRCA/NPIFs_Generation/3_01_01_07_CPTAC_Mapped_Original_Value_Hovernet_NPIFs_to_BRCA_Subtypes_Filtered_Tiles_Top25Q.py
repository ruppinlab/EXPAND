#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Mar 6, 2025
#### Mapped CPTAC HoverNet NPIFs to TCGA_BRCA subtypes status
#### --------------------------------------------------------------------------------------------

import os
import pandas as pd

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)

print(f"Working directory: {_wpath_}\n")

# Define dataset name and output file path dynamically
dataset_name = "CPTAC_BRCA"

# File paths
CPTAC_BRCA_HoverNet_NPIFs_file = f"{dataset_name}/HoverNet/outputs/CPTAC_BRCA_HoverNet_NPIFs_Filtered_Tiles_Top25Q.csv"
CPTAC_subtypes_file = "/data/Lab_ruppin/dhrubas2/HnE/CPTAC_BRCA/processed/CPTAC_BRCA_clinical_data_summary_matched.tsv"

# Load the original NPIFs values data
CPTAC_BRCA_HoverNet_NPIFs = pd.read_csv(CPTAC_BRCA_HoverNet_NPIFs_file)

# Drop the unnecessary columns
CPTAC_BRCA_HoverNet_NPIFs.drop(columns=["Total_Tiles", "Filtered_Tiles"], inplace=True)

# Remove trailing spaces from column names
CPTAC_BRCA_HoverNet_NPIFs.columns = CPTAC_BRCA_HoverNet_NPIFs.columns.str.strip()


# Load the TCGA BRCA subtype data
CPTAC_subtypes_data = pd.read_table(CPTAC_subtypes_file, sep="\t")

# Keep only the relevant columns from TCGA subtypes
CPTAC_subtypes_data = CPTAC_subtypes_data[["Patient_ID", "HER2_status", "PR_status", "ER_status"]]

# Rename columns to match the expected format
CPTAC_subtypes_data = CPTAC_subtypes_data.rename(columns={
    "HER2_status": "HER2_Status",
    "PR_status": "PR_Status",
    "ER_status": "ER_Status"
})


# Convert Patient_ID to string for proper merging
CPTAC_subtypes_data["Patient_ID"] = CPTAC_subtypes_data["Patient_ID"].astype(str)

# Zero-pad Patient_ID to ensure it has three digits
CPTAC_subtypes_data["Patient_ID"] = CPTAC_subtypes_data["Patient_ID"].astype(str).str.zfill(3)


CPTAC_BRCA_HoverNet_NPIFs
CPTAC_subtypes_data


# In[2]:


# Print unique Slide_Name and Patient_ID values
print("Patient_IDs from CPTAC_BRCA_HoverNet_NPIFs (Total: {}):".format(len(CPTAC_BRCA_HoverNet_NPIFs)))
print(CPTAC_BRCA_HoverNet_NPIFs["Patient_ID"].unique())

print("\nPatient IDs from CPTAC_subtypes_data (Total: {}):".format(len(CPTAC_subtypes_data)))
print(CPTAC_subtypes_data["Patient_ID"].unique())

# Find non-matching values
slide_names = set(CPTAC_BRCA_HoverNet_NPIFs["Patient_ID"])
patient_ids = set(CPTAC_subtypes_data["Patient_ID"])

# Find slides that are not matching with Patient_IDs
non_matching_slides = slide_names - patient_ids
non_matching_patients = patient_ids - slide_names

print("\nHoverNet NPIFs that do NOT have a matching Patient_ID:")
print(non_matching_slides)

print("\n BRCA_Status that do NOT have a matching Patient_ID:")
print(non_matching_patients)


# In[3]:


# Merge NPIFs with TCGA subtypes data on Sample_ID
merged_df = pd.merge(CPTAC_subtypes_data, CPTAC_BRCA_HoverNet_NPIFs, on='Patient_ID', how="inner")


# Define output file path
output_path = f"{dataset_name}/outputs/HoverNet/Subtypes/"

# Ensure the directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

# file_name for output file
file_name = "HoverNet_Original_NPIFs_Values_CPTAC_BRCA_Mapped_BRCA_Status_Filtered_Tiles_Top25Q.csv"

output_file = os.path.join(output_path, file_name)

output_file


# Save merged data
merged_df.to_csv(output_file, index=False)

print(f"Mapped data saved to: {output_file}")
print("Done!")

merged_df


# In[ ]:




