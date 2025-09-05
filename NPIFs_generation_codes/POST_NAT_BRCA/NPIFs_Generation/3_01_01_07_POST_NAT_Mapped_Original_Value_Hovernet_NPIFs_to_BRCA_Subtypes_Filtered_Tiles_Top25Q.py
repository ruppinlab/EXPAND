#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: June 22, 2025
#### Mapped POST_NAT HoverNet NPIFs to BRCA subtypes status (using top 25  tiles)
#### Includes patient ID trace and unmatched checks (cleaned version)
#### ------------------------------------------------------------------------------------------

import os
import pandas as pd

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)
print(f"Working directory: {_wpath_}\n")

# Dataset name
dataset_name = "POST_NAT_BRCA"

# File paths
npif_file = f"{dataset_name}/HoverNet/outputs/POST_NAT_BRCA_HoverNet_NPIFs_Filtered_Tiles_Top25Q.csv"
slide_list_file = "/data/Ruppin_AI/Datasets/Post_NAT_BRCA/processed/Post_NAT_BRCA_slide_list.tsv"
clinical_metadata_file = "/data/Ruppin_AI/Datasets/Post_NAT_BRCA/processed/Post_NAT_BRCA_clinical_metadata_short.tsv"

# Load NPIFs and extract Slide_ID
npif_df = pd.read_csv(npif_file)
npif_df["Slide_ID"] = npif_df["Slide_ID"].astype(int)

# Load slide-to-patient mapping and clinical metadata
slide_list_df = pd.read_csv(slide_list_file, sep="\t")
clinical_metadata_df = pd.read_csv(clinical_metadata_file, sep="\t")
slide_list_df
clinical_metadata_df


# In[2]:


# Merge Slide_ID to get Patient_ID
npif_mapped_df = pd.merge(npif_df, slide_list_df[["Patient_ID", "Slide_ID"]], on="Slide_ID", how="left")
npif_mapped_df = npif_mapped_df.dropna(subset=["Patient_ID"])
npif_mapped_df["Patient_ID"] = npif_mapped_df["Patient_ID"].astype(str)

# Drop any pre-existing clinical columns to avoid duplication
columns_to_drop = ["HER2_Status", "ER_Status", "PR_Status", "Clinical_subtype"]
npif_mapped_df = npif_mapped_df.drop(columns=[col for col in columns_to_drop if col in npif_mapped_df.columns])

# Prepare and rename clinical subtype columns
clinical_metadata_df["Patient_ID"] = clinical_metadata_df["Patient_ID"].astype(str)
subtypes_df = clinical_metadata_df.rename(columns={
    "HER2_status": "HER2_Status",
    "ER_status": "ER_Status",
    "PR_status": "PR_Status"
})[["Patient_ID", "HER2_Status", "ER_Status", "PR_Status", "Clinical_subtype"]]

# Map binary values to 'Positive'/'Negative'
binary_map = {1.0: "Positive", 0.0: "Negative"}
subtypes_df["HER2_Status"] = subtypes_df["HER2_Status"].map(binary_map)
subtypes_df["ER_Status"] = subtypes_df["ER_Status"].map(binary_map)
subtypes_df["PR_Status"] = subtypes_df["PR_Status"].map(binary_map)

# Merge subtype info into npif_mapped_df
merged_df = pd.merge(subtypes_df, npif_mapped_df, on="Patient_ID", how="inner")

# Reorder to make Patient_ID the first column
cols = merged_df.columns.tolist()
cols.insert(0, cols.pop(cols.index("Patient_ID")))
merged_df = merged_df[cols]

# Print unique Slide_Name and Patient_ID values
print("Patient_IDs from POST_NAT_BRCA_HoverNet_NPIFs (Total: {}):".format(len(merged_df)))
print(merged_df["Patient_ID"].unique())

print("\nPatient IDs from POST_NAT clinical metadata (Total: {}):".format(len(subtypes_df)))
print(subtypes_df["Patient_ID"].unique())

# Find non-matching values
slide_ids = set(merged_df["Patient_ID"])
patient_ids = set(subtypes_df["Patient_ID"])

non_matching_slides = slide_ids - patient_ids
non_matching_patients = patient_ids - slide_ids

print("\nHoverNet NPIFs that do NOT have a matching Patient_ID:")
print(non_matching_slides)

print("\nClinical metadata entries that do NOT have a matching Slide NPIF:")
print(non_matching_patients)



# In[3]:


# Save merged (slide-level) result
output_dir = f"{dataset_name}/outputs/HoverNet/Subtypes/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "HoverNet_Original_NPIFs_Values_POST_NAT_BRCA_Mapped_BRCA_Status_Filtered_Tiles_Top25Q.csv")
# --------------------------------------
# Patient-level aggregation of NPIFs
# --------------------------------------

# Identify NPIF feature columns to average
mean_std_cols = [col for col in merged_df.columns if col.startswith("Mean ") or col.startswith("Std ")]

# Aggregate NPIFs by Patient_ID (mean), keep first for clinical/status columns
patient_level_df = merged_df.groupby("Patient_ID").agg({
    "HER2_Status": "first",
    "ER_Status": "first",
    "PR_Status": "first",
    "Clinical_subtype": "first",
    **{col: "mean" for col in mean_std_cols}
}).reset_index()

# Save patient-level aggregated results to the same file (overwrite)
patient_level_df.to_csv(output_file, index=False)
print(f"\nPatient-level averaged data saved to: {output_file}")
patient_level_df


# In[ ]:




