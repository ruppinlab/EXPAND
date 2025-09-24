#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ---------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Mar 6, 2025
#### Mapped HoverNet NPIFs to TCGA_BRCA subtypes status for filtered tiles at top X percentile
#### ---------------------------------------------------------------------------------------------

import os
import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--percentile", type=int, required=True, help="Top X percentile for filtering tiles")
args = parser.parse_args()

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)

dataset_name = "TCGA_BRCA_FFPE"
hovernet_file = f"{dataset_name}/outputs/HoverNet/HoverNet_NPIFs_TCGA_BRCA_1106_Filtered_Top{args.percentile}Q.csv"
tcga_subtypes_file = "PA_HIF_BRCA/outputs_clinical/TCGA_BRCA_Subtypes_clinical.tsv"

# Load NPIFs data
hovernet_df = pd.read_csv(hovernet_file).drop(columns=["Total_Tiles", "Filtered_Tiles"])
hovernet_df["Slide_Name"] = hovernet_df["Slide_Name"].str[:12]
hovernet_df = hovernet_df.drop_duplicates(subset=['Slide_Name'])

# Load TCGA BRCA subtype data
tcga_df = pd.read_table(tcga_subtypes_file, sep="\t")[["sampleID", "HER2_Final_Status_nature2012", "PR_Status_nature2012", "ER_Status_nature2012"]]
tcga_df.columns = ["sampleID", "HER2_Status", "PR_Status", "ER_Status"]
tcga_df["sampleID"] = tcga_df["sampleID"].str.replace('-01', '', regex=False)

# Merge data
merged_df = pd.merge(tcga_df, hovernet_df, left_on='sampleID', right_on='Slide_Name', how="inner")
merged_df.drop(columns=["Slide_Name"], inplace=True)

# Save output
output_file = f"{dataset_name}/outputs/HoverNet/Subtypes/HoverNet_Original_NPIFs_Values_TCGA_BRCA_Mapped_BRCA_Status_Filtered_Tiles_Top{args.percentile}Q.csv"
merged_df.to_csv(output_file, index=False)

print(f"Mapped data saved to: {output_file}")


# In[ ]:




