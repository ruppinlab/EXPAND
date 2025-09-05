#!/usr/bin/env python
# coding: utf-8

# In[32]:


#### -----------------------------------------------------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Jul 11, 2024
#### Mapped TCGA_BRCA advance sub-types data to PathAI HIF data
#### ------------------------------------------------------------------------------------------------------------------------------------------
import os, sys, pickle, bz2
import numpy as np, pandas as pd
from time import time 
from tqdm import tqdm

# Set working directory as the parent directory where all datasets are saved
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)
print(f"working directory = {_wpath_}\n")

#%% get TCGA_BRCA subtypes & input HIF features data.

# Define dataset folder name
dataset_name = "PA_HIF_BRCA"

# Define input paths for sample list, HIF data, and subtype data
data_path = "/data/Ruppin_AI/BRCA_PIF/data/HIFs_Original/"

data_files = ["brca_hifs.csv",        ## Contains PathAI HIFs original feature data with other extra columns      
              "TCGA_BRCA_Subtypes_Class_Clinical.csv"] # Contains subtype labels for BRCA

# Create directory to save outputs if it doesn't exist
outputs_path = f"{dataset_name}/clinical_tcga_brca_outputs/"
os.makedirs(outputs_path, exist_ok=True)

# Read PathAI HIFs data
hif_df = pd.read_csv(os.path.join(data_path, data_files[0]))


# Read TCGA BRCA subtype classification data
subtype_df = pd.read_csv(os.path.join(data_path, data_files[1]))


hif_df
# subtype_df



# In[33]:


# Select 'bcr_patient_barcode' and columns 2 to 609 (inclusive)
selected_columns = ['bcr_patient_barcode'] + hif_df.columns[2:609].tolist()
hif_df_filtered = hif_df[selected_columns]
hif_df_filtered


# In[34]:


# Rename 'bcr_patient_barcode' to 'sample_id'
hif_df_filtered = hif_df_filtered.rename(columns={'bcr_patient_barcode': 'sample_id'})
hif_df_filtered

# Step 1: Average HIF rows if multiple rows per sample_id exist
hif_df_filtered_avg = hif_df_filtered.groupby('sample_id', as_index=False).mean()
hif_df_filtered_avg


# In[35]:


# Create 'sample_id' by removing '-01' suffix to align with HIF sample IDs
subtype_df['sampleID'] = subtype_df['sampleID'].str.replace('-01', '', regex=False)
subtype_df

# Rename multiple columns for consistency
subtype_df = subtype_df.rename(columns={
    'sampleID': 'sample_id',
    'HER2_Final_Status_nature2012': 'HER2_Status',
    'PR_Status_nature2012': 'PR_Status',
    'ER_Status_nature2012': 'ER_Status'
})

subtype_df


# In[36]:


# Merge on 'sample_id' (inner join to retain matched samples only)
merged_df = pd.merge(subtype_df, hif_df_filtered_avg, on='sample_id', how='inner')
merged_df


# In[37]:


# Define path to save the merged output
tcga_brca_subtype_to_original_hifs = f"{outputs_path}tcga_brca_subtype_to_original_hifs.csv"

# Save the final merged dataframe to CSV
merged_df.to_csv(tcga_brca_subtype_to_original_hifs, index=False)

# Confirm completion
print("The files have been mapped and saved to:", tcga_brca_subtype_to_original_hifs)
print("Done!")


# In[ ]:




