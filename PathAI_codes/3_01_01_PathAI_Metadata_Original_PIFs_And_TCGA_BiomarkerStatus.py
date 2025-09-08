#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### -----------------------------------------------------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Aug 27, 2024
#### Mapped PathAI metadata to nuHIFs data to TCGA biomarker status
#### ------------------------------------------------------------------------------------------------------------------------------------------
import os, sys, pickle, bz2
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from time import time 
from tqdm import tqdm

_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"        # set working directory as the parent directory where all datasets are saved
os.chdir(_wpath_)

print(f"working directory = {_wpath_}\n")


# In[2]:


#%% get TCGA_BRCA subtypes & input HIF features data.


data_files   = ["PA_NUHIF_BRCA/outputs_biomarker_status/tcga_brca_subtype_to_original_nuhifs.csv",
                "PA_HIF_BRCA/clinical_tcga_brca_outputs/tcga_brca_subtype_to_original_hifs.csv"]

# Data directories & files
dataset_name = "PA_PIF_BRCA"
feature_name = "PIF"
outcome_names = ["HER2_Status", "PR_Status", "ER_Status"]

## create directories to save outputs
outputs_path = f"{dataset_name}/outputs_biomarker_status/"
os.makedirs(outputs_path, exist_ok = True)

#read metadata file of PathAI 
PathAI_hifs_with_status = pd.read_csv( data_files[0])

#read nuHIF file of PathAI 
PathAI_nuhifs_with_status = pd.read_csv(data_files[1])

PathAI_hifs_with_status
PathAI_nuhifs_with_status


# In[3]:


# Drop the status columns from PathAI_hifs_with_status
PathAI_hifs_with_status = PathAI_hifs_with_status.drop(columns=['HER2_Status', 'PR_Status', 'ER_Status'])
PathAI_hifs_with_status


# In[4]:


# Merge on 'sample_id' 
merged_df = pd.merge(PathAI_nuhifs_with_status, PathAI_hifs_with_status, on='sample_id', how='inner')
merged_df


# In[5]:


# Define list of selected feature columns to retain
selected_features = [
    'MEAN[CANCER_NUCLEUS_AREA]_H & E',
    'MEAN[CANCER_NUCLEUS_MAJOR_AXIS_LENGTH]_H & E',
    'MEAN[CANCER_NUCLEUS_MINOR_AXIS_LENGTH]_H & E',
    'MEAN[CANCER_NUCLEUS_PERIMETER]_H & E',
    'MEAN[CANCER_NUCLEUS_CIRCULARITY]_H & E',
    'MEAN[CANCER_NUCLEUS_ECCENTRICITY]_H & E',
    'STD[CANCER_NUCLEUS_AREA]_H & E',
    'STD[CANCER_NUCLEUS_MAJOR_AXIS_LENGTH]_H & E',
    'STD[CANCER_NUCLEUS_MINOR_AXIS_LENGTH]_H & E',
    'STD[CANCER_NUCLEUS_PERIMETER]_H & E',
    'STD[CANCER_NUCLEUS_CIRCULARITY]_H & E',
    'STD[CANCER_NUCLEUS_ECCENTRICITY]_H & E',
    'DENSITY [CANCER CELLS] IN [TUMOR]_HE',
    'DENSITY RATIO [CANCER CELLS] IN [[EPITHELIAL] OVER [TUMOR]]_HE',
    'AREA PROP [[EPITHELIAL] OVER [TUMOR]] IN [TISSUE]_HE',
    'AREA PROP [[ESI_0080] OVER [TUMOR]] IN [TISSUE]_HE',
    'AREA PROP [[STROMA] OVER [TUMOR]] IN [TISSUE]_HE',
    'REGION PROPERTIES: AVERAGE ECCENTRICITY OF SIGNIFICANT REGIONS OF TUMOR_HE',
    'REGION PROPERTIES: AVERAGE SOLIDITY OF SIGNIFICANT REGIONS OF TUMOR_HE',
    'REGION PROPERTIES: ECCENTRICITY OF LARGEST REGION OF TUMOR_HE',
    'STD[CANCER_NUCLEUS_MIN_GRAYSCALE_CHANNEL_GRAY]_H & E',
    'STD[CANCER_NUCLEUS_MIN_HSV_CHANNEL_SATURATION]_H & E',
    'REGION PROPERTIES: FILLED AREA (MM2) OF LARGEST REGION OF TUMOR_HE',
    'REGION PROPERTIES: LACUNARITY OF LARGEST REGION OF TUMOR_HE',
    'REGION PROPERTIES: LACUNARITY OF TUMOR_HE'
]

# Optional: Add sample_id and subtype columns to keep
meta_columns = ['sample_id', 'HER2_Status', 'PR_Status', 'ER_Status']

# Filter merged_df to keep only the selected columns
filtered_df = merged_df[meta_columns + selected_features]



# In[6]:


# Path for the output CSV file
tcga_brca_subtype_to_original_pifs = f"{outputs_path}tcga_brca_subtype_to_original_pifs.csv"

# Write the merged dataframe to a new CSV file
filtered_df.to_csv(tcga_brca_subtype_to_original_pifs, index=False)

print("The files have been mapped and saved to:", tcga_brca_subtype_to_original_pifs)
print("Done!")

