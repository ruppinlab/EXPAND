#!/usr/bin/env python
# coding: utf-8

# In[11]:


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


# In[12]:


#%% get TCGA_BRCA subtypes & input HIF features data.

dataset_name = "PA_NUHIF_BRCA"

data_path    = [f"{dataset_name}/data/"]
data_files   = ["PathAI_BRCA_MetaData.xlsx",
                "PathAI_BRCA_NuHIFs.xlsx"]

# Data directories & files
dataset_name1 = "PA_HIF_BRCA"
feature_name1 = "HIF"
outcome_names = ["HER2_Status", "PR_Status", "ER_Status"]

data_path1 = f"{dataset_name1}/outputs_clinical/"
data_file1 = "TCGA_BRCA_Subtypes_clinical.tsv"

## create directories to save outputs
outputs_path = f"{dataset_name}/outputs_biomarker_status/"
os.makedirs(outputs_path, exist_ok = True)

#read metadata file of PathAI 
PathAI_meta_data = pd.read_excel(data_path[0] + data_files[0])

#read nuHIF file of PathAI 
PathAI_nuHIFs_data = pd.read_excel(data_path[0] + data_files[1])


# samples
PathAI_meta_data.head()
PathAI_nuHIFs_data.head()


# In[13]:


# 'H & E_ID' is the column name in df1 and 'H & E_ID' is the column name in df2
# Adjust these column names to match the actual column names in your CSV files
merged_df = pd.merge(PathAI_meta_data, PathAI_nuHIFs_data, left_on='H & E_ID', right_on='H & E_ID', how='inner')
merged_df


# In[14]:


# Define the patterns to look for at the start of column names
start_patterns = [
    'MEAN[CANCER', 
    'MEAN[FIBROBLAST', 
    'MEAN[LYMPHOCYTE', 
    'STD[CANCER', 
    'STD[FIBROBLAST', 
    'STD[LYMPHOCYTE'
]

# Define the strings that should be present in the column names
required_strings = [
    'AREA', 
    'MAJOR_AXIS_LENGTH', 
    'MINOR_AXIS_LENGTH', 
    'PERIMETER', 
    'CIRCULARITY', 
    'ECCENTRICITY', 
    'SOLIDITY', 
    'STD_GRAYSCALE_CHANNEL_GRAY', 
    'STD_HSV_CHANNEL_SATURATION', 
    'STD_LAB_CHANNEL_A', 
    'STD_LAB_CHANNEL_B', 
    'MIN_GRAYSCALE_CHANNEL_GRAY', 
    'MIN_HSV_CHANNEL_SATURATION', 
    'MEAN_LAB_CHANNEL_A', 
    'MEAN_LAB_CHANNEL_B'
]

# Extract columns that start with the specified patterns and contain any of the required strings
cancer_fibroblast_lymphocyte_columns = [
    col for col in merged_df.columns if 
    any(col.startswith(pattern) for pattern in start_patterns) and 
    any(req_str in col for req_str in required_strings)
]

# Ensure 'bcr_patient_barcode' and 'subtype' are present in the DataFrame
# Add them to the start of the filtered columns
essential_columns = ['bcr_patient_barcode']

# Combine the essential columns with cancer_fibroblast_lymphocyte_columns
final_columns = essential_columns + cancer_fibroblast_lymphocyte_columns


# Extract the relevant data
extracted_subtype_nuHIFs = merged_df[final_columns]

#Rename the bcr_patient_barcode column to sample
extracted_subtype_nuHIFs = extracted_subtype_nuHIFs.rename(columns={'bcr_patient_barcode': 'sample'})

extracted_subtype_nuHIFs


# In[15]:


# TCGA BRCA sub type data
tcga_subtypes_data   = pd.read_table(data_path1 + data_file1, sep = "\t")

# Remove '-01' from each sampleID to match with HIF Sample ID
tcga_subtypes_data['sampleID'] = tcga_subtypes_data['sampleID'].str.replace('-01', '', regex=False)

# Rename the columns in tcga_subtypes_data
tcga_subtypes_data = tcga_subtypes_data.rename(columns={"HER2_Final_Status_nature2012": "HER2_Status", "PR_Status_nature2012": "PR_Status", "ER_Status_nature2012": "ER_Status"})

tcga_subtypes_data


# In[16]:


# '# 'sample' is the column name in df1 and 'sampleID' is the column name in df2
# Adjust these column names to match the actual column names in your CSV files
merged_df1 = pd.merge(extracted_subtype_nuHIFs, tcga_subtypes_data, left_on='sample', right_on='sampleID', how='inner')

merged_df1


# In[17]:


# Remove the 'sampleID' column
merged_df1 = merged_df1.drop(columns=['sampleID'])

# Define the desired order for the specified columns
desired_columns = ['HER2_Status', 'PR_Status', 'ER_Status', 'BRCA_Subtypes']

# Get the list of existing columns excluding the desired columns
remaining_columns = [col for col in merged_df1.columns if col not in desired_columns and col != 'sample']

# Reorder the columns: 'sample' followed by desired_columns and then the remaining columns
new_column_order = ['sample'] + desired_columns + remaining_columns

# Apply the new column order to the DataFrame
merged_df1 = merged_df1[new_column_order]

# Rename 'sample' to 'sample_id'
merged_df1 = merged_df1.rename(columns={'sample': 'sample_id'})
merged_df1

# Display the resulting DataFrame
merged_df1


# In[18]:


# Path for the output CSV file
tcga_brca_subtype_to_original_nuhifs = f"{outputs_path}tcga_brca_subtype_to_original_nuhifs.csv"

# Write the merged dataframe to a new CSV file
merged_df1.to_csv(tcga_brca_subtype_to_original_nuhifs, index=False)

print("The files have been mapped and saved to:", tcga_brca_subtype_to_original_nuhifs)
print("Done!")

