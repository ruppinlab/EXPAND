#!/usr/bin/env python
# coding: utf-8

# In[19]:


#### -----------------------------------------------------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: May 5, 2025
#### Mapped PathAI nuHIFs to TCGA biomarker status with Age
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


# In[20]:


# Define dataset and file paths
dataset_name = "TCGA_BRCA_FFPE"
outcome_names = ["HER2_Status", "PR_Status", "ER_Status"]


# File paths
PathAI_HIFs_with_BRCA_status_file = "PA_PIF_BRCA/outputs_biomarker_status/tcga_brca_subtype_to_original_pifs.csv"

pathAI_TCGA_BRCA_survival_data_file = "PA_NUHIF_BRCA/data/PathAI_BRCA_MetaData.xlsx"

out_path = f"{dataset_name}/outputs/PathAI/Survival_V2/"

os.makedirs(out_path, exist_ok=True) # Creates directory if it doesn't exist already

# Load data
data_full = pd.read_csv(PathAI_HIFs_with_BRCA_status_file)
patient_list = data_full['sample_id'].reset_index(drop=True)
data_full
data_full


# Filter to include only rows where outcome is "Positive" or "Negative"
data_filtered = data_full[data_full[outcome_names].isin(["Positive", "Negative"]).all(axis=1)].copy() # Use .copy() to avoid SettingWithCopyWarning
filtered_patient_list = patient_list[data_filtered.index]

# Create a new DataFrame for original nuHIFs with proper status
data_orginal_HIFs = data_filtered.copy()  # Keep a copy of the filtered data before dropping columns

data_orginal_HIFs




# The number of samples used for TCGA BRCA subtype prediction 
TCGA_BRCA_Subtypes_556_file = f"{dataset_name}/outputs/HoverNet/Subtypes/outputs_biomarker_status_prediction_results_All_HoverNet_NPIFs/combined_class_predictions_all_features_for_subtypes.csv"

TCGA_BRCA_Subtypes_556 = pd.read_csv(TCGA_BRCA_Subtypes_556_file)

# Convert the first column ("sample") to a series
sample_list_556 = TCGA_BRCA_Subtypes_556['sample_id'].reset_index(drop=True)

# Filter sample based on `sample_list_556`
PathAI_HIFs_with_BRCA_status = data_orginal_HIFs[data_orginal_HIFs['sample_id'].isin(sample_list_556)].copy()
PathAI_HIFs_with_BRCA_status


#read files 
pathAI_TCGA_BRCA_survival_data = pd.read_excel(pathAI_TCGA_BRCA_survival_data_file)



# Rename the columns in pathAI_TCGA_BRCA_survival_data
pathAI_TCGA_BRCA_survival_data = pathAI_TCGA_BRCA_survival_data.rename(columns={"age_at_initial_pathologic_diagnosis": "Age", "ajcc_pathologic_tumor_stage": "Stage"})

PathAI_HIFs_with_BRCA_status
# pathAI_TCGA_BRCA_survival_data



# In[21]:


# Define the desired order for the specified columns
desired_columns = ['os', 'os_time', 'pfs', 'pfs_time', 'Age', 'Stage', 'bcr_patient_barcode']

# Apply the new column order to the DataFrame
pathAI_TCGA_BRCA_survival_data_with_desired_columns = pathAI_TCGA_BRCA_survival_data[desired_columns]
pathAI_TCGA_BRCA_survival_data_with_desired_columns


# In[22]:


# 'sample_id' is the column name in df1 and 'Sample' is the column name in df2
# Adjust these column names to match the actual column names in your CSV files
merged_df1 = pd.merge(PathAI_HIFs_with_BRCA_status, pathAI_TCGA_BRCA_survival_data_with_desired_columns, left_on='sample_id', right_on='bcr_patient_barcode', how='inner')

# Drop the redundant 'bcr_patient_barcode' column
merged_df1.drop(columns='bcr_patient_barcode', inplace=True)

merged_df1



# In[23]:


# Path for the output CSV file
BrcaBiomarkerStatusToPathAI_PIFsToSurvival = f"{out_path}BrcaBiomarkerStatusToPathAI_PIFsToSurvival.csv"

# Write the merged dataframe to a new CSV file
merged_df1.to_csv(BrcaBiomarkerStatusToPathAI_PIFsToSurvival, index=False)

print("The files have been mapped and saved to:", BrcaBiomarkerStatusToPathAI_PIFsToSurvival)
print("Done!")


# In[ ]:




