#!/usr/bin/env python
# coding: utf-8

# In[9]:


#### ------------------------------------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Aug 18, 2024
#### predict individual BRCA Subtype Status based on all HIFs using nested cross-validation 
#### ------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve, auc
from collections import defaultdict
import random

_wpath_ = "/data/Ruppin_AI/Datasets/TCGA_BRCA_FFPE/"   # set working directory as the parent directory where all datasets are saved
os.chdir(_wpath_)

print(f"working directory = {_wpath_}\n")


# In[10]:


# Data directories & files
outcome_names = ["HER2_Status", "PR_Status", "ER_Status"]

# Load merge data file
BrcaSubtypesToDeePT = "TCGA_BRCA_Subtypes/data/BrcaSubtypesToDeePT.csv"
data_full = pd.read_csv(BrcaSubtypesToDeePT)

# create directories to save outputs
outputs_path = f"TCGA_BRCA_Subtypes/Ranjan_New/outputs_clinical/resnet_result/"
os.makedirs(outputs_path, exist_ok = True)


# Rename the columns in data_full
data_full = data_full.rename(columns={"sampleID": "sample_id", "HER2_Final_Status_nature2012": "HER2_Status", "PR_Status_nature2012": "PR_Status", "ER_Status_nature2012": "ER_Status"})
data_full

# The number of samples used for TCGA BRCA subtype prediction 
TCGA_BRCA_Subtypes_556_file = "/data/Lab_ruppin/Ranjan/HnE/TCGA_BRCA_FFPE/outputs/HoverNet/Subtypes/outputs_biomarker_status_prediction_results_All_HoverNet_NPIFs/combined_class_predictions_all_features_for_subtypes.csv"

TCGA_BRCA_Subtypes_556 = pd.read_csv(TCGA_BRCA_Subtypes_556_file)

# Convert the first column ("sample") to a series
sample_list_556 = TCGA_BRCA_Subtypes_556['sample_id'].reset_index(drop=True)

# Filter sample based on `sample_list_556`
data_full = data_full[data_full['sample_id'].isin(sample_list_556)].copy()

# Convert the first column ("sample") to a series
patient_list = data_full['sample_id'].reset_index(drop=True)

# Filter to include only rows where outcome is "Positive" or "Negative"
data_filtered = data_full[
    data_full[outcome_names].isin(["Positive", "Negative"]).all(axis=1)
]


# Filter out the patient list
filtered_patient_list = data_filtered['sample_id'].reset_index(drop=True)




data_full
data_filtered
# X


# In[11]:


# load features list

model = "resnet"

# CTransPath features
if model == "CtransPath":
    path2features = "/data/Ruppin_AI/Datasets/TCGA_BRCA_FFPE/features.pkl"

# resnet features
if model == "resnet":
    path2features = "/data/Ruppin_AI/Datasets/TCGA_BRCA_FFPE/AE_features.pkl"

features_list = np.load(path2features, allow_pickle=True)


# In[12]:


# make a new features list, only including the patients that have labeled clinical subtypes, as indicated by map_df
new_features = []

for idx in data_filtered['index']:
    # obtain mean of all tiles for a slide
    avg_features = np.mean(features_list[idx][1], axis=0)
    new_features.append(avg_features)
    
new_features = np.array(new_features)

# save file
# CTransPath features
if model == "CtransPath":
    new_features_path = "/data/Ruppin_AI/Datasets/TCGA_BRCA_FFPE/TCGA_BRCA_Subtypes/data/CtransPath_features_for_subtypes.txt"

# resnet features
if model == "resnet":
    new_features_path = "/data/Ruppin_AI/Datasets/TCGA_BRCA_FFPE/TCGA_BRCA_Subtypes/data/resnet50_features_for_subtypes.txt"

np.savetxt(new_features_path, new_features)


# In[ ]:




