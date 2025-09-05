#!/usr/bin/env python
# coding: utf-8

# In[23]:


#### ------------------------------------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: July 2, 2025
#### predict external BRCA Subtype Status using All HoverNet predicted NPIFs and trained Lasso models from internal CV
#### ------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import random
import joblib

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)
print(f"Working directory = {_wpath_}\n")



# In[24]:


# Define dataset and file paths
dataset_name = "POST_NAT_BRCA"

external_file = f"{dataset_name}/outputs/HoverNet/Subtypes/HoverNet_Original_NPIFs_Values_POST_NAT_BRCA_Mapped_BRCA_Status_Filtered_Tiles_Top25Q.csv"
out_path = f"{dataset_name}/outputs/HoverNet/Subtypes/POST_NAT_BRCA_Predictions/Lasso_HoverNet_All_NPIFs_Filtered_Tiles_Top25Q_Binary_Subtypes/"

model_path = "TCGA_BRCA_FFPE/outputs/HoverNet/Subtypes/Prediction_results_All_HoverNet_NPIFs_Filtered_Tiles_Top25Q_Binary_Subtypes/saved_models/"
os.makedirs(out_path, exist_ok=True)

# Subtype names
subtype_names_3 = ['HER2_Positive', 'HR_Positive', 'TNBC_3']
subtype_names_4 = ['HR+', 'TNBC_4', 'TPBC', 'HER2+']
all_subtypes = subtype_names_3 + subtype_names_4

# Load external dataset
df_external = pd.read_csv(external_file)
df_external.columns = df_external.columns.str.strip()
df_external = df_external[df_external[['HER2_Status', 'PR_Status', 'ER_Status']].isin(["Positive", "Negative"]).all(axis=1)].copy()

df_external


# In[25]:


# Define true subtypes from status
def define_true_subtypes(df):
    HER2 = df['HER2_Status'] == 'Positive'
    ER = df['ER_Status'] == 'Positive'
    PR = df['PR_Status'] == 'Positive'

    df['HER2_Positive'] = HER2.astype(int)
    df['HR_Positive'] = ((ER | PR) & ~HER2).astype(int)
    df['TNBC_3'] = (~HER2 & ~ER & ~PR).astype(int)

    df['HR+'] = (ER & ~HER2).astype(int)
    df['TNBC_4'] = (~ER & ~HER2).astype(int)
    df['TPBC'] = (ER & HER2).astype(int)
    df['HER2+'] = (~ER & HER2).astype(int)
    return df

df_external = define_true_subtypes(df_external)

# Extract features
columns_to_exclude = ['sample_id', 'Patient_ID', 'HER2_Status', 'PR_Status', 'ER_Status', 'Clinical_subtype'] + all_subtypes
feature_columns = [col for col in df_external.columns if col not in columns_to_exclude]
X_external = df_external[feature_columns].copy()

X_external



# In[26]:


# Sample ID list
external_ids = df_external['Patient_ID'].reset_index(drop=True)

# Initialize dataframe to store predictions
external_predictions_df = pd.DataFrame({'Patient_ID': external_ids})
external_predictions_df.set_index('Patient_ID', inplace=True)

# Store true labels
for subtype in all_subtypes:
    external_predictions_df[f'true_label_{subtype}'] = df_external.set_index("Patient_ID")[subtype]

# Prediction function
def predict_external_subtype(subtype_name):
    print(f"\nPredicting subtype: {subtype_name}")
    model_files = sorted([f for f in os.listdir(model_path) if f.startswith(f"{subtype_name}_model_fold_")])
    if not model_files:
        print(f"No model found for {subtype_name}")
        return

    all_preds = []
    for fold_file in model_files:
        model = joblib.load(os.path.join(model_path, fold_file))
        scaler = model.named_steps['scaler']
        classifier = model.named_steps['classifier']

        trained_features = classifier.feature_names_in_ if hasattr(classifier, 'feature_names_in_') else feature_columns
        X_test = X_external[trained_features].copy()
        X_test_scaled = scaler.transform(X_test)

        y_pred = classifier.predict_proba(X_test_scaled)[:, 1]
        all_preds.append(y_pred)

    mean_preds = np.mean(np.column_stack(all_preds), axis=1)
    external_predictions_df[f'predicted_probability_{subtype_name}'] = mean_preds

# Run prediction for all subtypes
for subtype in all_subtypes:
    predict_external_subtype(subtype)

# Save prediction output
output_csv = os.path.join(out_path, "External_POST_NAT_BRCA_Predictions_All_HoverNet_NPIFs_All_Subtypes.csv")
external_predictions_df.to_csv(output_csv)
print(f"\n Predictions saved to: {output_csv}")


# In[27]:


external_predictions_df


# In[28]:


# Mapping for 3-subtype ROC plotting
subtypes_3 = {
    'HER2_Positive': ('true_label_HER2_Positive', 'predicted_probability_HER2_Positive'),
    'HR_Positive': ('true_label_HR_Positive', 'predicted_probability_HR_Positive'),
    'TNBC_3': ('true_label_TNBC_3', 'predicted_probability_TNBC_3')
}

# Mapping for 4-subtype ROC plotting
subtypes_4 = {
    'HR+': ('true_label_HR+', 'predicted_probability_HR+'),
    'TNBC_4': ('true_label_TNBC_4', 'predicted_probability_TNBC_4'),
    'TPBC': ('true_label_TPBC', 'predicted_probability_TPBC'),
    'HER2+': ('true_label_HER2+', 'predicted_probability_HER2+')
}


# In[29]:


colors = {
    'HER2_Positive': 'darkorange',
    'HR_Positive': 'blue',
    'TNBC_3': 'green',
    'HR+': 'blue',
    'TNBC_4': 'green',
    'TPBC': 'tomato',
    'HER2+': 'darkorange'
}
def plot_combined_roc_curves(df, subtypes, title, filename):
    plt.figure(figsize=(10, 8))
    total_samples = len(df)

    for subtype, (true_label_col, pred_prob_col) in subtypes.items():
        if subtype in ['TNBC_3', 'TNBC_4']:
            if 'TNBC' in colors:  # Avoid duplicated legend for TNBC
                continue
            label = 'TNBC'
        elif subtype == 'HER2_Positive':
            label = 'HER2+'
        elif subtype == 'HR_Positive':
            label = 'HR+'
        else:
            label = subtype

        # Filter valid entries
        valid_indices = (~df[true_label_col].isna()) & (~df[pred_prob_col].isna())
        true_labels = df[true_label_col][valid_indices]
        predicted_probs = df[pred_prob_col][valid_indices]

        # Compute ROC and AUC
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.plot(fpr, tpr, color=colors[subtype], lw=2,
                 label=f'{label} (AUC = {roc_auc:.2f}, n = {true_labels.sum():.0f})')

    # Reference diagonal
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(True)

    # Save and show
    plt.savefig(os.path.join(out_path, filename), bbox_inches='tight')
    plt.show()


# In[30]:


plot_combined_roc_curves(
    external_predictions_df,
    subtypes_3,
    'ROC Curves for BRCA 3 Clinical Subtypes',
    'roc_3_subtypes.png'
)

plot_combined_roc_curves(
    external_predictions_df,
    subtypes_4,
    'ROC Curves for BRCA 4 Clinical Subtypes',
    'roc_4_subtypes.png'
)



# In[31]:


# Prepare positive and negative subsets using true label columns
HER2_pos = external_predictions_df[external_predictions_df['true_label_HER2_Positive'] == 1]
HER2_neg = external_predictions_df[external_predictions_df['true_label_HER2_Positive'] == 0]

HR_pos = external_predictions_df[external_predictions_df['true_label_HR_Positive'] == 1]
HR_neg = external_predictions_df[external_predictions_df['true_label_HR_Positive'] == 0]

TNBC_pos = external_predictions_df[external_predictions_df['true_label_TNBC_3'] == 1]
TNBC_neg = external_predictions_df[external_predictions_df['true_label_TNBC_3'] == 0]

# Construct a DataFrame for boxplot visualization
plot_df = pd.DataFrame({
    'Subtype_Label': ['HER2_Positive_Pos'] * len(HER2_pos) +
                     ['HER2_Positive_Neg'] * len(HER2_neg) +
                     ['HR_Positive_Pos'] * len(HR_pos) +
                     ['HR_Positive_Neg'] * len(HR_neg) +
                     ['TNBC_Pos'] * len(TNBC_pos) +
                     ['TNBC_Neg'] * len(TNBC_neg),
    'Predicted_Probability': pd.concat([
        HER2_pos['predicted_probability_HER2_Positive'],
        HER2_neg['predicted_probability_HER2_Positive'],
        HR_pos['predicted_probability_HR_Positive'],
        HR_neg['predicted_probability_HR_Positive'],
        TNBC_pos['predicted_probability_TNBC_3'],
        TNBC_neg['predicted_probability_TNBC_3']
    ], axis=0)
})

# Define color palette for positive/negative cases
palette = {
    'HER2_Positive_Pos': '#1f77b4',
    'HER2_Positive_Neg': '#d62728',
    'HR_Positive_Pos': '#1f77b4',
    'HR_Positive_Neg': '#d62728',
    'TNBC_Pos': '#1f77b4',
    'TNBC_Neg': '#d62728'
}

# Create the boxplot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(
    data=plot_df,
    x='Subtype_Label',
    y='Predicted_Probability',
    hue='Subtype_Label',
    palette=palette,
    legend=False
)

# Annotate sample sizes above each box
group_counts = plot_df['Subtype_Label'].value_counts()
for tick, label in enumerate(ax.get_xticklabels()):
    subtype = label.get_text()
    count = group_counts[subtype]
    ax.text(tick, 1.08, f'n = {count}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='black')

# Final formatting
ax.set_ylim(0, 1.15)
plt.title("Predicted Probability by Subtype (Positive vs Negative Cases)", fontsize=14)
plt.xlabel("Subtype Group", fontsize=12)
plt.ylabel("Predicted Probability", fontsize=12)
plt.xticks(rotation=30, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# In[32]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define subtype labels
subtypes = ['HER2_Positive', 'HR_Positive', 'TNBC']

# Step 1: Define true labels using true_label_<subtype> columns
def get_true_subtype(row):
    if row['true_label_HER2_Positive'] == 1:
        return 'HER2_Positive'
    elif row['true_label_HR_Positive'] == 1:
        return 'HR_Positive'
    elif row['true_label_TNBC_3'] == 1:
        return 'TNBC'
    else:
        return 'Unknown'

# Step 2: Assign predicted subtype using max of predicted probabilities
def get_predicted_subtype(row):
    probs = {
        'HER2_Positive': row['predicted_probability_HER2_Positive'],
        'HR_Positive': row['predicted_probability_HR_Positive'],
        'TNBC': row['predicted_probability_TNBC_3']
    }
    return max(probs, key=probs.get)

# Apply true and predicted subtype logic
external_predictions_df['true_subtype'] = external_predictions_df.apply(get_true_subtype, axis=1)
external_predictions_df['predicted_subtype'] = external_predictions_df.apply(get_predicted_subtype, axis=1)

# Step 3: Filter valid rows (exclude ambiguous/unknown cases)
valid_df = external_predictions_df[external_predictions_df['true_subtype'] != 'Unknown']

# Step 4: Compute confusion matrix
cm = confusion_matrix(valid_df['true_subtype'], valid_df['predicted_subtype'], labels=subtypes)

# Step 5: Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=subtypes, yticklabels=subtypes)
plt.xlabel("Predicted Subtype", fontsize=12)
plt.ylabel("True Subtype", fontsize=12)
plt.title("Confusion Matrix (Max Predicted Probability)", fontsize=14)
plt.tight_layout()
plt.show()


# In[33]:


import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define subtype order
subtypes = ['HER2_Positive', 'HR_Positive', 'TNBC']

# Step 1: Extract true and predicted labels
y_true = valid_df['true_subtype']
y_pred = valid_df['predicted_subtype']

# Step 2: Compute overall accuracy
overall_accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Step 3: Generate classification report for defined subtypes
report = classification_report(y_true, y_pred, labels=subtypes, output_dict=True)

# Step 4: Compute per-class metrics using confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=subtypes)
balanced_accuracies = []
specificities = []
recalls = []

for i, label in enumerate(subtypes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # same as recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    recalls.append(round(sensitivity, 4))
    specificities.append(round(specificity, 4))
    balanced_accuracies.append(round(balanced_accuracy, 4))

# Step 5: Create metrics dataframe in required column order
metrics_df = pd.DataFrame({
    'balanced_accuracy': balanced_accuracies,
    'sensitivity (recall)': recalls,
    'specificity': specificities,
    'precision': [report[label]['precision'] for label in subtypes],
    'f1-score': [report[label]['f1-score'] for label in subtypes],
    'support': [report[label]['support'] for label in subtypes],
}, index=subtypes)

# Step 6: Round for display
metrics_df = metrics_df.round(4)

# Step 7: Compute and print mean balanced accuracy
mean_balanced_accuracy = round(sum(balanced_accuracies) / len(balanced_accuracies), 4)
print(f"\nMean Balanced Accuracy: {mean_balanced_accuracy:.4f}")

# Step 8: Print final metrics
print("\nPer-Class Performance Metrics:")
metrics_df

