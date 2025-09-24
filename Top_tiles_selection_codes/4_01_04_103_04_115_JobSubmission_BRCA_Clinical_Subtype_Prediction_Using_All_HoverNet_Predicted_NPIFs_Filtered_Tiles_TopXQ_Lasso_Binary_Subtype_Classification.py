#!/usr/bin/env python
# coding: utf-8

# In[57]:


#### ------------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: June 25, 2025 (Modified)
#### job submission script
#### Process HoverNet output using percentile-based tile selection
#### Predict BRCA Subtype Status using all NPIFs with tile filtering from Top X% tiles (5% to 50%)
#### ------------------------------------------------------------------------------------------------

import os
from datetime import date
from argparse import ArgumentParser

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)
print(f"Changed working directory to: {_wpath_}")

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("-run", type=str, default="n", help="whether to submit jobs for running [y/n]")
args = parser.parse_args()
submit_jobs = (args.run.lower() == "y")

# Define dataset and paths
dataset_name = "TCGA_BRCA_FFPE"
datestamp = date.today().strftime("%d%b%Y")
trial = 1

code_path = "/data/Ruppin_AI/BRCA_PIF/Ranjan/Codes/MFfromHnE/"
compute_npifs_script = "2_03_06_NPIFs_Calculation_HoverNetPrediction_All_TCGA_BRCA_Filter_Tiles_TopXQ.py"
map_subtypes_script = "3_01_01_09_Mapped_Original_Value_Hovernet_NPIFs_to_BRCA_Subtypes_Filtered_Tiles_TopXQ.py"
predict_brca_script = "4_01_04_103_04_105_BRCA_Clinical_Subtype_Prediction_Using_All_HoverNet_Predicted_NPIFs_Filtered_Tiles_TopXQ_Binary_Subtype_Classification.py"

job_path = f"{_wpath_}{dataset_name}/outputs/HoverNet/jobs/{datestamp}_{trial}/"
log_path = f"{_wpath_}{dataset_name}/outputs/HoverNet/jobs/logs/{datestamp}_{trial}/"

print(f"Creating job path: {job_path}")
print(f"Creating log path: {log_path}")

os.makedirs(job_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# Define percentile ranges (50% to 95%, in 5% increments)
percentiles = list(range(50, 100, 5))

# Generate job scripts for each percentile
for percentile in percentiles:
    job_file = os.path.join(job_path, f"run_process_Top{percentile}Q.sh")

    job_txt = [
        "#!/bin/bash\n",
        f"#SBATCH --job-name=NPIFs_{percentile}Q\n",
        "#SBATCH --ntasks=1\n",
        "#SBATCH --mem=32g\n",
        "#SBATCH --time=02:00:00\n",
        "#SBATCH --cpus-per-task=4\n",
        f"#SBATCH --output={log_path}process_Top{percentile}Q_%j.out\n",
        "\n",
        "module load python/3.10\n",
        "\n",
        f'echo "Starting NPIF computation for Top {percentile}% tiles..."\n',
        f'compute_job=$(sbatch --parsable <<EOF\n',
        "#!/bin/bash\n",
        "#SBATCH --ntasks=1\n",
        "#SBATCH --mem=32g\n",
        "#SBATCH --time=02:00:00\n",
        "#SBATCH --cpus-per-task=4\n",
        f"#SBATCH --output={log_path}compute_Top{percentile}Q_%j.out\n",
        "module load python/3.10\n",
        f"python {code_path + compute_npifs_script} --percentile {percentile}\n",
        "EOF\n",
        ')\n',
        "\n",
        f'echo "Submitting mapping job for Top {percentile}% tiles after NPIF computation..."\n',
        f'map_job=$(sbatch --parsable --dependency=afterok:${{compute_job}} <<EOF\n',
        "#!/bin/bash\n",
        "#SBATCH --ntasks=1\n",
        "#SBATCH --mem=16g\n",
        "#SBATCH --time=01:00:00\n",
        "#SBATCH --cpus-per-task=2\n",
        f"#SBATCH --output={log_path}map_Top{percentile}Q_%j.out\n",
        "module load python/3.10\n",
        f"python {code_path + map_subtypes_script} --percentile {percentile}\n",
        "EOF\n",
        ')\n',
        "\n",
        f'echo "Submitting prediction job for Top {percentile}% tiles after mapping..."\n',
        f'sbatch --dependency=afterok:${{map_job}} <<EOF\n',
        "#!/bin/bash\n",
        "#SBATCH --ntasks=1\n",
        "#SBATCH --mem=32g\n",
        "#SBATCH --time=04:00:00\n",
        "#SBATCH --cpus-per-task=8\n",
        f"#SBATCH --output={log_path}predict_Top{percentile}Q_%j.out\n",
        "module load python/3.10\n",
        f"python {code_path + predict_brca_script} --percentile {percentile}\n",
        "EOF\n",
    ]

    # Write the job script
    try:
        with open(job_file, "w") as file:
            file.writelines(job_txt)
        print(f"Job script successfully written: {job_file}")
    except Exception as e:
        print(f"Error writing job file: {e}")

# Submit jobs if argument -run=y is passed
if submit_jobs:
    for percentile in percentiles:
        job_file = os.path.join(job_path, f"run_process_Top{percentile}Q.sh")
        print(f"Submitting job for Top {percentile}% tiles...")
        os.system(f"sbatch {job_file}")

print("All percentile jobs submitted!")



