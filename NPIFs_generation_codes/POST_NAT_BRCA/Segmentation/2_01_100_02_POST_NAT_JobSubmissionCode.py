#!/usr/bin/env python
# coding: utf-8

# In[18]:


#### -------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: June 18, 2025
#### Job submission code for extracting morphological features from H&E
#### --------------------------------------------------------------------------------------

import os
import time
from datetime import date
from tqdm import tqdm
from argparse import ArgumentParser

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)

# Command-line arguments
parser = ArgumentParser(description="Submit SLURM jobs for HoVer-Net inference on multiple slides.")
parser.add_argument("-run", type=str, default="n", help="Whether to submit jobs for running [y/n]")
parser.add_argument("-date", type=str, default=date.today().strftime("%d%b%Y"), help="Date stamp for logs, e.g., 22Jan2025")
parser.add_argument("-delay", type=int, default=5, help="Delay in seconds between job submissions")
args = parser.parse_args()

# Validate arguments
submit_jobs = (args.run.lower() == "y")
datestamp = args.date
submission_delay = args.delay

# Paths and directories
dataset_name = "POST_NAT_BRCA"
tiles_path = "/data/Ruppin_AI/Datasets/Post_NAT_BRCA/outputs/tiles/"
tiles_files = [f for f in os.listdir(tiles_path) if f.endswith(".bz2")]

if not tiles_files:
    raise FileNotFoundError(f"No .bz2 files found in the tiles directory: {tiles_path}")

code_path = "/data/Ruppin_AI/BRCA_PIF/Ranjan/Codes/POST_NAT_Codes/"
code_file = "2_01_22_02_Test_POST_NAT_Dataset_ExtractMorphologicalFeaturesFromHnE.py"

trial = 1
job_path = f"{_wpath_}{dataset_name}/HoverNet/outputs/jobs/{datestamp}_{trial}/"
log_path = f"{_wpath_}{dataset_name}/HoverNet/outputs/jobs/logs/{datestamp}_{trial}/"
os.makedirs(job_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# Write job scripts
n_slides = len(tiles_files)
job_txt = [
    "#!/bin/bash\n",
    "#SBATCH --ntasks=4\n",
    "#SBATCH --partition=gpu\n",
    "#SBATCH --gres=gpu:k80:2,lscratch:20\n",
    "#SBATCH --cpus-per-task=8\n",
    "#SBATCH --mem=64g\n",
    "#SBATCH --time=8:00:00\n",
]

print(f"Writing job scripts for {n_slides:,} slides...")

for idx, slide_file in enumerate(tqdm(tiles_files)):
    slide_name = os.path.splitext(slide_file)[0]
    job_file_j = f"hovernet_slide_{idx + 1}.sh"
    job_txt_j = job_txt + [
        f"#SBATCH --job-name=hovernet_{slide_name}\n",
        f"#SBATCH --output={log_path}hovernet_{idx + 1}.out\n",
        f"#SBATCH --error={log_path}hovernet_{idx + 1}.err\n",
        "\n",
        f'SCRIPT="{code_path + code_file}"\n',
        f'TILE_PATH="{tiles_path}"\n',
        f'SLIDE="{slide_file}"\n',
        f'WD="{_wpath_}"\n',
        "\n",
        "source ~/miniconda3/etc/profile.d/conda.sh\n",
        "conda activate hovernet\n",
        "module load CUDA/10.2\n",
        "module load gcc/13.2.0\n",
        "python $SCRIPT -slide $SLIDE -tile_path $TILE_PATH -wd $WD\n",
    ]

    with open(os.path.join(job_path, job_file_j), "w") as file:
        file.writelines(job_txt_j)

print(f"Done! Job scripts are saved in: {job_path}\n")

# Submit jobs
if submit_jobs:
    print(f"Submitting jobs for {n_slides:,} slides with a delay of {submission_delay} seconds...")

    for idx in range(n_slides):
        job_file_j = os.path.join(job_path, f"hovernet_slide_{idx + 1}.sh")
        os.system(f"sbatch {job_file_j}")
        time.sleep(submission_delay)

    print("All jobs submitted!\n")
else:
    print("Job submission skipped. To submit jobs, use -run y.")


# In[1]:


# import torch

# # Check GPU and CUDA compatibility
# def check_gpu():
#     if not torch.cuda.is_available():
#         raise EnvironmentError("No GPU detected. Ensure CUDA and NVIDIA drivers are properly installed.")
#     print(f"PyTorch version: {torch.__version__}")
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"GPU: {torch.cuda.get_device_name(0)} is available.")

# # Set device for inference
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Call the GPU check function
# check_gpu()


# In[ ]:




