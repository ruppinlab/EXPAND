#!/usr/bin/env python
# coding: utf-8

# In[18]:


#### -------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Feb 5, 2025
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
dataset_name = "CPTAC_BRCA"
tiles_path = "/data/Lab_ruppin/dhrubas2/HnE/CPTAC_BRCA/outputs/tiles"


selected_slides = {
    "11BR006-a3e2ab4d-ef4a-497e-b0d2-e8d1cf_649_tiles.bz2",
    "11BR057-7a9a5fc7-9bef-420c-9c9d-6d8810_2058_tiles.bz2",
}

slides_files = [f for f in os.listdir(tiles_path) if f in selected_slides]

if not slides_files:
    raise FileNotFoundError("No matching .bz2 files found in the tiles directory.")

code_path = "/data/Ruppin_AI/BRCA_PIF/Ranjan/Codes/MFfromHnE/"
code_file = "2_01_22_02_Test_Dataset_ExtractMorphologicalFeaturesFromHnE.py"

trial = 2
job_path = f"{_wpath_}{dataset_name}/HoverNet/outputs/jobs/{datestamp}_{trial}/"
log_path = f"{_wpath_}{dataset_name}/HoverNet/outputs/jobs/logs/{datestamp}_{trial}/"
os.makedirs(job_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

for idx, slide_file in enumerate(tqdm(slides_files)):
    slide_name = os.path.splitext(slide_file)[0]
    job_file_j = f"hovernet_slide_{idx + 1}.sh"
    job_script_path = os.path.join(job_path, job_file_j)
    with open(job_script_path, "w") as file:
        file.writelines([
            "#!/bin/bash\n",
            "#SBATCH --ntasks=4\n",
            "#SBATCH --partition=gpu\n",
            "#SBATCH --gres=gpu:k80:2,lscratch:20\n",
            "#SBATCH --cpus-per-task=8\n",
            "#SBATCH --mem=64g\n",
            "#SBATCH --time=36:00:00\n",
            f"#SBATCH --job-name=hovernet_{slide_name}\n",
            f"#SBATCH --output={log_path}hovernet_{idx + 1}.out\n",
            f"#SBATCH --error={log_path}hovernet_{idx + 1}.err\n",
            "\n",
            "source ~/miniconda3/etc/profile.d/conda.sh\n",
            "conda activate hovernet\n",
            "module load CUDA/10.2\n",
            "module load gcc/13.2.0\n",
            f"python {code_path + code_file} -slide {slide_file} -tile_path {tiles_path} -wd {_wpath_}\n",
        ])
    if submit_jobs:
        os.system(f"sbatch {job_script_path}")
        time.sleep(submission_delay)

print("Job submission completed.")


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




