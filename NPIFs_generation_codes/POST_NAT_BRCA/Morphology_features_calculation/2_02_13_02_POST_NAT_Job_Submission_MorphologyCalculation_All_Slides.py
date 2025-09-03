#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ----------------------------------------------------------------
#### job submission script
#### author: Ranjan Barman, date: June 22, 2025
#### Process all POST_NAT_BRCA slides in HoverNet output directory
#### ----------------------------------------------------------------

import os
from datetime import date
from argparse import ArgumentParser

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.chdir(_wpath_)
print(f"Changed working directory to: {_wpath_}")

parser = ArgumentParser()
parser.add_argument("-run", type=str, default="n", help="whether to submit jobs for running [y/n]")
args = parser.parse_args()

dataset_name = "POST_NAT_BRCA"
submit_jobs = (args.run.lower() == "y")
datestamp = date.today().strftime("%d%b%Y")
trial = 1

code_path = "/data/Ruppin_AI/BRCA_PIF/Ranjan/Codes/POST_NAT_Codes/"
code_file = "2_02_03_02_POST_NAT_MorphologyCalculation_All_Slides.py"

job_path = f"{_wpath_}{dataset_name}/HoverNet/outputs/jobs/{datestamp}_{trial}/"
log_path = f"{_wpath_}{dataset_name}/HoverNet/outputs/jobs/logs/{datestamp}_{trial}/"
print(f"Creating job path: {job_path}")
print(f"Creating log path: {log_path}")
os.makedirs(job_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

hovernet_base_dir = f"{dataset_name}/HoverNet/outputs/"
slide_folders = sorted([
    f for f in os.listdir(hovernet_base_dir)
    if os.path.isdir(os.path.join(hovernet_base_dir, f))
    and f not in ["jobs", "models"]
])

if not slide_folders:
    print("No slide folders found for processing.")
    exit(1)

for slide_folder in slide_folders:
    job_file = os.path.join(job_path, f"run_process_{slide_folder}.sh")

    job_txt = [
        "#!/bin/bash\n",
        "#SBATCH --ntasks=1\n",
        "#SBATCH --mem=64g\n",
        "#SBATCH --time=00:30:00\n",
        "#SBATCH --gres=lscratch:20\n",
        "#SBATCH --cpus-per-task=4\n",
        f"#SBATCH --output={log_path}process_{slide_folder}_%j.out\n",
        "\n",
        f'SCRIPT="{code_path + code_file}"\n',
        f'SLIDE="{slide_folder}"\n',
        "\n",
        "module load python/3.10\n",
        "python $SCRIPT -slide $SLIDE\n"
    ]

    try:
        with open(job_file, "w") as file:
            file.writelines(job_txt)
        print(f"Job script written: {job_file}")
    except Exception as e:
        print(f"Error writing job script for {slide_folder}: {e}")

if submit_jobs:
    print(f"Submitting {len(slide_folders)} jobs...")
    for slide_folder in slide_folders:
        job_file = os.path.join(job_path, f"run_process_{slide_folder}.sh")
        os.system(f"sbatch {job_file}")

print("All job scripts created and submitted (if -run y was specified).")


# In[ ]:




