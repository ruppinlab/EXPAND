#!/usr/bin/env python3
# coding: utf-8

#### ----------------------------------------------------------------
#### refactored job submission script
#### author: dhrubas2, date: Jun 07, 2024
#### collect slide features and masks in a dataset in single files
#### ----------------------------------------------------------------

import os
from datetime import date
from argparse import ArgumentParser

# _wpath_ = "/data/Lab_ruppin/dhrubas2/HnE/"        # set working directory as the parent directory where all datasets are saved
# os.chdir(_wpath_)


#### ----------------------------------------------------------------
#%%  cmd arguments.
#### ----------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("-dat", type = str, 
                    help = "dataset to use i.e., folder name containing the slides in slides/ subdirectory")
parser.add_argument("-feat", type = str,
                    help = "feature type [ResNet / ViT / CTrans] for each slide to collate")
parser.add_argument("-wd", type = str, default = "/data/Lab_ruppin/dhrubas2/HnE/", 
                    help = "working directory - set this as the parent directory where all datasets are stored")
parser.add_argument("-mask", type = str, default = "n", 
                    help = "whether to save all masks in a dataset in one single PDF file [y/n]")
parser.add_argument("-run", type = str, default = "n", 
                    help = "whether to submit job for running [y/n]")
parser.add_argument("-date", type = str, default = date.today().strftime("%d%b%Y"), 
                    help = "date stamp (in dd-mmm-YYYY format) for analysis results/logs e.g., 14Feb2024")
args = parser.parse_args()

dataset_name = args.dat
feature_name = args.feat
_wpath_      = args.wd
save_masks   = args.mask
submit_jobs  = (args.run.lower() == "y")
datestamp    = args.date

os.chdir(_wpath_)


#### ----------------------------------------------------------------
#%%  get path & directories.
#### ----------------------------------------------------------------

code_path   = "/data/Lab_ruppin/dhrubas2/HistologyToProteomics/analysis/"
code_file = "1_03_collect_all_features_masks.py"

## create job paths.
job_path  = f"{_wpath_}{dataset_name}/jobs/features/{datestamp}/"
log_path  = f"{_wpath_}{dataset_name}/jobs/logs/features/{datestamp}/"
os.makedirs(job_path, exist_ok = True)                      # creates directory if doesn't exist already
os.makedirs(log_path, exist_ok = True)


#### ----------------------------------------------------------------
#%%  write job.
#### ----------------------------------------------------------------

n_slides = len(os.listdir(f"{_wpath_}{dataset_name}/outputs/features/"))
print(f"writing job script for collecting {n_slides:,} slide features/masks for {dataset_name}...")

job_file = "collate_features_masks.sh"

## job script.
job_txt  = [
    "#!/bin/bash\n", 
    "#SBATCH --ntasks=4\n", 
    "#SBATCH --mem=30g\n", 
    "#SBATCH --time=02:00:00\n", 
    "#SBATCH --partition=norm\n", 
    "#SBATCH --gres=lscratch:20\n", 
    "#SBATCH --cpus-per-task=14\n", 
    f"#SBATCH --output={log_path}job_collate_%j.out\n", 
    "#SBATCH --mail-type=FAIL,TIME_LIMIT_80\n", 
    "\n", 
    f'SCRIPT="{code_path + code_file}"\n', 
    f'PROJ="{dataset_name}"\n', 
    f'FEAT="{feature_name}"\n', 
    f'WD="{_wpath_}"\n', 
    f'MASK="{save_masks}"\n', 
    # f'LOG="{log_path}$PBS_JOBID.log"\n', 
    "\n", 
    "module load python/3.10\n", 
    "python $SCRIPT -dat=$PROJ -feat=$FEAT -wd=$WD -mask=$MASK"]

with open(job_path + job_file, "w") as file:
    file.writelines(job_txt)

print(f"done! job script is in = {job_path + job_file}\n")


#### ----------------------------------------------------------------
#%%  submit job.
#### ----------------------------------------------------------------

if submit_jobs:
    print(f"submitting job for collecting {n_slides:,} features/masks for {dataset_name}...")
    
    job_submit = f"sbatch --job-name=collate_fm {job_path + job_file}"    # e.g.: sbatch --job-name=collate_fm jobs/features/06Nov2023/collate_features_masks.sh
    os.system(job_submit)                                                 # execute commands
    
    print("done!\n")


#### ----------------------------------------------------------------
## to run on cmd: 
## python 1_13_jobs_to_collect_features2.py -dat "CPTAC_BRCA" -feat "CTrans" -mask "y" -run "y"

