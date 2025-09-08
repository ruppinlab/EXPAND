#!/usr/bin/env python3
# coding: utf-8

#### ----------------------------------------------------------------
#### refactored job submission script
#### author: dhrubas2, date: Nov 02, 2023
#### extract image features from all slides in a dataset
#### ----------------------------------------------------------------

import os
from datetime import date
from tqdm import tqdm
from argparse import ArgumentParser

# _wpath_ = "/data/Lab_ruppin/dhrubas2/HnE/"        # set working directory as the parent directory where all datasets are saved
# os.chdir(_wpath_)


#### ----------------------------------------------------------------
# %% cmd arguments.
#### ----------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("-dat", type = str, 
                    help = "dataset to use i.e., folder name containing the slides in slides/ subdirectory")
parser.add_argument("-feat", type = str, 
                    help = "feature type to extract from the tiles [ResNet / ViT / CTrans]")
parser.add_argument("-wd", type = str, default = "/data/Lab_ruppin/dhrubas2/HnE/", 
                    help = "working directory - set this as the parent directory where all datasets are stored")
parser.add_argument("-run", type = str, default = "n", 
                    help = "whether to submit jobs for running [y/n]")
parser.add_argument("-date", type = str, default = date.today().strftime("%d%b%Y"), 
                    help = "date stamp (in dd-mmm-YYYY format) for analysis results/logs e.g., 14Feb2024")
args = parser.parse_args()

dataset_name = args.dat
feature_name = args.feat
_wpath_      = args.wd
submit_jobs  = (args.run.lower() == "y")
datestamp    = args.date

os.chdir(_wpath_)


#### ----------------------------------------------------------------
# %% get path & directories.
#### ----------------------------------------------------------------

tiles_path  = f"{dataset_name}/outputs/tiles/"
tiles_files = sorted([fn for fn in os.listdir(tiles_path) if "tiles.bz2" in fn])

code_path   = "/data/Lab_ruppin/dhrubas2/HistologyToProteomics/analysis/"
code_file   = "1_02_get_features_from_tiles2.py"

## create job paths.
job_path    = f"{_wpath_}{dataset_name}/jobs/features/{datestamp}/"
log_path    = f"{_wpath_}{dataset_name}/jobs/logs/features/{datestamp}/"
os.makedirs(job_path, exist_ok = True)                      # creates directory if doesn't exist already
os.makedirs(log_path, exist_ok = True)


#### ----------------------------------------------------------------
# %% write jobs.
#### ----------------------------------------------------------------

n_slides = len(tiles_files)

## job scripts.
job_txt  = ["#!/bin/bash\n", 
            "#SBATCH --ntasks=4\n", 
            "#SBATCH --mem=30g\n", 
            "#SBATCH --time=06:00:00\n", 
            "#SBATCH --partition=norm\n", 
            "#SBATCH --gres=lscratch:20\n", 
            "#SBATCH --cpus-per-task=14\n"]

print(f"writing jobs for {n_slides:,} slides...")

for j in tqdm(range(n_slides)):
    job_file_j = f"get_features_s{j+1}.sh"
    job_txt_j  = job_txt + [
            f"#SBATCH --output={log_path}job_feat_s{j+1}_%j.out\n", 
            "#SBATCH --mail-type=FAIL,TIME_LIMIT_80\n", 
            "\n", 
            f'SCRIPT="{code_path + code_file}"\n', 
            f'PROJ="{dataset_name}"\n', 
            f'FEAT="{feature_name}"\n', 
            f'TILES="{tiles_files[j]}"\n', 
            f'WD="{_wpath_}"\n', 
            # f'LOG="{log_path}$PBS_JOBID.log"\n', 
            "\n", 
            "module load python/3.10\n", 
            'python $SCRIPT -dat=$PROJ -feat=$FEAT -fn=$TILES -wd=$WD -sv="y"']
    
    with open(job_path + job_file_j, "w") as file:
        file.writelines(job_txt_j)

print(f"done! job scripts are in = {job_path}\n")


#### ----------------------------------------------------------------
# %% submit jobs.
#### ----------------------------------------------------------------

if submit_jobs:
    print(f"submitting jobs for {n_slides:,} slides...")

    for j in range(n_slides):
        job_file_j   = job_path + f"get_features_s{j+1}.sh"
        job_submit_j = f"sbatch --job-name=feat_s{j+1} {job_file_j}"    # e.g.: sbatch --job-name=feat_s1 jobs/features/02Nov2023/get_features_s1.sh
        os.system(job_submit_j)                                         # execute commands

    print("done!\n")


#### ----------------------------------------------------------------
## to run on cmd: 
## python 1_12_jobs_to_get_features2.py -dat "CPTAC_BRCA" -feat "CTrans" -run "y"

