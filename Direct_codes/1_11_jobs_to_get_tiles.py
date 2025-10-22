#!/usr/bin/env python3
# coding: utf-8

#### ----------------------------------------------------------------
#### refactored job submission script
#### author: dhrubas2, date: Oct 31, 2023
#### extract tiles from all slides in a dataset
#### ----------------------------------------------------------------

import os
from datetime import date
from tqdm import tqdm
from argparse import ArgumentParser

# _wpath_ = "/data/Lab_ruppin/dhrubas2/HnE/"        # set working directory as the parent directory where all datasets are saved
# os.chdir(_wpath_)


#### ----------------------------------------------------------------
#%%  cmd arguments.
#### ----------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("-dat", type = str, 
                    help = "dataset to use i.e., folder name containing the slides inside slides/ subdirectory")
parser.add_argument("-fmt", type = str, default = "svs", 
                    help = "digital slide format e.g., svs, tif, ndpi etc. separate multiple slide formats by ; e.g., ndpi;svs etc.")
parser.add_argument("-mag", type = str, default = "40", 
                    help = "default magnification to use when the information cannot be extracted from the slide file (for .czi, .tif etc.)")
parser.add_argument("-wd", type = str, default = "/data/Lab_ruppin/dhrubas2/HnE/", 
                    help = "working directory - set this as the parent directory where all datasets are stored")
parser.add_argument("-thmag", type = str, default = "15", 
                    help = "threshold for edge detection module (i.e., edge_mag_th)")
parser.add_argument("-thfrac", type = str, default = "0.5", 
                    help = "threshold for edge detection module (i.e., edge_frac_th)")
parser.add_argument("-run", type = str, default = "n", 
                    help = "whether to submit jobs for running [y/n]")
parser.add_argument("-date", type = str, default = date.today().strftime("%d%b%Y"), 
                    help = "date stamp (in dd-mmm-YYYY format) for analysis results/logs e.g., 14Feb2024")
args = parser.parse_args()


dataset_name = args.dat
file_format  = args.fmt
mag_assumed  = args.mag
_wpath_      = args.wd
mag_th       = args.thmag
frac_th      = args.thfrac
submit_jobs  = (args.run.lower() == "y")
datestamp    = args.date

os.chdir(_wpath_)


#### ----------------------------------------------------------------
#%%  get path & directories.
#### ----------------------------------------------------------------

slide_path  = f"{dataset_name}/slides/"
# slide_files = sorted([fn for fn in os.listdir(slide_path) if f".{file_format}" in fn])
slide_files = sorted([fn for fn in os.listdir(slide_path) if any(
    [f".{fmt}" in fn for fmt in file_format.split(";")])])      


code_path   = "/data/Lab_ruppin/dhrubas2/HistologyToProteomics/analysis/"
code_file   = "1_01_get_tiles_from_slide.py"

## create job paths.
job_path    = f"{_wpath_}{dataset_name}/jobs/tiles/{datestamp}/"
log_path    = f"{_wpath_}{dataset_name}/jobs/logs/tiles/{datestamp}/"
os.makedirs(job_path, exist_ok = True)                      # creates directory if doesn't exist already
os.makedirs(log_path, exist_ok = True)


#### ----------------------------------------------------------------
#%%  write jobs.
#### ----------------------------------------------------------------

n_slides = len(slide_files)

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
    job_file_j = f"get_tiles_s{j+1}.sh"
    job_txt_j  = job_txt + [
            f"#SBATCH --output={log_path}job_tiles_s{j+1}_%j.out\n", 
            "#SBATCH --mail-type=FAIL,TIME_LIMIT_80\n", 
            "\n", 
            f'SCRIPT="{code_path + code_file}"\n', 
            f'PROJ="{dataset_name}"\n', 
            f'SLIDE="{slide_files[j]}"\n', 
            f'MAG="{mag_assumed}"\n', 
            f'WD="{_wpath_}"\n', 
            f'THMAG={mag_th}\n', 
            f'THFRAC={frac_th}\n', 
            # f'LOG="{log_path}$PBS_JOBID.log"\n', 
            "\n", 
            "module load OpenSlide python/3.10\n", 
            'python $SCRIPT -dat=$PROJ -fn=$SLIDE -mag=$MAG -wd=$WD -thmag=$THMAG -thfrac=$THFRAC -svt="y" -svm="y"']
    
    with open(job_path + job_file_j, "w") as file:
        file.writelines(job_txt_j)

print(f"done! job scripts are in = {job_path}\n")


#### ----------------------------------------------------------------
#%%  submit jobs.
#### ----------------------------------------------------------------

if submit_jobs:
    print(f"submitting jobs for {n_slides:,} slides...")

    for j in range(n_slides):
        job_file_j   = job_path + f"get_tiles_s{j+1}.sh"
        job_submit_j = f"sbatch --job-name=tile_s{j+1} {job_file_j}"    # e.g.: sbatch --job-name=tile_s1 jobs/tiles/31Oct2023/get_tiles_s1.sh
        os.system(job_submit_j)                                         # execute commands

    print("done!\n")


#### ----------------------------------------------------------------
## to run on cmd: 
## python 1_11_jobs_to_get_tiles.py -dat="Wirtz_MIBC" -fmt="svs" -run="y"

