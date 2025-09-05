#!/usr/bin/env python
# coding: utf-8

# In[18]:


#### --------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Aug 22, 2025
#### Final job submission code: Steps 2→4 (HoVer-Net → Morphology → NPIFs)
#### --------------------------------------------------------------------------------------

import os
import fnmatch
import time
import subprocess
from datetime import date
from tqdm import tqdm
from argparse import ArgumentParser

# =========================
# Static paths (edit if needed)
# =========================
_wpath_       = "/data/Lab_ruppin/Ranjan/HnE/"
dataset_name  = "TCGA_BRCA_FFPE"
tiles_path    = "/data/Ruppin_AI/Datasets/TCGA_BRCA_FFPE/outputs/tiles/"

# Your exact .py codes (update paths if different)
step2_code_path = "/data/Ruppin_AI/BRCA_PIF/Ranjan/Codes/EXPAND_Pipeline/"
step2_code_file = "2_01_22_ExtractMorphologicalFeaturesFromHnE.py"

step3_code_path = "/data/Ruppin_AI/BRCA_PIF/Ranjan/Codes/EXPAND_Pipeline/"
step3_code_file = "2_02_03_MorphologyCalculation_All_Slides_Pipeline.py"  # expects: -slide <SLIDE_NAME> and --out_dir

step4_code_path = "/data/Ruppin_AI/BRCA_PIF/Ranjan/Codes/EXPAND_Pipeline/"
step4_code_file = "2_03_01_01_NPIFs_Calculation_HoverNet_Pipeline.py"     # computes Top-25% internally

# Parameters (kept for reference)
mpp_value = 0.248

# SLURM resources
gpu_partition = "gpu"
gpu_gres      = "gpu:k80:2,lscratch:20"
gpu_cpus      = 8
gpu_mem       = "64g"
gpu_time      = "36:00:00"   # Step 2

cpu_partition = "norm"
cpu_cpus      = 8
cpu_mem       = "32g"
cpu_time_s3   = "06:00:00"   # Step 3
cpu_time_s4   = "04:00:00"   # Step 4

# Environment
conda_activate = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate hovernet"
module_lines   = ["module load CUDA/10.2", "module load gcc/13.2.0"]

# =========================
# CLI
# =========================
os.chdir(_wpath_)
parser = ArgumentParser(description="Submit SLURM jobs for Steps 2→4 (HoVer-Net→Morph→NPIFs) with strict dependencies.")
parser.add_argument("-run",   type=str, default="n", help="Submit jobs? [y/n]")
parser.add_argument("-date",  type=str, default=date.today().strftime("%d%b%Y"), help="Date stamp (e.g., 22Jan2025)")
parser.add_argument("-delay", type=int, default=5, help="Delay (sec) between job submissions")
parser.add_argument("-trial", type=int, default=1, help="Trial index for job folder naming")
args = parser.parse_args()

submit_jobs      = (args.run.lower() == "y")
datestamp        = args.date
submission_delay = args.delay
trial            = args.trial

# =========================
# Slide selection
# =========================
sample_prefixes = ["TCGA-AC-A23H", "TCGA-BH-A1EN", "TCGA-A2-A04X"]

all_files     = os.listdir(tiles_path)
matched_files = sorted({f for p in sample_prefixes for f in all_files if fnmatch.fnmatch(f, f"{p}*tiles.bz2")})

print("Matched Files:")
for f in matched_files:
    print(" ", f)

if not matched_files:
    raise FileNotFoundError("No matching *.tiles.bz2 found in the tiles directory.")

# =========================
# Job/Log directories
# =========================
job_path = f"{_wpath_}{dataset_name}/outputs/HoverNet/jobs/{datestamp}_{trial}/"
log_path = f"{_wpath_}{dataset_name}/outputs/HoverNet/jobs/logs/{datestamp}_{trial}/"
os.makedirs(job_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# Derived output roots for later steps
morph_root = f"{_wpath_}{dataset_name}/outputs/Morph/"
npifs_root = f"{_wpath_}{dataset_name}/outputs/NPIFs/"
os.makedirs(morph_root, exist_ok=True)
os.makedirs(npifs_root, exist_ok=True)

# Helpers
def sbatch(path, dependency=None):
    cmd = ["sbatch"]
    if dependency:
        cmd += [f"--dependency=afterok:{dependency}"]
    cmd += [path]
    out = subprocess.check_output(cmd, text=True).strip()
    print(out)
    # "Submitted batch job <id>"
    return out.split()[-1]

# =========================
# Write & submit Step 2 (GPU) and Step 3 (CPU) per slide with dependency
# =========================
s3_job_ids = []   # collect all Step-3 jobids for final Step-4 dependency

for idx, slide_file in enumerate(tqdm(matched_files), start=1):
    slide_name = os.path.splitext(slide_file)[0]

    # ---- Step 2 (GPU)
    s2_jobid = None  # CHANGED: define before use so we can test below
    s2_script_path = os.path.join(job_path, f"s2_{idx}_{slide_name}.sh")
    with open(s2_script_path, "w") as f:
        f.writelines([
            "#!/bin/bash\n",
            f"#SBATCH --partition={gpu_partition}\n",
            f"#SBATCH --gres={gpu_gres}\n",
            f"#SBATCH --cpus-per-task={gpu_cpus}\n",
            f"#SBATCH --mem={gpu_mem}\n",
            f"#SBATCH --time={gpu_time}\n",
            f"#SBATCH --job-name=s2_{slide_name}\n",
            f"#SBATCH --output={log_path}s2_{idx}_{slide_name}.out\n",
            f"#SBATCH --error={log_path}s2_{idx}_{slide_name}.err\n",
            "\n",
            "set -euo pipefail\n",
            f"{conda_activate}\n",
            *(line + "\n" for line in module_lines),
            "\n",
            f"python {os.path.join(step2_code_path, step2_code_file)} "
            f"-slide {slide_file} -tile_path {tiles_path} -wd {_wpath_}\n",
            # optional completion flag
            f"mkdir -p {_wpath_}{dataset_name}/outputs/HoverNet/{slide_name}/masks\n",
            f"touch {_wpath_}{dataset_name}/outputs/HoverNet/{slide_name}/masks/_PRED.done\n",
        ])
    if submit_jobs:
        s2_jobid = sbatch(s2_script_path)
        time.sleep(submission_delay)

    # ---- Guard for resume-only runs (do NOT check during fresh submission)
    pred_done = f"{_wpath_}{dataset_name}/outputs/HoverNet/{slide_name}/masks/_PRED.done"
    if not submit_jobs and not os.path.exists(pred_done):  # CHANGED: only guard when not submitting
        raise FileNotFoundError(
            f"Missing Step-2 flag: {pred_done}. Make sure Hover-Net finished for slide: {slide_name}"
        )

    # ---- Step 3 (CPU)  (write to Morph root directly)
    morph_out_dir = f"{morph_root}{slide_name}/"
    masks_dir     = f"{_wpath_}{dataset_name}/outputs/HoverNet/{slide_name}/masks"

    s3_script_path = os.path.join(job_path, f"s3_{idx}_{slide_name}.sh")
    with open(s3_script_path, "w") as f:
        f.writelines([
            "#!/bin/bash\n",
            f"#SBATCH --partition={cpu_partition}\n",
            f"#SBATCH --cpus-per-task={cpu_cpus}\n",
            f"#SBATCH --mem={cpu_mem}\n",
            f"#SBATCH --time={cpu_time_s3}\n",
            f"#SBATCH --job-name=s3_{slide_name}\n",
            f"#SBATCH --output={log_path}s3_{idx}_{slide_name}.out\n",
            f"#SBATCH --error={log_path}s3_{idx}_{slide_name}.err\n",
            "\n",
            "set -euo pipefail\n",
            f"{conda_activate}\n",
            "module load gcc/13.2.0\n",  # CPU-only; avoids Lmod swapping python with CUDA
            "\n",
            f"mkdir -p {morph_out_dir}\n",
            # Step-3 writes CSV to Morph/<slide>/tumor_nuclei.csv
            f"python {os.path.join(step3_code_path, step3_code_file)} -slide {slide_name} --out_dir {morph_root}\n",
        ])

    if submit_jobs:
        # CHANGED: depend on Step-2 so we don't need to check for flag here
        s3_jobid = sbatch(s3_script_path, dependency=s2_jobid)
        s3_job_ids.append(s3_jobid)
        time.sleep(submission_delay)

# =========================
# Write & submit Step 4 (CPU) after ALL Step-3 jobs complete
# =========================
npifs_25q_csv = os.path.join(npifs_root, "TCGA_BRCA_HoverNet_NPIFs_25Q.csv")

s4_script_path = os.path.join(job_path, "s4_npifs_25q.sh")  # renamed file for clarity
with open(s4_script_path, "w") as f:
    f.writelines([
        "#!/bin/bash\n",
        f"#SBATCH --partition={cpu_partition}\n",
        f"#SBATCH --cpus-per-task={cpu_cpus}\n",
        f"#SBATCH --mem={cpu_mem}\n",
        f"#SBATCH --time={cpu_time_s4}\n",
        f"#SBATCH --job-name=s4_npifs\n",
        f"#SBATCH --output={log_path}s4_npifs.out\n",
        f"#SBATCH --error={log_path}s4_npifs.err\n",
        "\n",
        "set -euo pipefail\n",
        f"{conda_activate}\n",
        "module load gcc/13.2.0\n",  # CPU-only
        "\n",
        "# ---------- NPIFs from TOP-25% tiles (default in the script) ----------\n",
        f"python {os.path.join(step4_code_path, step4_code_file)} "
        f"--morph_root {morph_root} --output_csv {npifs_25q_csv}\n",
    ])

if submit_jobs:
    if s3_job_ids:
        dep_all = ":".join(s3_job_ids)
        out = subprocess.check_output(["sbatch", f"--dependency=afterok:{dep_all}", s4_script_path], text=True).strip()
        print(out)
    else:
        print(f"Would submit: sbatch --dependency=afterok:<s3_job_ids> {s4_script_path}")

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




