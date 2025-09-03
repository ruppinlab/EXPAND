#!/usr/bin/env python
# coding: utf-8

# In[18]:


#### -------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Jan 24, 2025
#### Extract morphological features from H&E slides using HoVer-Net for multiple slides
#### --------------------------------------------------------------------------------------


import os
import bz2
import pickle
import subprocess
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

# Set working directory and paths
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.makedirs(_wpath_, exist_ok=True)
os.chdir(_wpath_)

# Add HoVer-Net to Python path
hovernet_path = "/data/Ruppin_AI/BRCA_PIF/Ranjan/Codes/hover_net/"
dataset_name = "TCGA_BRCA_FFPE"
slides_path = "/data/Ruppin_AI/Datasets/TCGA_BRCA_FFPE/outputs/tiles/"
output_base_dir = f"{dataset_name}/outputs/HoverNet/"

# Function to read .bz2 file
def read_bz2_file(tiles_path, tiles_file):
    with bz2.open(os.path.join(tiles_path, tiles_file), "rb") as file:
        slide_tiles_info, slide_tiles = pickle.load(file)
    return slide_tiles_info, slide_tiles

# Function to save tiles
def save_tiles_to_dir(slide_tiles, tile_path):
    os.makedirs(tile_path, exist_ok=True)
    for idx, tile in enumerate(slide_tiles):
        tile.save(os.path.join(tile_path, f"tile_{idx + 1}.png"))

# Function to run HoVer-Net inference
def run_hovernet_inference(tile_dir, output_dir, gpu_id="0,1", model_mode="fast", batch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    run_command = [
        "python",
        os.path.join(hovernet_path, "run_infer.py"),
        f"--gpu={gpu_id}",
        f"--nr_types=6",
        f"--type_info_path={os.path.join(hovernet_path, 'type_info.json')}",
        f"--batch_size={batch_size}",
        f"--model_mode={model_mode}",
        f"--model_path={os.path.join(_wpath_, 'model_hovernet/hovernet_fast_pannuke_type_tf2pytorch.tar')}",
        f"--nr_inference_workers=4",
        f"--nr_post_proc_workers=8",
        "tile",
        f"--input_dir={tile_dir}",
        f"--output_dir={output_dir}",
        f"--mem_usage=0.1",
        "--draw_dot",
        "--save_qupath"
    ]
    subprocess.run(run_command, check=True)

# Command-line arguments
parser = ArgumentParser()
parser.add_argument("-slide", type=str, required=True, help="Slide file name")
parser.add_argument("-tile_path", type=str, required=True, help="Path to tiles directory")
parser.add_argument("-wd", type=str, required=True, help="Working directory")
args = parser.parse_args()

slide_file = args.slide
tiles_path = args.tile_path
_wpath_ = args.wd

# Process the specified slide
slide_name = Path(slide_file).stem
output_dir = os.path.join(output_base_dir, slide_name)
tile_dir = os.path.join(output_dir, "tiles")
mask_dir = os.path.join(output_dir, "masks")

# Read and process the slide
slide_tiles_info, slide_tiles = read_bz2_file(tiles_path, slide_file)
save_tiles_to_dir(slide_tiles, tile_dir)
run_hovernet_inference(tile_dir, mask_dir, gpu_id="0,1", batch_size=8)


# In[ ]:




