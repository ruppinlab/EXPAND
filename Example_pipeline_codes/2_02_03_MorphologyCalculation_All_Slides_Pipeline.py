#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: Aug 22, 2025
#### Compute molecular features (tumor/cancer only) from Hover-Net predicted json files (all slides)
#### --------------------------------------------------------------------------------------------

import os
import json
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from argparse import ArgumentParser

# Set working directory
_wpath_ = "/data/Lab_ruppin/Ranjan/HnE/"
os.makedirs(_wpath_, exist_ok=True)
os.chdir(_wpath_)

print("Working directory:", _wpath_)

# Dataset name
dataset_name = "TCGA_BRCA_FFPE"

# Parse args
parser = ArgumentParser()
parser.add_argument("-slide", type=str, required=True, help="Slide folder name to process")
# NEW: where Hover-Net outputs live (jsons under <hovernet_root>/<slide>/masks/json)
parser.add_argument(
    "--hovernet_root",
    type=str,
    default=os.path.join(_wpath_, dataset_name, "outputs", "HoverNet") + "/",
    help="Root directory containing <slide>/masks/json from HoVer-Net"
)
# NEW: where to save morphology results (Morph/<slide>/tumor_nuclei.csv)
parser.add_argument(
    "--out_dir",
    type=str,
    default=os.path.join(_wpath_, dataset_name, "outputs", "Morph") + "/",
    help="Base directory to write morphology CSVs (per slide subfolder)"
)
args = parser.parse_args()

slide_folder = args.slide

# Base dirs (NOW come from args)
hovernet_base_dir = args.hovernet_root  # CHANGED
slide_path = os.path.join(hovernet_base_dir, slide_folder)

if not os.path.isdir(slide_path):
    print(f"Error: {slide_folder} is not a valid directory under {hovernet_base_dir}")
    exit(1)

tiles_dir    = os.path.join(slide_path, "tiles")
json_dir     = os.path.join(slide_path, "masks", "json")
overlay_dir  = os.path.join(slide_path, "masks", "overlay")

# Output dir/file (write directly to Morph/<slide>/tumor_nuclei.csv)
features_dir = os.path.join(args.out_dir, slide_folder)  # CHANGED
os.makedirs(features_dir, exist_ok=True)
output_file  = os.path.join(features_dir, "tumor_nuclei.csv")  # CHANGED

# List all JSON files
if not os.path.exists(json_dir):
    print(f"Skipping {slide_folder}: No JSON directory found at {json_dir}.")
    exit(1)

json_files = sorted(
    [f for f in os.listdir(json_dir) if f.endswith(".json")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

# Process each tile
results = []
for json_file in json_files:
    json_path = os.path.join(json_dir, json_file)
    tile_name = json_file.replace(".json", "")

    # Read JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter nuclei labeled as tumor (type=1)
    nuclei = data.get('nuc', {})
    tumor_nuclei = {key: val for key, val in nuclei.items() if val.get('type') == 1}

    if tumor_nuclei:
        for key, nucleus in tumor_nuclei.items():
            nucleus_id = key

            # Ensure proper numpy formats
            contour_pts = np.asarray(nucleus['contour'], dtype=np.float32)  # Nx2
            polygon = Polygon(contour_pts)
            area = polygon.area
            perimeter = polygon.length

            # Fit ellipse (cv2 needs shape Nx1x2)
            if contour_pts.shape[0] >= 5:
                ellipse = cv2.fitEllipse(contour_pts.reshape(-1, 1, 2))
                major_axis = float(max(ellipse[1]))
                minor_axis = float(min(ellipse[1]))
            else:
                major_axis = minor_axis = 0.0

            # Eccentricity
            if major_axis > 0:
                eccentricity = float(np.sqrt(1.0 - (minor_axis ** 2) / (major_axis ** 2)))
            else:
                eccentricity = 0.0

            # Circularity
            if perimeter > 0:
                circularity = float((4.0 * np.pi * area) / (perimeter ** 2))
            else:
                circularity = 0.0

            results.append([
                slide_folder, tile_name, nucleus_id,
                area, major_axis, minor_axis, perimeter, eccentricity, circularity
            ])

# To DataFrame & save
columns = ["Slide", "Tile", "Nucleus ID", "Area", "Major Axis", "Minor Axis", "Perimeter", "Eccentricity", "Circularity"]
df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv(output_file, mode='w', index=False, header=True)

print(f"Nuclear morphology features saved for slide {slide_folder} to {output_file}")


# In[ ]:




