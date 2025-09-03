#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### ------------------------------------------------------------------------------------------
#### author: Ranjan Barman, date: June 22, 2025
#### Compute molecular features (tumor/cancer only) from Hover-Net predicted json files (POST_NAT_BRCA)
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
dataset_name = "POST_NAT_BRCA"

# Parse slide argument
parser = ArgumentParser()
parser.add_argument("-slide", type=str, required=True, help="Slide folder name to process")
args = parser.parse_args()
slide_folder = args.slide

# Define the base directory containing all slides
hovernet_base_dir = f"{dataset_name}/HoverNet/outputs/"
slide_path = os.path.join(hovernet_base_dir, slide_folder)

if not os.path.isdir(slide_path):  
    print(f"Error: {slide_folder} is not a valid directory.")
    exit(1)

tiles_dir = os.path.join(slide_path, "tiles")
json_dir = os.path.join(slide_path, "masks/json")
overlay_dir = os.path.join(slide_path, "masks/overlay")
features_dir = os.path.join(slide_path, "features")
os.makedirs(features_dir, exist_ok=True)

output_file = os.path.join(features_dir, f"{slide_folder}.csv")

if not os.path.exists(json_dir):
    print(f"Skipping {slide_folder}: No JSON directory found.")
    exit(1)

json_files = sorted(
    [f for f in os.listdir(json_dir) if f.endswith(".json")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

results = []
for json_file in json_files:
    json_path = os.path.join(json_dir, json_file)
    tile_name = json_file.replace(".json", "")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    nuclei = data.get('nuc', {})
    tumor_nuclei = {key: val for key, val in nuclei.items() if val['type'] == 1}
    
    if tumor_nuclei:
        for key, nucleus in tumor_nuclei.items():
            nucleus_id = key
            contour = np.array(nucleus['contour'])
            polygon = Polygon(contour)
            area = polygon.area
            perimeter = polygon.length

            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
            else:
                major_axis = minor_axis = 0
            
            eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2)) if major_axis > 0 else 0
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            results.append([
                slide_folder, tile_name, nucleus_id,
                area, major_axis, minor_axis,
                perimeter, eccentricity, circularity
            ])

columns = ["Slide", "Tile", "Nucleus ID", "Area", "Major Axis", "Minor Axis", "Perimeter", "Eccentricity", "Circularity"]
df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv(output_file, mode='w', index=False, header=True)

print(f"Nuclear morphology features saved for slide {slide_folder} to {output_file}")


# In[ ]:




