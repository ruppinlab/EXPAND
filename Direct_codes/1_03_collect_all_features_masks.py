#!/usr/bin/env python3
# coding: utf-8

#### ----------------------------------------------------------------
#### refactored code from Tai
#### author: dhrubas2, date: Nov 06, 2023
#### collect all features for the slides in a dataset into a file
#### collect all masks for the slides in a dataset into a file
#### ----------------------------------------------------------------

import os, sys, pickle, bz2
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from PyPDF2 import PdfReader, PdfMerger
from math import ceil, floor
from time import time
from tqdm import tqdm
from warnings import warn
from argparse import ArgumentParser


#### ----------------------------------------------------------------
#%%  functions.
#### ----------------------------------------------------------------

## functions.
def format_time(dt):
    dt = ceil(dt)
    dt = {"hr": dt // 3600, "min": (dt % 3600) // 60, "sec": dt % 60}
    dt = " ".join([f"{tv} {tk}" for tk, tv in dt.items() if tv > 0])
    return dt


#### ----------------------------------------------------------------
#%%  cmd arguments.
#### ----------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("-dat", type = str, 
                    help = "dataset to use i.e., folder name containing the slides in slides/ subdirectory")
parser.add_argument("-feat", type = str,
                    help = "feature type [ResNet / ViT / CTrans] for each slide to collate")
parser.add_argument("-wd", type = str, default = "/data/Lab_ruppin/dhrubas2/HnE/", 
                    help = "working directory; set this as the parent data directory where all datasets are saved")
parser.add_argument("-mask", type = str, default = "n", 
                    help = "whether to save all masks in a dataset in one single PDF file [y/n]")
parser.add_argument("-sv", type = str, default = "y", 
                    help = "whether to save the collated features [y/n]")
parser.add_argument("-svplt", type = str, default = "y", 
                    help = "whether to save tile count distribution plot [y/n]")
args = parser.parse_args()

dataset_name      = args.dat
feature_name      = args.feat.lower()
_wpath_           = args.wd
save_masks        = (args.mask.lower() == "y")
save_features     = (args.sv.lower() == "y")
save_distribution = (args.svplt.lower() == "y")


#### ----------------------------------------------------------------
#%%  get path & directories.
#### ----------------------------------------------------------------

## set working directory.
os.chdir(_wpath_)
print(f"working directory = {_wpath_}\n")

## get filepaths & filenames.
feature_path  = f"{dataset_name}/outputs/features/"
feature_files = sorted([fn for fn in os.listdir(feature_path) \
                        if feature_name.lower() in fn])
feature_files = [fn for fn in feature_files if ".npy" in fn]          # features for a single slide is saved as .npy

mask_path     = f"{dataset_name}/outputs/masks/"
mask_files    = sorted([fn for fn in os.listdir(mask_path) if ".pdf" in fn])
mask_files    = [fn for fn in mask_files if "_masks_" not in fn]      # exclude the merged file if exists already

## sanity check.
n_slides      = len(mask_files)
if len(feature_files) == n_slides:
    nf_slides = n_slides
else:
    # raise ValueError("Number of feature files and number of mask files are not the same!")
    warn(f"Number of feature files (n = {len(feature_files):,}) and number of mask files (n = {n_slides:,}) are not the same!")
    nf_slides = len(feature_files)

print(f"dataset = {dataset_name}, #slides = {n_slides:,}, #slides with features = {nf_slides:,}")


#### ----------------------------------------------------------------
#%%  collect features for all slides in a dataset & combine.
#### ----------------------------------------------------------------

print(f"""
collating {feature_name} features...
dataset = {dataset_name}
#slides = {n_slides:,}
#slides with features = {nf_slides:,}
""") 
_dt_ = time()

features_all, slide_names_all, n_tiles_all = [ ], [ ], [ ]
for feature_file in tqdm(feature_files):
    slide_name    = feature_file.split("_features")[0]
    slide_feature = np.load(feature_path + feature_file)
    slide_n_tiles = slide_feature.shape[0]
    slide_names_all.append( slide_name )
    features_all.append( slide_feature )
    n_tiles_all.append( slide_n_tiles )

_dt_ = time() - _dt_
print(f"done! elapsed time = {format_time(_dt_)}.")


#### ----------------------------------------------------------------
#%%  plot tiles distribution.
#### ----------------------------------------------------------------

print(f"""\n
tiles statistics:
#slides = {nf_slides:,}, min #tiles = {min(n_tiles_all):,}, max #tiles = {max(n_tiles_all):,}
plotting tile distributions:
""")

## plot parameters.
fontdict = {
    "label": {"fontfamily": "sans", "fontsize": 18, "fontweight": "regular"}, 
    "title": {"fontfamily": "sans", "fontsize": 24, "fontweight": "bold"}}

colors   = ["#75D0A6", "#000000"]

## get bin spacing: use min(1000, median #tiles) as spacing.
# bin_cuts = np.arange(0, max(n_tiles_all) + 1e3, 1e3)
cut_spc  = min(1e3, 100 * floor(np.median(n_tiles_all) / 100))
if cut_spc == 0:
    cut_spc = 100 * ceil(np.mean(n_tiles_all) / 100)
bin_cuts = np.arange(0, max(n_tiles_all) + cut_spc, cut_spc)

## make plot.
sns.set_style("ticks")
plt.rcParams.update({"xtick.major.size": 6, "xtick.major.width": 2, 
                     "ytick.major.size": 6, "ytick.major.width": 2, 
                     "xtick.bottom": True, "ytick.left": True, 
                     "axes.edgecolor": "#000000", "axes.linewidth": 2})

fig, ax  = plt.subplots(figsize = (10, 6), nrows = 1, ncols = 1)
counts, bins, patches = ax.hist(
    n_tiles_all, bins = bin_cuts, histtype = "bar", rwidth = 0.5, 
    linewidth = 2, color = colors[0], edgecolor = colors[1]);
ax.bar_label(patches, color = colors[1], **fontdict["label"]);
sns.despine(ax = ax, offset = 2, trim = False);

ax.set_xticks(ticks = bin_cuts, labels = bin_cuts.astype(int), 
              rotation = 45 if (len(bin_cuts) > 15) else 0, 
              ha = "center", va = "top", **fontdict["label"]);
ax.tick_params(axis = "both", labelsize = fontdict["label"]["fontsize"]);
ax.set_xlabel("Number of tiles", **fontdict["label"]);
ax.set_ylabel("Number of slides", **fontdict["label"]);
ax.set_title(f"Distribution of tile counts across {nf_slides:,} slides in {dataset_name.replace('_', '-')}", 
             wrap = True, **fontdict["title"]);

fig.tight_layout(h_pad = 0.4, w_pad = 0.5);
plt.show()


#### ----------------------------------------------------------------
#%%  save data.
#### ----------------------------------------------------------------

if save_features:
    feature_file = f"{dataset_name}_{feature_name}_features_all.bz2"
    feature_data = (slide_names_all, features_all)
    
    print(f"\nsaving {feature_name.lower()} features for {nf_slides:,} slides in {dataset_name}...")
    print("using bz2 compression")
    _dt_ = time()
    
    with bz2.open(feature_path + feature_file, "wb") as file:
        pickle.dump(feature_data, file)
    
    _dt_ = time() - _dt_
    print(f"done! elapsed time = {format_time(_dt_)}.")
    print(f"features saved in = {feature_path + feature_file}")


if save_distribution:
    fig_path = f"{dataset_name}/outputs/tiles/"
    fig_file = f"{dataset_name}_tile_distribution.pdf"
    
    print(f"\nsaving tile distribution plot for {nf_slides:,} slides in {dataset_name} as PDF [DPI = 50]...")
    fig.savefig(fig_path + fig_file, format = "pdf", dpi = 50)
    plt.close(fig)    
    print(f"done! figure is in = {fig_path + fig_file}")


#### ----------------------------------------------------------------
#%%  collecting masks for all slides & combine.
#### ----------------------------------------------------------------

if save_masks:
    print(f"collating tile masks for all {n_slides:,} slides in {dataset_name}...") 
    _dt_ = time()

    masks_slides_all = PdfMerger()
    for mask_file in tqdm(mask_files):
        mask_slide = PdfReader(mask_path + mask_file, "rb")
        masks_slides_all.append( mask_slide )    

    _dt_ = time() - _dt_
    print(f"done! elapsed time = {format_time(_dt_)}.")


#### ----------------------------------------------------------------
#%%  save data.
#### ----------------------------------------------------------------

if save_masks:
    print(f"\nsaving all {n_slides:,} masks for {dataset_name} in a single file...")
    _dt_ = time()
    
    mask_file = f"{dataset_name}_masks_all.pdf"
    masks_slides_all.write(mask_path + mask_file)
    
    _dt_ = time() - _dt_
    print(f"done! elapsed time = {format_time(_dt_)}.")
    print(f"masks saved in = {mask_path + mask_file}")

