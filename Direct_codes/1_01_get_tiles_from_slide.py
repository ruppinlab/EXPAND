#!/usr/bin/env python3
# coding: utf-8

#### ----------------------------------------------------------------
#### refactored code from Tai
#### author: dhrubas2, date: Oct 30, 2023
#### divide a H&E slide into tiles & keep viable tiles
#### ----------------------------------------------------------------

import os, pickle, bz2, openslide, czifile
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
from utils_preprocessing import evaluate_tile
from utils_color_norm import macenko_normalizer
from itertools import product
from math import ceil
from time import time
from tqdm import tqdm
from warnings import warn
from argparse import ArgumentParser

Image.MAX_IMAGE_PIXELS = None                      # to avoid DecompressionBombError


#### ----------------------------------------------------------------
#%%  functions.
#### ----------------------------------------------------------------

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
parser.add_argument("-fn", type = str, 
                    help = "filename of the slide to be processed")
parser.add_argument("-mag", type = str, default = "40", 
                    help = "default magnification to use when the information cannot be extracted from the slide file (for .czi, .tif, .jpg etc.)")
parser.add_argument("-wd", type = str, default = "/data/Lab_ruppin/dhrubas2/HnE/", 
                    help = "working directory; set this as the parent data directory where all datasets are saved")
parser.add_argument("-thmag", type = str, default = "15", 
                    help = "threshold for edge detection module (i.e., edge_mag_th)")
parser.add_argument("-thfrac", type = str, default = "0.5", 
                    help = "threshold for edge detection module (i.e., edge_frac_th)")
parser.add_argument("-svt", type = str, default = "y", 
                    help = "whether to save the tiles [y/n]")
parser.add_argument("-svm", type = str, default = "y", 
                    help = "whether to save the mask figure [y/n]")
args = parser.parse_args()

dataset_name   = args.dat
slide_file     = args.fn
mag_assumed    = int(args.mag)
_wpath_        = args.wd
edge_mag_th    = int(args.thmag)
edge_frac_th   = float(args.thfrac)
save_tile_file = (args.svt.lower() == "y")
save_mask_file = (args.svm.lower() == "y")


#### ----------------------------------------------------------------
#%%  data directories & files.
#### ----------------------------------------------------------------

## set working directory.
os.chdir(_wpath_)
print(f"\nworking directory = {_wpath_}\n")

## get filepaths & filenames.
slide_path = f"{dataset_name}/slides/"
slide_name = slide_file.split(".", maxsplit = 1)[0]          # slide name shorthand [slide name should not contain "_"]

## create directories to save masks & tiles.
mask_path = f"{dataset_name}/outputs/masks/"
tile_path = f"{dataset_name}/outputs/tiles/"
os.makedirs(mask_path, exist_ok = True)                      # creates directory if doesn't exist already
os.makedirs(tile_path, exist_ok = True)

print(f"""
dataset = {dataset_name}
slide path = {slide_path}
slide name = {slide_name}
save masks in = {mask_path}
save tiles in = {tile_path}
""")


#### ----------------------------------------------------------------
#%%  slide/tile parameters.
#### ----------------------------------------------------------------

slide_format   = slide_file.split(".")[1].lower()            # slide file format
mag_selected   = 20                                          # magnification level to use
tile_size      = 512
disp_dwnsmpl   = 16
disp_tile_size = ceil(tile_size / disp_dwnsmpl)
print(f"tile size = {tile_size}, display tile size = {disp_tile_size}")


#### ----------------------------------------------------------------
#%%  process each slide.
#### ----------------------------------------------------------------

print(f"""
current slide: {slide_name}
reading file = {slide_file}
""", end = "")
_dt_ = time()

## read slide image & get max. magnification.
if slide_format == "czi":                              # not supported by openslide: czi
    slide    = czifile.imread(slide_path + slide_file)
    if np.ndim(slide) == 6:      slide = slide[0][0][0]
    elif np.ndim(slide) == 5:    slide = slide[0][0]
    elif np.ndim(slide) == 4:    slide = slide[0]
    else:  raise ValueError("Unknown file shape! Please check the input file.")
    w_slide, h_slide = slide.shape[1], slide.shape[0]  # slide size at largest level
    
    vendor   = "zeiss"                                 # known vendor for CZI format
    warn(f"[WARNING] magnification not found! assuming: {mag_assumed}")
    mag_max  = mag_assumed                             # enter known magnification level as mag_assumed
    mag_orig = 0
elif slide_format in ["jpg", "jpeg"]:                  # not supported by openslide: jpg/jpeg 
    slide    = Image.open(slide_path + slide_file, mode = "r")
    w_slide, h_slide = slide.size                      # slide size at largest level
    slide    = np.array(slide)
    
    vendor   = "unknown"
    warn(f"[WARNING] magnification not found! assuming: {mag_assumed}")
    mag_max  = mag_assumed                             # enter known magnification level as mag_assumed
    mag_orig = 0
else:                                                  # supported by openslide: svs, ndpi, tif etc.
    slide  = openslide.OpenSlide(slide_path + slide_file)
    w_slide, h_slide = slide.level_dimensions[0]       # slide size at largest level (level = 0)
    
    vendor = slide.properties[openslide.PROPERTY_NAME_VENDOR]
    if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
        mag_max  = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        mag_orig = mag_max
    elif f"{vendor}.AppMag" in slide.properties:       # vendor-specific slot for magnification level
        mag_max  = float(slide.properties[f"{vendor}.AppMag"])
        mag_orig = mag_max
    else:
        warn(f"[WARNING] magnification not found! assuming: {mag_assumed}")
        mag_max  = mag_assumed                         # enter known magnification level as mag_assumed
        mag_orig = 0

dwnsmpl      = round(mag_max / mag_selected)           # downsampling level
tile_size_ds = tile_size * dwnsmpl

_dt_ = time() - _dt_
print(f"read image data! elapsed time = {format_time(_dt_)}.")

print(f"""
magnification info:
slide vendor = {vendor}
maximum value = {mag_max:0.2g} {'(assumed)' if not mag_orig else ''}
downsampling rate = {dwnsmpl}
tile size for downsampling = {tile_size_ds}
""", end = "")

## extract tile division from slide size.
n_rows,  n_cols  = ceil(h_slide / tile_size_ds), ceil(w_slide / tile_size_ds)
n_tiles_total    = n_rows * n_cols

print(f"""
size info:
height x width = {h_slide:,} x {w_slide:,}
#rows x #columns = {n_rows} x {n_cols}
total #tiles = {n_tiles_total:,}
""")


#### ----------------------------------------------------------------
#%%  examine each tile & keep relevant ones.
#### ----------------------------------------------------------------

color_norm       = macenko_normalizer()

## image masks for visual inspection.
disp_size        = (n_rows * disp_tile_size, n_cols * disp_tile_size, 3)
slide_disp_orig  = np.full(shape = disp_size, fill_value = 255, dtype = np.uint8)
slide_disp_clean = np.full(shape = disp_size, fill_value = 255, dtype = np.uint8)

## start tile processing.
print(f"""
evaluating {n_tiles_total:,} tiles for slide: {slide_name}...
edge_mag_th = {edge_mag_th}, edge_frac_th = {edge_frac_th}
""", end = "")
_dt_ = time()

slide_tiles = [ ];    slide_tiles_info = [ ]
for tile_i, (y, x) in tqdm(enumerate(product(*map(range, [n_rows, n_cols]))), 
                           total = n_tiles_total):
    ## read tile as RGB image.
    if not isinstance(slide, openslide.OpenSlide):
        tile = slide[(y * tile_size_ds):((y + 1) * tile_size_ds), 
                     (x * tile_size_ds):((x + 1) * tile_size_ds), :]
        tile = Image.fromarray(tile, mode = "RGB")
    else:
        tile = slide.read_region(location = (x * tile_size_ds, y * tile_size_ds), 
                                 size = (tile_size_ds, tile_size_ds), level = 0)
        tile = tile.convert(mode = "RGB")                # RGBA --> RGB

    ## begin tile processing.
    if tile.size == (tile_size_ds, tile_size_ds):
        ## downsample to target tile size.
        tile = tile.resize(size = (tile_size, tile_size))

        ## further downsample for display purposes.
        tile_disp = np.array(tile.resize(size = (disp_tile_size, disp_tile_size)))
        slide_disp_orig[(y * disp_tile_size):((y + 1) * disp_tile_size), 
                        (x * disp_tile_size):((x + 1) * disp_tile_size), :] = tile_disp

        ## evaluate tile to keep or discard.
        tile = np.array(tile)
        tile_select = evaluate_tile(tile, edge_mag_thrsh = edge_mag_th, 
                                    edge_fraction_thrsh = edge_frac_th)

        if tile_select:
            ## 2022.09.08: color normalization:
            tile_norm = Image.fromarray(color_norm.transform(tile))
            slide_tiles.append(tile_norm)

            ## save tile info as row-col-tile#-downsample.
            tile_info = f"tile_{str(y).zfill(5)}_{str(x).zfill(5)}_{str(tile_i).zfill(5)}_{str(dwnsmpl).zfill(3)}"                           
            slide_tiles_info.append(tile_info)

            ## further downsample for display purposes.
            tile_norm_disp = np.array(tile_norm.resize(size = (disp_tile_size, disp_tile_size)))
            slide_disp_clean[(y * disp_tile_size):((y + 1) * disp_tile_size), 
                             (x * disp_tile_size):((x + 1) * disp_tile_size), :] = tile_norm_disp

#### end loop ####

n_tiles = len(slide_tiles)
print(f"done! #tiles kept = {n_tiles:,}")

_dt_ = time() - _dt_
print(f"elapsed time = {format_time(_dt_)}.")


#### ----------------------------------------------------------------
#%% plot: draw color lines on the cleaned mask.
#### ----------------------------------------------------------------

print("completed cleaning! displaying side-by-side for inspection:")

## outline the tiles.
line_color = [0, 240, 0]
slide_disp_orig[:, ::disp_tile_size, :]  = line_color
slide_disp_orig[::disp_tile_size, :, :]  = line_color
slide_disp_clean[:, ::disp_tile_size, :] = line_color
slide_disp_clean[::disp_tile_size, :, :] = line_color


## make side-by-side plot of original & processed slides.
fontdict  = {
    "title": {"fontfamily": "sans", "fontsize": 24, "fontweight": "bold"}, 
    "super": {"fontfamily": "sans", "fontsize": 32, "fontweight": "bold"}}

titles = {"orig" : f"magnification: original = {mag_orig:0.4g}, used = {mag_max:0.4g}, display downsampling = {disp_dwnsmpl}", 
          "clean": f"#tiles: total = {n_tiles_total:,} ({n_rows} x {n_cols}), selected = {n_tiles:,}"}

sns.set_style("white")
fig, axs = plt.subplots(figsize = (40, 20), nrows = 1, ncols = 2)
axs[0].imshow(slide_disp_orig);     axs[0].axis("off");
axs[1].imshow(slide_disp_clean);    axs[1].axis("off");
axs[0].set_title(titles["orig"],  y = 1.005, **fontdict["title"])
axs[1].set_title(titles["clean"], y = 1.005, **fontdict["title"])
fig.suptitle(f"{dataset_name}: {slide_name}", y = 0.995, **fontdict["super"]);

fig.tight_layout(h_pad = 0.4, w_pad = 0.5);
plt.show()


#### ----------------------------------------------------------------
#%%  save data.
#### ----------------------------------------------------------------

## save selected tiles.
if save_tile_file:
    tile_data = (slide_tiles_info, slide_tiles)
    tile_file = f"{slide_name}_{len(slide_tiles)}_tiles.bz2"
    
    print(f"saving {n_tiles:,} tiles for slide: {slide_name}...")
    print("using bz2 compression (ext = .bz2)")
    with bz2.open(tile_path + tile_file, "wb") as file:
        pickle.dump(tile_data, file)        
    
    print(f"done! tiles saved in = {tile_path + tile_file}")

## save mask.
if save_mask_file:
    print("saving mask figure as PDF [DPI = 50]...")
    fig.savefig(mask_path + f"{slide_name}.pdf", format = "pdf", dpi = 50)
    plt.close(fig)
    print(f"done! figure is in = {mask_path + slide_name}.pdf")

