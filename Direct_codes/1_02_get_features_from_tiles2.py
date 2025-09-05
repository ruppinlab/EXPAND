#!/usr/bin/env python3
# coding: utf-8

#### ----------------------------------------------------------------
#### refactored code from Tai
#### author: dhrubas2, date: Nov 02, 2023
#### extract image features from each tile of a H&E slide
#### ----------------------------------------------------------------

import os, sys, pickle, bz2
import numpy as np, pandas as pd
from PIL import Image
from math import ceil
from time import time
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from utils_preprocessing import set_device, set_random_state, ResNetModel
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from ctrans_model import CTransPath


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
                    help = "feature type to extract from the tiles [ResNet / ViT / CTrans]")
parser.add_argument("-fn", type = str, 
                    help = "filename of the tiles file for the slide to be processed")
parser.add_argument("-wd", type = str, default = "/data/Lab_ruppin/dhrubas2/HnE/", 
                    help = "working directory; set this as the parent data directory where all datasets are saved")
parser.add_argument("-sv", type = str, default = "y", 
                    help = "whether to save the extracted features [y/n]")
args = parser.parse_args()

dataset_name  = args.dat
feature_name  = args.feat
tiles_file    = args.fn
_wpath_       = args.wd
save_features = (args.sv.lower() == "y")


#### ----------------------------------------------------------------
#%%  data directories & files.
#### ----------------------------------------------------------------

## set working directory.
os.chdir(_wpath_)
print(f"working directory = {_wpath_}\n")

## init torch functions.
device = set_device()
set_random_state(seed = 42)                                # seed for reproducibility

## get filepaths & filenames.
tiles_path = f"{dataset_name}/outputs/tiles/"
# slide_name = tiles_file.split("_", maxsplit = 1)[0]        # slide name shorthand [slide name should not contain "_"]
slide_name   = "_".join(tiles_file.split("_")[:-2])        # slide name shorthand
n_tiles    = int(tiles_file.split("_")[-2])                # tile filename: {slide_name}_{n_tiles}_tiles.bz2

## create directories to save features.
feature_path = f"{dataset_name}/outputs/features/"
os.makedirs(feature_path, exist_ok = True)                 # creates directory if doesn't exist already

print(f"""
dataset          = {dataset_name}
slide name       = {slide_name}
tiles path       = {tiles_path}
#tiles           = {n_tiles:,}
feature type     = {feature_name}
save features in = {feature_path}
""")


#### ----------------------------------------------------------------
#%%  load tiles for each slide.
#### ----------------------------------------------------------------

print(f"reading {n_tiles:,} tiles for slide: {slide_name}...")
_dt_ = time()

with bz2.open(tiles_path + tiles_file, "rb") as file:
    slide_tiles_info, slide_tiles = pickle.load(file)

_dt_ = time() - _dt_
print(f"done! loading time = {format_time(_dt_)}.")


#### ----------------------------------------------------------------
#%%  feature extraction function.
#### ----------------------------------------------------------------

def extract_features_from_tiles(tiles_list, use_model = "ResNet", batch_size = 64):
    ## load pretrained model.
    if "resnet" in use_model.lower():
        use_model  = "ResNet50"
        # model_path = "/data/Lab_ruppin/dhrubas2/HistologyToProteomics/codes/ResNet50_IMAGENET1K_V2.pt"
        # model = ResNetModel(model_type = "load_from_saved_file")
        # model.load_state_dict(torch.load(model_path, map_location = device))
        model = ResNetModel(model_type = "load_from_internet")
        # model.to(device)
    elif "vit" in use_model.lower():
        use_model  = "Vision Transformer"
        model_path = "google/vit-base-patch16-224-in21k"
        processor  = ViTImageProcessor.from_pretrained(model_path)
        model      = ViTModel.from_pretrained(model_path, return_dict = True)
        # model.to(device)
    elif "ctrans" in use_model.lower():
        use_model  = "CTransPath"
        model_path = "/data/Lab_ruppin/dhrubas2/HistologyToProteomics/analysis/ctranspath.pth"
        model      = CTransPath(num_classes = 0)
        # model.to(device)
        # weights    = torch.load(model_path)["model"]
        model.load_state_dict( torch.load(model_path, map_location = device)["model"] )
    else:
        raise ValueError("Undefined model! Use either 'ResNet', 'ViT', or 'CTrans' for feature extraction.")
        
    model.eval()                                           # inference mode
    print(f"\nusing feature extraction model = {use_model}")
    
    ## preprocess tiles by ImageNet specifications.
    im_size = 224
    im_stat = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    transform_list = [transforms.Resize(im_size), 
                      transforms.ToTensor(), 
                      transforms.Normalize(**im_stat)]
    
    n_tiles = len(tiles_list)
    print(f"preparing {n_tiles:,} tiles...")
    _dt_ = time()
    
    if use_model == "ResNet50":
        ## resize tiles & normalize by ImageNet mean & std.
        tile_transform = transforms.Compose(transform_list)
        tiles_list_im  = [tile_transform(tile).unsqueeze(0) for tile in tiles_list]
        tiles_list_im  = torch.cat(tiles_list_im, dim = 0)
    elif use_model == "Vision Transformer":
        ## resize tiles & preprocess by ViT standards.
        tile_transform = transforms.Compose(transform_list[:1])
        tiles_list_im  = [tile_transform(tile) for tile in tiles_list]
        tiles_list_im  = processor(tiles_list, return_tensors = "pt")["pixel_values"]
    elif use_model == "CTransPath":
        ## resize tiles & normalize by ImageNet mean & std.
        tile_transform = transforms.Compose(transform_list)
        tiles_list_im  = [tile_transform(tile).unsqueeze(0) for tile in tiles_list]
        tiles_list_im  = torch.cat(tiles_list_im, dim = 0)
    
    _dt_ = time() - _dt_
    print(f"done! elapsed time = {format_time(_dt_)}.\n")
    
    ## extract features from each tile.
    print(f"extracting features for {n_tiles:,} tiles...")
    print(f"using batch size = {batch_size}, #batches = {ceil(n_tiles / batch_size):,}")
    _dt_ = time()
    
    features_list = [ ]
    for idx_start in tqdm(range(0, n_tiles, batch_size)):
        idx_end = idx_start + min(batch_size, n_tiles - idx_start)
        
        # feature = model( tiles_list_im[idx_start : idx_end] )
        with torch.no_grad():
            feature = model( tiles_list_im[idx_start : idx_end] )
        if use_model == "Vision Transformer":
            feature = feature.last_hidden_state[:, 0]
        
        features_list.append( feature.detach().cpu().numpy() )
    
    features_list = np.concatenate(features_list)
    
    _dt_ = time() - _dt_
    print(f"done! elapsed time = {format_time(_dt_)}.")
    print(f"feature size = {features_list.shape})")
    
    return features_list


#### ----------------------------------------------------------------
#%%  extract features from each tiles of each slide.
#### ----------------------------------------------------------------

print(f"feature extraction for slide: {slide_name}...")
print(f"feature type = {feature_name}")

batch_size   = 64
features_all = extract_features_from_tiles(
    slide_tiles, use_model = feature_name, batch_size = batch_size)

print("\ndone!")


#### ----------------------------------------------------------------
#%%  save data.
#### ----------------------------------------------------------------

## save extracted features.
if save_features:
    feature_file = f"{slide_name}_features_{feature_name.lower()}.npy"
    
    print(f"saving {feature_name} features for slide: {slide_name}...")
    np.save(feature_path + feature_file, features_all)
    
    print(f"done! features saved in:\n{feature_path + feature_file}")

