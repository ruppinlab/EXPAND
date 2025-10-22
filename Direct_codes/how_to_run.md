cmd prompts:

1. generate tiles: python 1_11_jobs_to_get_tiles.py -dat="TCGA_BRCA_FFPE" -fmt="svs" -mag="20" -date="18Jun2025" -run="y" -wd="/data/Ruppin_AI/Datasets/"

2. generate features: python 1_12_jobs_to_get_features2.py -dat="TCGA_BRCA_FFPE" -feat="ResNet" -date="18Jun2025" -run="y" -wd="/data/Ruppin_AI/Datasets/"

3. collect features for all slides: 1_13_jobs_to_collect_features2.py -dat="TCGA_BRCA_FFPE" -feat="ResNet" -mask="y" -date="18Jun2025" -run="y" -wd="/data/Ruppin_AI/Datasets/"
