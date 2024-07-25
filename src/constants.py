import torch

COMP_NAME = "ISIC2024"
ROOT_DIR = "/kaggle/input/isic-2024-challenge"
TRAIN_DIR = f"{ROOT_DIR}/train-image/image"
TEST_CSV = f"{ROOT_DIR}/test-metadata.csv"
TEST_HDF = f"{ROOT_DIR}/test-image.hdf5"
SAMPLE = f"{ROOT_DIR}/sample_submission.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORICAL_COLS = [
    "tbp_lv_location_simple",
    "tbp_lv_location",
    "tbp_tile_type",
    "attribution",
    "anatom_site_general",
    "sex",
    "copyright_license",
]
