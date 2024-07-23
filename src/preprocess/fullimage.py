from glob import glob

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

from src.constants import ROOT_DIR, SAMPLE, TEST_CSV, TRAIN_DIR


def get_train_file_path(image_id):
    return f"{TRAIN_DIR}/{image_id}.jpg"


def train_fullimage_preprocess(cfg):
    train_images = sorted(glob(f"{TRAIN_DIR}/*.jpg"))

    df = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv")
    oe = OrdinalEncoder()
    categorical_cols = [
        "tbp_lv_location_simple",
        "tbp_lv_location",
        "tbp_tile_type",
        "attribution",
        "anatom_site_general",
        "sex",
        "copyright_license",
    ]
    df[categorical_cols] = oe.fit_transform(df[categorical_cols])

    df["file_path"] = df["isic_id"].apply(get_train_file_path)
    df = df[df["file_path"].isin(train_images)].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_fold)

    for fold, (_, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
        df.loc[val_, "kfold"] = int(fold)

    return df


def infer_fullimage_preprocess(cfg):
    df = pd.read_csv(TEST_CSV)
    df["target"] = 0  # dummy

    df_sub = pd.read_csv(SAMPLE)
    return df, df_sub
