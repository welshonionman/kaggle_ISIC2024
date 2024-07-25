from glob import glob

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

from ..constants import CATEGORICAL_COLS, ROOT_DIR, SAMPLE, TEST_CSV, TRAIN_DIR
from .utils import get_train_file_path, impute_missing_values


def train_fullimage_preprocess(cfg):
    train_images = sorted(glob(f"{TRAIN_DIR}/*.jpg"))

    df = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv", low_memory=False)

    oe = OrdinalEncoder()
    df[CATEGORICAL_COLS] = oe.fit_transform(df[CATEGORICAL_COLS])
    df = impute_missing_values(df)

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
