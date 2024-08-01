from glob import glob

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from ..constants import CATEGORICAL_COLS, ROOT_DIR, SAMPLE, TEST_CSV, TRAIN_DIR
from .utils import get_train_file_path, impute_missing_values


def train_extended_preprocess(cfg):
    # originalデータセット内の陽性サンプルは全てvalidationに使う
    # validationは全てoriginalデータセットから取得
    image_dir = getattr(cfg, "image_dir", TRAIN_DIR)
    meta_path = getattr(cfg, "meta_path", f"{ROOT_DIR}/train-metadata.csv")

    train_images = sorted(glob(f"{image_dir}/*.jpg"))

    df = pd.read_csv(meta_path, low_memory=False)

    oe = OrdinalEncoder()
    df[CATEGORICAL_COLS] = oe.fit_transform(df[CATEGORICAL_COLS])
    df = impute_missing_values(df)

    df_arch_pos = df[(df["source"] == "archive") & (df["target"] == 1)]
    df_arch_pos.loc[:, ["kfold"]] = 1  # train

    df_orig_pos = df[(df["source"] == "orig") & (df["target"] == 1)]
    df_orig_pos.loc[:, ["kfold"]] = 0  # validation

    df_neg = df[df["target"] == 0]
    train_neg_sample_num = len(df_arch_pos) * cfg.sampling_factor
    valid_neg_sample_num = len(df_orig_pos) * cfg.sampling_factor
    df_train_neg = df_neg.sample(train_neg_sample_num).reset_index(drop=True)
    df_valid_neg = df_neg.sample(valid_neg_sample_num).reset_index(drop=True)
    df_train_neg.loc[:, ["kfold"]] = 1  # train
    df_valid_neg.loc[:, ["kfold"]] = 0  # validation

    df = pd.concat(
        [df_arch_pos, df_orig_pos, df_train_neg, df_valid_neg], ignore_index=True
    )

    df["file_path"] = df["isic_id"].apply(get_train_file_path, cfg=cfg)
    df = df[df["file_path"].isin(train_images)].reset_index(drop=True)

    return df


def infer_extended_preprocess(cfg):
    df = pd.read_csv(TEST_CSV)
    df["target"] = 0  # dummy

    df_sub = pd.read_csv(SAMPLE)
    return df, df_sub
