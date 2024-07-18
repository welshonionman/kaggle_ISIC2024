from .base import infer_base_preprocess, train_base_preprocess
from .fullimage import infer_fullimage_preprocess, train_fullimage_preprocess


def get_train_preprocess(cfg):
    if cfg.preprocess == "base":
        return train_base_preprocess
    elif cfg.preprocess == "fullimage":
        return train_fullimage_preprocess
    else:
        raise ValueError(f"Invalid Preprocess Name: {cfg.preprocess}")


def get_infer_preprocess(cfg):
    if cfg.preprocess == "base":
        return infer_base_preprocess
    elif cfg.preprocess == "fullimage":
        return infer_fullimage_preprocess
    else:
        raise ValueError(f"Invalid Preprocess Name: {cfg.preprocess}")
