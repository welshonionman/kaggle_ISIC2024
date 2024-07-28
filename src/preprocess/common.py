from .base_negative_sampling import infer_base_preprocess, train_base_preprocess
from .fullimage import infer_fullimage_preprocess, train_fullimage_preprocess


def get_train_preprocess(cfg):
    match cfg.preprocess:
        case "base_negative_sampling":
            return train_base_preprocess
        case "fullimage":
            return train_fullimage_preprocess
        case _:
            raise ValueError(f"Invalid Preprocess Name: {cfg.preprocess}")


def get_infer_preprocess(cfg):
    match cfg.preprocess:
        case "base_negative_sampling":
            return infer_base_preprocess
        case "fullimage":
            return infer_fullimage_preprocess
        case _:
            raise ValueError(f"Invalid Preprocess Name: {cfg.preprocess}")
