from .base import infer_base_preprocess, train_base_preprocess


def get_train_preprocess(cfg):
    if cfg.preprocess == "base":
        return train_base_preprocess
    else:
        raise ValueError(f"Invalid Preprocess Name: {cfg.preprocess}")


def get_infer_preprocess(cfg):
    if cfg.preprocess == "base":
        return infer_base_preprocess
    else:
        raise ValueError(f"Invalid Preprocess Name: {cfg.preprocess}")
