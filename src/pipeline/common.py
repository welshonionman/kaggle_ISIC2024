from .base import base_infer_pipeline, base_train_pipeline


def get_train_pipeline(cfg):
    if cfg.pipeline == "base":
        return base_train_pipeline
    else:
        raise ValueError(f"Invalid Pipeline Name: {cfg.pipeline}")


def get_infer_pipeline(cfg):
    if cfg.pipeline == "base":
        return base_infer_pipeline
    else:
        raise ValueError(f"Invalid Pipeline Name: {cfg.pipeline}")
