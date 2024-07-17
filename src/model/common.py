from .base import ISIC_Base_Model


def get_model(cfg):
    if cfg.model_name == "base":
        model = ISIC_Base_Model(cfg.encoder_name)
    else:
        raise ValueError(f"Invalid Model Name: {cfg.model_name}")

    return model
