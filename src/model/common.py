from .aux import ISIC_Aux_Model
from .base import ISIC_Base_Model
from .auxv2 import ISIC_AuxV2_Model


def get_model(cfg):
    match cfg.model_name:
        case "base":
            model = ISIC_Base_Model(cfg.encoder_name)
        case "aux":
            model = ISIC_Aux_Model(cfg.encoder_name, cfg)
        case "auxv2":
            model = ISIC_AuxV2_Model(cfg.encoder_name, cfg)
        case _:
            raise ValueError(f"Invalid Model Name: {cfg.model_name}")

    return model
