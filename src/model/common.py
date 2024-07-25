from .base import ISIC_Base_Model


def get_model(cfg):
    auxtarget = getattr(cfg, "auxtarget", [])
    num_classes = len(auxtarget) + 1

    match cfg.model_name:
        case "base":
            model = ISIC_Base_Model(cfg.encoder_name, num_classes=num_classes)
        case _:
            raise ValueError(f"Invalid Model Name: {cfg.model_name}")

    return model
