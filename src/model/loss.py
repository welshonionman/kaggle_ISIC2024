import torch.nn as nn


def get_lossfn(cfg):
    if cfg.lossfn == "BCELoss":
        return nn.BCELoss()
    else:
        raise ValueError(f"Invalid Loss Function: {cfg.lossfn}")
