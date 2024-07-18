import torch
import torch.nn as nn


def get_lossfn(cfg):
    weights = torch.tensor(cfg.loss_weight).cuda()
    if cfg.lossfn == "BCELoss":
        return nn.BCELoss(weight=weights)
    else:
        raise ValueError(f"Invalid Loss Function: {cfg.lossfn}")
