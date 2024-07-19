import torch
import torch.nn as nn


def get_lossfn(cfg):
    weights = torch.tensor(cfg.loss_weight).cuda()
    match cfg.lossfn:
        case "BCELoss":
            return nn.BCELoss(weight=weights)
        case _:
            raise ValueError(f"Invalid Loss Function: {cfg.lossfn}")