from torch import optim


def get_optimizer(model, cfg):
    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

    else:
        raise ValueError(f"Invalid Optimizer: {cfg.optimizer}")
    return optimizer
