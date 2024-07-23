from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.constants import COMP_NAME, DEVICE
from src.dataset import get_test_dataloader, get_train_dataloader
from src.model import get_lossfn, get_model, get_optimizer, get_scheduler
from src.preprocess import get_infer_preprocess, get_train_preprocess

from .epoch import epoch_end, train_1epoch, valid_1epoch
from .utils import set_seed


def base_train_pipeline(cfg):
    set_seed(cfg.seed)

    df = get_train_preprocess(cfg)(cfg)
    train_loader, valid_loader = get_train_dataloader(df, 0, cfg)

    model = get_model(cfg).to(DEVICE)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    best_score = 0

    wandb.init(
        project=COMP_NAME, name=cfg.exp_name, dir="/workspace/", mode=cfg.wandb_mode
    )

    for epoch in range(cfg.epochs):
        train_loss = train_1epoch(model, train_loader, optimizer, scheduler, epoch, cfg)
        valid_loss, score = valid_1epoch(model, valid_loader, epoch, cfg)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "score": score,
            }
        )

        model_path = Path(f"/kaggle/weights/{cfg.exp_name}/{cfg.exp_name}.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        best_score = epoch_end(valid_loss, best_score, score, model, model_path)


def base_infer_pipeline(cfg):
    set_seed(cfg.seed)

    df, df_sub = get_infer_preprocess(cfg)(cfg)
    test_loader = get_test_dataloader(df, cfg)

    model = get_model(cfg)
    model.load_state_dict(torch.load(cfg.weight_path)).to(DEVICE)

    preds = []
    with torch.no_grad():
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for step, data in bar:
            images = data["image"].to(cfg["device"], dtype=torch.float)
            outputs = model(images)
            preds.append(outputs.detach().cpu().numpy())
    preds = np.concatenate(preds).flatten()
    df_sub["target"] = preds
    df_sub.to_csv("submission.csv", index=False)
