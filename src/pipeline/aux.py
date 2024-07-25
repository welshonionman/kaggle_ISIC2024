from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from src.constants import COMP_NAME, DEVICE
from src.dataset import get_test_dataloader, get_train_dataloader
from src.model import get_lossfn, get_model, get_optimizer, get_scheduler
from src.preprocess import get_infer_preprocess, get_train_preprocess

from .metrics import calc_score
from .utils import set_seed


def pbar_train_desc(pbar_train, scheduler, epoch, total_epoch, average_loss):
    learning_rate = f"LR : {scheduler.get_last_lr()[0]:.2E}"
    gpu_memory = f"Mem : {torch.cuda.memory_reserved() / 1E9:.3g}GB"
    epoch_info = f"Epoch {epoch}/{total_epoch}"
    loss = f"Loss: {average_loss:.4f}"

    description = f"{epoch_info:12} {gpu_memory:15} {learning_rate:15} {loss:15}"
    pbar_train.set_description(description)


def pbar_valid_desc(pbar_val, average_loss):
    average_val_loss = f"Val Loss: {average_loss:.4f}"

    description = f"{average_val_loss:18}"
    pbar_val.set_description(description)


def train_1epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    auxtarget = getattr(cfg, "auxtarget", [])
    train_loss = 0

    model.train()

    pbar_train = enumerate(train_loader)
    pbar_train = tqdm(
        pbar_train,
        total=len(train_loader),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-0b}",
    )

    for batch_idx, batch in pbar_train:
        optimizer.zero_grad()

        inputs = batch["image"].to(DEVICE, dtype=torch.float)
        labels = batch["target"]
        for key, value in labels.items():
            labels[key] = value.to(DEVICE, dtype=torch.float)

        outputs = model(inputs)

        criterion_mal = nn.BCELoss()
        criterion_sex = nn.BCELoss()
        # criterion_age = nn.BCELoss()
        # criterion_site = nn.BCELoss()

        loss_mal = criterion_mal(outputs["malignant"].squeeze(), labels["malignant"])
        loss_sex = criterion_sex(outputs["sex"].squeeze(), labels["sex"])
        # loss_age = criterion_age(outputs[:, [1]], aux_labels)
        # loss_site = criterion_site(outputs[:, [1]], aux_labels)

        loss_sum = loss_mal + loss_sex
        loss_sum.backward()

        optimizer.step()

        train_loss += loss_sum.item()
        average_loss = train_loss / (batch_idx + 1)

        pbar_train_desc(pbar_train, scheduler, epoch, cfg.epochs, average_loss)
    scheduler.step()
    return average_loss


def valid_1epoch(model, valid_loader, epoch, cfg):
    auxtarget = getattr(cfg, "auxtarget", [])
    valid_loss = 0
    all_true = torch.tensor([], device=DEVICE)
    all_outputs = torch.tensor([], device=DEVICE)
    model.eval()

    pbar_val = enumerate(valid_loader)
    pbar_val = tqdm(
        pbar_val,
        total=len(valid_loader),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )

    with torch.inference_mode():
        for batch_idx, batch in pbar_val:
            inputs = batch["image"].to(DEVICE, dtype=torch.float)
            labels = batch["target"]
            for key, value in labels.items():
                labels[key] = value.to(DEVICE, dtype=torch.float)

            outputs = model(inputs)

            # criterion_mal = get_lossfn(cfg, labels)
            criterion_mal = nn.BCELoss()
            criterion_sex = nn.BCELoss()

            loss_mal = criterion_mal(
                outputs["malignant"].squeeze(), labels["malignant"]
            )
            loss_sex = criterion_sex(outputs["sex"].squeeze(), labels["sex"])
            loss_sum = loss_mal + loss_sex
            valid_loss += loss_sum
            all_true = torch.cat((all_true, labels["malignant"].squeeze()), 0)
            all_outputs = torch.cat((all_outputs, outputs["malignant"].squeeze()), 0)

            average_loss = valid_loss / (batch_idx + 1)
            pbar_valid_desc(pbar_val, average_loss)

    score = calc_score(all_true.cpu(), all_outputs.cpu())
    return average_loss, score


def epoch_end(avg_val_loss, best_score, score, model, save_path):
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), save_path)
        print(f"Val Loss: {avg_val_loss:.4f}  score: {score:.4f}\tSAVED MODEL\n")

    else:
        print(f"Val Loss: {avg_val_loss:.4f}  score: {score:.4f}\n")

    return best_score


def aux_train_pipeline(cfg):
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


def aux_infer_pipeline(cfg):
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
