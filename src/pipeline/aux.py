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


def calc_loss(outputs, labels, auxtargets):
    losses = {}

    criterion_mal = nn.BCEWithLogitsLoss()
    output_mal = outputs["malignant"].squeeze()
    label_mal = labels["malignant"]
    losses["malignant"] = criterion_mal(output_mal, label_mal)

    if "sex" in auxtargets:
        criterion_sex = nn.BCEWithLogitsLoss()
        output_sex = outputs["sex"].squeeze()
        label_sex = labels["sex"]
        losses["sex"] = criterion_sex(output_sex, label_sex) / 3

    if "age_approx" in auxtargets:
        criterion_age = nn.MSELoss()
        output_age = outputs["age_approx"].squeeze()
        label_age = labels["age_approx"]
        losses["age_approx"] = criterion_age(output_age, label_age) / 500

    if "anatom_site_general" in auxtargets:
        criterion_site = nn.CrossEntropyLoss()
        output_site = outputs["anatom_site_general"].squeeze()
        label_site = labels["anatom_site_general"].long()
        losses["anatom_site_general"] = criterion_site(output_site, label_site) / 6
    return losses


def train_1epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    auxtargets = getattr(cfg, "auxtargets", [])
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

        losses = calc_loss(outputs, labels, auxtargets)

        loss_sum = losses["malignant"] + sum(
            [losses[auxtarget] for auxtarget in auxtargets]
        )

        loss_sum.backward()

        optimizer.step()

        train_loss += loss_sum.item()
        average_loss = train_loss / (batch_idx + 1)

        pbar_train_desc(pbar_train, scheduler, epoch, cfg.epochs, average_loss)
    scheduler.step()
    return average_loss


def calc_acc_bin(true, pred):
    pred = (torch.sigmoid(pred) > 0.5).float()
    return (true == pred).sum() / len(true)


def calc_mse(true, pred):
    return nn.MSELoss()(true, pred)


def calc_acc_mul(true, pred):
    pred = torch.argmax(pred, dim=1)
    return (true == pred).sum() / len(true)


def valid_1epoch(model, valid_loader, epoch, cfg):
    auxtargets = getattr(cfg, "auxtargets", [])
    valid_loss = 0
    mal_true = torch.tensor([], device=DEVICE)
    mal_outputs = torch.tensor([], device=DEVICE)
    # sex_true = torch.tensor([], device=DEVICE)
    # sex_outputs = torch.tensor([], device=DEVICE)
    age_true = torch.tensor([], device=DEVICE)
    age_outputs = torch.tensor([], device=DEVICE)
    # site_true = torch.tensor([], device=DEVICE)
    # site_outputs = torch.tensor([], device=DEVICE)
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

            losses = calc_loss(outputs, labels, auxtargets)

            loss_sum = losses["malignant"] + sum(
                [losses[auxtarget] for auxtarget in auxtargets]
            )

            valid_loss += loss_sum
            mal_true = torch.cat((mal_true, labels["malignant"].squeeze()), 0)
            mal_outputs = torch.cat((mal_outputs, outputs["malignant"].squeeze()), 0)
            # sex_true = torch.cat((sex_true, labels["sex"].squeeze()), 0)
            # sex_outputs = torch.cat((sex_outputs, outputs["sex"].squeeze()), 0)
            age_true = torch.cat((age_true, labels["age_approx"].squeeze()), 0)
            age_outputs = torch.cat((age_outputs, outputs["age_approx"].squeeze()), 0)
            # site_true = torch.cat(
            #     (site_true, labels["anatom_site_general"].squeeze()), 0
            # )
            # site_outputs = torch.cat((site_outputs, outputs["anatom_site_general"].squeeze()), 0)

            average_loss = valid_loss / (batch_idx + 1)
            pbar_valid_desc(pbar_val, average_loss)

    mal_score = calc_score(mal_true.cpu(), mal_outputs.cpu())
    # sex_acc = calc_acc_bin(sex_true.cpu(), sex_outputs.cpu())
    age_acc = calc_mse(age_true.cpu(), age_outputs.cpu())
    # site_acc = calc_acc_mul(site_true.cpu(), site_outputs.cpu())
    print(age_acc)
    return average_loss, mal_score


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
    model.load_state_dict(torch.load(cfg.weight_path))
    model.to(DEVICE)

    preds = []
    with torch.no_grad():
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for step, data in bar:
            images = data["image"].to(DEVICE, dtype=torch.float)
            outputs = model(images)
            preds.append(torch.sigmoid(outputs["malignant"]).detach().cpu().numpy())
    preds = np.concatenate(preds).flatten()
    df_sub["target"] = preds
    df_sub.to_csv("submission.csv", index=False)
