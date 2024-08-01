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
        losses["age_approx"] = criterion_age(output_age, label_age) / 5000

    if "anatom_site_general" in auxtargets:
        criterion_site = nn.CrossEntropyLoss()
        output_site = outputs["anatom_site_general"].squeeze()
        label_site = labels["anatom_site_general"].long()
        losses["anatom_site_general"] = criterion_site(output_site, label_site) / 6

    if "clin_size_long_diam_mm" in auxtargets:
        criterion_clin_size_long_diam_mm = nn.MSELoss()
        output_clin_size_long_diam_mm = outputs["clin_size_long_diam_mm"].squeeze()
        label_clin_size_long_diam_mm = labels["clin_size_long_diam_mm"]
        losses["clin_size_long_diam_mm"] = (
            criterion_clin_size_long_diam_mm(
                output_clin_size_long_diam_mm, label_clin_size_long_diam_mm
            )
            / 5000
        )

    if "tbp_lv_A" in auxtargets:
        criterion_tbp_lv_A = nn.MSELoss()
        output_tbp_lv_A = outputs["tbp_lv_A"].squeeze()
        label_tbp_lv_A = labels["tbp_lv_A"]
        losses["tbp_lv_A"] = criterion_tbp_lv_A(output_tbp_lv_A, label_tbp_lv_A) / 5000

    if "tbp_lv_Aext" in auxtargets:
        criterion_tbp_lv_Aext = nn.MSELoss()
        output_tbp_lv_Aext = outputs["tbp_lv_Aext"].squeeze()
        label_tbp_lv_Aext = labels["tbp_lv_Aext"]
        losses["tbp_lv_Aext"] = (
            criterion_tbp_lv_Aext(output_tbp_lv_Aext, label_tbp_lv_Aext) / 5000
        )

    if "tbp_lv_area_perim_ratio" in auxtargets:
        criterion_tbp_lv_area_perim_ratio = nn.MSELoss()
        output_tbp_lv_area_perim_ratio = outputs["tbp_lv_area_perim_ratio"].squeeze()
        label_tbp_lv_area_perim_ratio = labels["tbp_lv_area_perim_ratio"]
        losses["tbp_lv_area_perim_ratio"] = (
            criterion_tbp_lv_area_perim_ratio(
                output_tbp_lv_area_perim_ratio, label_tbp_lv_area_perim_ratio
            )
            / 5000
        )

    if "tbp_lv_areaMM2" in auxtargets:
        criterion_tbp_lv_areaMM2 = nn.MSELoss()
        output_tbp_lv_areaMM2 = outputs["tbp_lv_areaMM2"].squeeze()
        label_tbp_lv_areaMM2 = labels["tbp_lv_areaMM2"]
        losses["tbp_lv_areaMM2"] = (
            criterion_tbp_lv_areaMM2(output_tbp_lv_areaMM2, label_tbp_lv_areaMM2) / 5000
        )

    if "tbp_lv_B" in auxtargets:
        criterion_tbp_lv_B = nn.MSELoss()
        output_tbp_lv_B = outputs["tbp_lv_B"].squeeze()
        label_tbp_lv_B = labels["tbp_lv_B"]
        losses["tbp_lv_B"] = criterion_tbp_lv_B(output_tbp_lv_B, label_tbp_lv_B) / 5000

    if "tbp_lv_Bext" in auxtargets:
        criterion_tbp_lv_Bext = nn.MSELoss()
        output_tbp_lv_Bext = outputs["tbp_lv_Bext"].squeeze()
        label_tbp_lv_Bext = labels["tbp_lv_Bext"]
        losses["tbp_lv_Bext"] = (
            criterion_tbp_lv_Bext(output_tbp_lv_Bext, label_tbp_lv_Bext) / 5000
        )

    if "tbp_lv_C" in auxtargets:
        criterion_tbp_lv_C = nn.MSELoss()
        output_tbp_lv_C = outputs["tbp_lv_C"].squeeze()
        label_tbp_lv_C = labels["tbp_lv_C"]
        losses["tbp_lv_C"] = criterion_tbp_lv_C(output_tbp_lv_C, label_tbp_lv_C) / 5000

    if "tbp_lv_Cext" in auxtargets:
        criterion_tbp_lv_Cext = nn.MSELoss()
        output_tbp_lv_Cext = outputs["tbp_lv_Cext"].squeeze()
        label_tbp_lv_Cext = labels["tbp_lv_Cext"]
        losses["tbp_lv_Cext"] = (
            criterion_tbp_lv_Cext(output_tbp_lv_Cext, label_tbp_lv_Cext) / 5000
        )

    if "tbp_lv_color_std_mean" in auxtargets:
        criterion_tbp_lv_color_std_mean = nn.MSELoss()
        output_tbp_lv_color_std_mean = outputs["tbp_lv_color_std_mean"].squeeze()
        label_tbp_lv_color_std_mean = labels["tbp_lv_color_std_mean"]
        losses["tbp_lv_color_std_mean"] = (
            criterion_tbp_lv_color_std_mean(
                output_tbp_lv_color_std_mean, label_tbp_lv_color_std_mean
            )
            / 5000
        )

    if "tbp_lv_deltaA" in auxtargets:
        criterion_tbp_lv_deltaA = nn.MSELoss()
        output_tbp_lv_deltaA = outputs["tbp_lv_deltaA"].squeeze()
        label_tbp_lv_deltaA = labels["tbp_lv_deltaA"]
        losses["tbp_lv_deltaA"] = (
            criterion_tbp_lv_deltaA(output_tbp_lv_deltaA, label_tbp_lv_deltaA) / 5000
        )

    if "tbp_lv_deltaB" in auxtargets:
        criterion_tbp_lv_deltaB = nn.MSELoss()
        output_tbp_lv_deltaB = outputs["tbp_lv_deltaB"].squeeze()
        label_tbp_lv_deltaB = labels["tbp_lv_deltaB"]
        losses["tbp_lv_deltaB"] = (
            criterion_tbp_lv_deltaB(output_tbp_lv_deltaB, label_tbp_lv_deltaB) / 5000
        )

    if "tbp_lv_deltaL" in auxtargets:
        criterion_tbp_lv_deltaL = nn.MSELoss()
        output_tbp_lv_deltaL = outputs["tbp_lv_deltaL"].squeeze()
        label_tbp_lv_deltaL = labels["tbp_lv_deltaL"]
        losses["tbp_lv_deltaL"] = (
            criterion_tbp_lv_deltaL(output_tbp_lv_deltaL, label_tbp_lv_deltaL) / 5000
        )

    if "tbp_lv_deltaLBnorm" in auxtargets:
        criterion_tbp_lv_deltaLBnorm = nn.MSELoss()
        output_tbp_lv_deltaLBnorm = outputs["tbp_lv_deltaLBnorm"].squeeze()
        label_tbp_lv_deltaLBnorm = labels["tbp_lv_deltaLBnorm"]
        losses["tbp_lv_deltaLBnorm"] = (
            criterion_tbp_lv_deltaLBnorm(
                output_tbp_lv_deltaLBnorm, label_tbp_lv_deltaLBnorm
            )
            / 5000
        )

    if "tbp_lv_eccentricity" in auxtargets:
        criterion_tbp_lv_eccentricity = nn.MSELoss()
        output_tbp_lv_eccentricity = outputs["tbp_lv_eccentricity"].squeeze()
        label_tbp_lv_eccentricity = labels["tbp_lv_eccentricity"]
        losses["tbp_lv_eccentricity"] = (
            criterion_tbp_lv_eccentricity(
                output_tbp_lv_eccentricity, label_tbp_lv_eccentricity
            )
            / 5000
        )

    if "tbp_lv_H" in auxtargets:
        criterion_tbp_lv_H = nn.MSELoss()
        output_tbp_lv_H = outputs["tbp_lv_H"].squeeze()
        label_tbp_lv_H = labels["tbp_lv_H"]
        losses["tbp_lv_H"] = criterion_tbp_lv_H(output_tbp_lv_H, label_tbp_lv_H) / 5000

    if "tbp_lv_Hext" in auxtargets:
        criterion_tbp_lv_Hext = nn.MSELoss()
        output_tbp_lv_Hext = outputs["tbp_lv_Hext"].squeeze()
        label_tbp_lv_Hext = labels["tbp_lv_Hext"]
        losses["tbp_lv_Hext"] = (
            criterion_tbp_lv_Hext(output_tbp_lv_Hext, label_tbp_lv_Hext) / 5000
        )

    if "tbp_lv_L" in auxtargets:
        criterion_tbp_lv_L = nn.MSELoss()
        output_tbp_lv_L = outputs["tbp_lv_L"].squeeze()
        label_tbp_lv_L = labels["tbp_lv_L"]
        losses["tbp_lv_L"] = criterion_tbp_lv_L(output_tbp_lv_L, label_tbp_lv_L) / 5000

    if "tbp_lv_Lext" in auxtargets:
        criterion_tbp_lv_Lext = nn.MSELoss()
        output_tbp_lv_Lext = outputs["tbp_lv_Lext"].squeeze()
        label_tbp_lv_Lext = labels["tbp_lv_Lext"]
        losses["tbp_lv_Lext"] = (
            criterion_tbp_lv_Lext(output_tbp_lv_Lext, label_tbp_lv_Lext) / 5000
        )

    if "tbp_lv_minorAxisMM" in auxtargets:
        criterion_tbp_lv_minorAxisMM = nn.MSELoss()
        output_tbp_lv_minorAxisMM = outputs["tbp_lv_minorAxisMM"].squeeze()
        label_tbp_lv_minorAxisMM = labels["tbp_lv_minorAxisMM"]
        losses["tbp_lv_minorAxisMM"] = (
            criterion_tbp_lv_minorAxisMM(
                output_tbp_lv_minorAxisMM, label_tbp_lv_minorAxisMM
            )
            / 5000
        )

    if "tbp_lv_norm_border" in auxtargets:
        criterion_tbp_lv_norm_border = nn.MSELoss()
        output_tbp_lv_norm_border = outputs["tbp_lv_norm_border"].squeeze()
        label_tbp_lv_norm_border = labels["tbp_lv_norm_border"]
        losses["tbp_lv_norm_border"] = (
            criterion_tbp_lv_norm_border(
                output_tbp_lv_norm_border, label_tbp_lv_norm_border
            )
            / 5000
        )

    if "tbp_lv_norm_color" in auxtargets:
        criterion_tbp_lv_norm_color = nn.MSELoss()
        output_tbp_lv_norm_color = outputs["tbp_lv_norm_color"].squeeze()
        label_tbp_lv_norm_color = labels["tbp_lv_norm_color"]
        losses["tbp_lv_norm_color"] = (
            criterion_tbp_lv_norm_color(
                output_tbp_lv_norm_color, label_tbp_lv_norm_color
            )
            / 5000
        )

    if "tbp_lv_perimeterMM" in auxtargets:
        criterion_tbp_lv_perimeterMM = nn.MSELoss()
        output_tbp_lv_perimeterMM = outputs["tbp_lv_perimeterMM"].squeeze()
        label_tbp_lv_perimeterMM = labels["tbp_lv_perimeterMM"]
        losses["tbp_lv_perimeterMM"] = (
            criterion_tbp_lv_perimeterMM(
                output_tbp_lv_perimeterMM, label_tbp_lv_perimeterMM
            )
            / 5000
        )
    if "tbp_lv_stdL" in auxtargets:
        criterion_tbp_lv_stdL = nn.MSELoss()
        output_tbp_lv_stdL = outputs["tbp_lv_stdL"].squeeze()
        label_tbp_lv_stdL = labels["tbp_lv_stdL"]
        losses["tbp_lv_stdL"] = (
            criterion_tbp_lv_stdL(output_tbp_lv_stdL, label_tbp_lv_stdL) / 5000
        )

    if "tbp_lv_stdLExt" in auxtargets:
        criterion_tbp_lv_stdLExt = nn.MSELoss()
        output_tbp_lv_stdLExt = outputs["tbp_lv_stdLExt"].squeeze()
        label_tbp_lv_stdLExt = labels["tbp_lv_stdLExt"]
        losses["tbp_lv_stdLExt"] = (
            criterion_tbp_lv_stdLExt(output_tbp_lv_stdLExt, label_tbp_lv_stdLExt) / 5000
        )

    if "tbp_lv_symm_2axis_angle" in auxtargets:
        criterion_tbp_lv_symm_2axis_angle = nn.MSELoss()
        output_tbp_lv_symm_2axis_angle = outputs["tbp_lv_symm_2axis_angle"].squeeze()
        label_tbp_lv_symm_2axis_angle = labels["tbp_lv_symm_2axis_angle"]
        losses["tbp_lv_symm_2axis_angle"] = (
            criterion_tbp_lv_symm_2axis_angle(
                output_tbp_lv_symm_2axis_angle, label_tbp_lv_symm_2axis_angle
            )
            / 5000
        )

    if "tbp_lv_dnn_lesion_confidence" in auxtargets:
        criterion_tbp_lv_dnn_lesion_confidence = nn.MSELoss()
        output_tbp_lv_dnn_lesion_confidence = outputs[
            "tbp_lv_dnn_lesion_confidence"
        ].squeeze()
        label_tbp_lv_dnn_lesion_confidence = labels["tbp_lv_dnn_lesion_confidence"]
        losses["tbp_lv_dnn_lesion_confidence"] = (
            criterion_tbp_lv_dnn_lesion_confidence(
                output_tbp_lv_dnn_lesion_confidence, label_tbp_lv_dnn_lesion_confidence
            )
            / 5000
        )

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


def auxv2_train_pipeline(cfg):
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


def auxv2_infer_pipeline(cfg):
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
