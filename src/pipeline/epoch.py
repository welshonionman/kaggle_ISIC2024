import torch
from tqdm import tqdm

from src.constants import DEVICE
from src.model import get_lossfn

from .metrics import calc_score


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
    train_loss = 0

    model.train()

    pbar_train = enumerate(train_loader)
    pbar_train = tqdm(
        pbar_train,
        total=len(train_loader),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-0b}",
    )

    for batch_idx, batch in pbar_train:
        inputs = batch["image"].to(DEVICE, dtype=torch.float)
        labels = batch["target"].to(DEVICE, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        criterion = get_lossfn(cfg, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        average_loss = train_loss / (batch_idx + 1)

        pbar_train_desc(pbar_train, scheduler, epoch, cfg.epochs, average_loss)
    scheduler.step()
    return average_loss


def valid_1epoch(model, valid_loader, epoch, cfg):
    valid_loss = 0
    y_true = torch.tensor([], device=DEVICE)
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
            labels = batch["target"].to(DEVICE, dtype=torch.float)
            outputs = model(inputs).squeeze()
            criterion = get_lossfn(cfg, labels)
            valid_loss += criterion(outputs, labels).item()
            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

            average_loss = valid_loss / (batch_idx + 1)
            pbar_valid_desc(pbar_val, average_loss)

    score = calc_score(y_true.cpu(), all_outputs.cpu())
    return average_loss, score


def epoch_end(avg_val_loss, best_score, score, model, save_path):
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), save_path)
        print(f"Val Loss: {avg_val_loss:.4f}  score: {score:.4f}\tSAVED MODEL\n")

    else:
        print(f"Val Loss: {avg_val_loss:.4f}  score: {score:.4f}\n")

    return best_score
