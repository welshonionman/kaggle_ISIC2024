from torch.utils.data import DataLoader

from src.constants import TEST_HDF

from .base import (
    ISIC_Base_Test_Dataset,
    ISIC_Base_Train_Dataset,
    ISIC_Base_Valid_Dataset,
)


def get_train_dataloader(df, fold, cfg):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = ISIC_Base_Train_Dataset(df_train, transforms=cfg.train_transform)
    valid_dataset = ISIC_Base_Valid_Dataset(df_valid, transforms=cfg.valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader


def get_test_dataloader(df, cfg):
    test_dataset = ISIC_Base_Test_Dataset(df, TEST_HDF, transforms=cfg.valid_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return test_loader
