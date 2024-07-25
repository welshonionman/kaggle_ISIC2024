from io import BytesIO

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISIC_Fullimage_Train_Dataset(Dataset):
    def __init__(self, df, cfg):
        auxtarget = getattr(cfg, "auxtarget", [])
        self.df = df
        self.file_names = self.df["file_path"].values
        self.targets = self.df[["target"] + auxtarget].values
        self.transforms = cfg.train_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "target": target}


class ISIC_Fullimage_Valid_Dataset(Dataset):
    def __init__(self, df, cfg):
        auxtarget = getattr(cfg, "auxtarget", [])
        self.df = df
        self.file_names = df["file_path"].values
        self.targets = self.df[["target"] + auxtarget].values
        self.transforms = cfg.valid_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "target": target}


class ISIC_Fullimage_Test_Dataset(Dataset):
    def __init__(self, df, file_hdf, cfg):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df["isic_id"].values
        self.targets = df["target"].values
        self.transforms = cfg.valid_transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "target": target}
