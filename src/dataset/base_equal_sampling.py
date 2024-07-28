import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ISIC_Base_Train_Dataset(Dataset):
    def __init__(self, df, cfg):
        self.auxtargets = getattr(cfg, "auxtargets", [])
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive["file_path"].values
        self.file_names_negative = self.df_negative["file_path"].values
        self.targets_positive = self.df_positive[["target"] + self.auxtargets].values
        self.targets_negative = self.df_negative[["target"] + self.auxtargets].values
        self.transforms = cfg.train_transform

    def __len__(self):
        return len(self.df_positive) * 2

    def gen_target_dict(self, target):
        target_dict = {}
        target_dict["malignant"] = target[0]
        for i_target, auxtargets in enumerate(self.auxtargets):
            target_dict[auxtargets] = target[i_target + 1]
        return target_dict

    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % df.shape[0]

        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = torch.tensor(targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        target_dict = self.gen_target_dict(target)

        return {"image": img, "target": target_dict}


class ISIC_Base_Valid_Dataset(Dataset):
    def __init__(self, df, cfg):
        self.auxtargets = getattr(cfg, "auxtargets", [])
        self.df = df
        self.file_names = df["file_path"].values
        self.targets = df[["target"] + self.auxtargets].values
        self.transforms = cfg.valid_transform

    def __len__(self):
        return len(self.df)

    def gen_target_dict(self, target):
        target_dict = {}
        target_dict["malignant"] = target[0]
        for i_target, auxtargets in enumerate(self.auxtargets):
            target_dict[auxtargets] = target[i_target + 1]
        return target_dict

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        target_dict = self.gen_target_dict(target)

        return {"image": img, "target": target_dict}


class ISIC_Base_Test_Dataset(Dataset):
    def __init__(self, df, file_hdf, cfg):
        self.df = df
        self.hdf_path = h5py.File(file_hdf, mode="r")
        self.isic_ids = df["isic_id"].values
        self.targets = df["target"].values
        self.transforms = cfg.valid_transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img_data = self.hdf_path[isic_id][()]
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "target": target}
