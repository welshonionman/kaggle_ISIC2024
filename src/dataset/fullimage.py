from io import BytesIO

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISIC_Fullimage_Train_Dataset(Dataset):
    def __init__(self, df, cfg):
        self.auxtargets = getattr(cfg, "auxtargets", [])
        self.df = df
        self.file_names = self.df["file_path"].values
        self.targets = self.df[["target"] + self.auxtargets].values
        self.transforms = cfg.train_transform

    def __len__(self):
        return len(self.df)

    def gen_target_dict(self, target):
        target_dict = {}
        target_dict["malignant"] = target[0]
        for i_target, auxtarget in enumerate(self.auxtargets):
            target_dict[auxtarget] = target[i_target + 1]
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


class ISIC_Fullimage_Valid_Dataset(Dataset):
    def __init__(self, df, cfg):
        self.auxtargets = getattr(cfg, "auxtargets", [])
        self.df = df
        self.file_names = df["file_path"].values
        self.targets = self.df[["target"] + self.auxtargets].values
        self.transforms = cfg.valid_transform

    def __len__(self):
        return len(self.df)

    def gen_target_dict(self, target):
        target_dict = {}
        target_dict["malignant"] = target[0]
        for i_target, auxtarget in enumerate(self.auxtargets):
            target_dict[auxtarget] = target[i_target + 1]
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
        img_data = self.hdf_path[isic_id][()]
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "target": target}
