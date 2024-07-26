import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ISIC_Aux_Model(nn.Module):
    def __init__(self, encoder_name, cfg):
        super(ISIC_Aux_Model, self).__init__()
        auxtarget = getattr(cfg, "auxtarget", [])

        self.model = timm.create_model(encoder_name, pretrained=True)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()

        self.linear_mal = nn.Linear(in_features, 1)
        self.linear_sex = nn.Linear(in_features, 1)
        self.linear_age = nn.Linear(in_features, 1)
        self.linear_site = nn.Linear(in_features, 1)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)

        malignant = self.linear_mal(pooled_features)
        sex = self.linear_sex(pooled_features)
        age = self.linear_age(pooled_features)
        site = self.linear_site(pooled_features)

        return {
            "malignant": malignant,
            "sex": sex,
            "age": age,
            "site": site,
        }
