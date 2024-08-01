import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import IS_KAGGLE_NOTEBOOK


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


class ISIC_AuxV2_Model(nn.Module):
    def __init__(self, encoder_name, cfg):
        super(ISIC_AuxV2_Model, self).__init__()
        self.model = timm.create_model(encoder_name, pretrained=not IS_KAGGLE_NOTEBOOK)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()

        self.linear_mal = nn.Linear(in_features, 1)
        self.linear_sex = nn.Linear(in_features, 1)
        self.linear_age = nn.Linear(in_features, 1)
        self.linear_site = nn.Linear(in_features, 5)

        self.linear_clin_size_long_diam_mm = nn.Linear(in_features, 1)
        self.linear_tbp_lv_A = nn.Linear(in_features, 1)
        self.linear_tbp_lv_Aext = nn.Linear(in_features, 1)
        self.linear_tbp_lv_area_perim_ratio = nn.Linear(in_features, 1)
        self.linear_tbp_lv_areaMM2 = nn.Linear(in_features, 1)
        self.linear_tbp_lv_B = nn.Linear(in_features, 1)
        self.linear_tbp_lv_Bext = nn.Linear(in_features, 1)
        self.linear_tbp_lv_C = nn.Linear(in_features, 1)
        self.linear_tbp_lv_Cext = nn.Linear(in_features, 1)
        self.linear_tbp_lv_color_std_mean = nn.Linear(in_features, 1)
        self.linear_tbp_lv_deltaA = nn.Linear(in_features, 1)
        self.linear_tbp_lv_deltaB = nn.Linear(in_features, 1)
        self.linear_tbp_lv_deltaL = nn.Linear(in_features, 1)
        self.linear_tbp_lv_deltaLBnorm = nn.Linear(in_features, 1)
        self.linear_tbp_lv_eccentricity = nn.Linear(in_features, 1)
        self.linear_tbp_lv_H = nn.Linear(in_features, 1)
        self.linear_tbp_lv_Hext = nn.Linear(in_features, 1)
        self.linear_tbp_lv_L = nn.Linear(in_features, 1)
        self.linear_tbp_lv_Lext = nn.Linear(in_features, 1)
        self.linear_tbp_lv_minorAxisMM = nn.Linear(in_features, 1)
        self.linear_tbp_lv_norm_border = nn.Linear(in_features, 1)
        self.linear_tbp_lv_norm_color = nn.Linear(in_features, 1)
        self.linear_tbp_lv_perimeterMM = nn.Linear(in_features, 1)
        self.linear_tbp_lv_stdL = nn.Linear(in_features, 1)
        self.linear_tbp_lv_stdLExt = nn.Linear(in_features, 1)
        self.linear_tbp_lv_symm_2axis_angle = nn.Linear(in_features, 1)
        self.linear_tbp_lv_dnn_lesion_confidence = nn.Linear(in_features, 1)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)

        malignant = self.linear_mal(pooled_features)
        sex = self.linear_sex(pooled_features)
        age = self.linear_age(pooled_features)
        site = self.linear_site(pooled_features)

        clin_size_long_diam_mm = self.linear_clin_size_long_diam_mm(pooled_features)
        tbp_lv_A = self.linear_tbp_lv_A(pooled_features)
        tbp_lv_Aext = self.linear_tbp_lv_Aext(pooled_features)
        tbp_lv_area_perim_ratio = self.linear_tbp_lv_area_perim_ratio(pooled_features)
        tbp_lv_areaMM2 = self.linear_tbp_lv_areaMM2(pooled_features)
        tbp_lv_B = self.linear_tbp_lv_B(pooled_features)
        tbp_lv_Bext = self.linear_tbp_lv_Bext(pooled_features)
        tbp_lv_C = self.linear_tbp_lv_C(pooled_features)
        tbp_lv_Cext = self.linear_tbp_lv_Cext(pooled_features)
        tbp_lv_color_std_mean = self.linear_tbp_lv_color_std_mean(pooled_features)
        tbp_lv_deltaA = self.linear_tbp_lv_deltaA(pooled_features)
        tbp_lv_deltaB = self.linear_tbp_lv_deltaB(pooled_features)
        tbp_lv_deltaL = self.linear_tbp_lv_deltaL(pooled_features)
        tbp_lv_deltaLBnorm = self.linear_tbp_lv_deltaLBnorm(pooled_features)
        tbp_lv_eccentricity = self.linear_tbp_lv_eccentricity(pooled_features)
        tbp_lv_H = self.linear_tbp_lv_H(pooled_features)
        tbp_lv_Hext = self.linear_tbp_lv_Hext(pooled_features)
        tbp_lv_L = self.linear_tbp_lv_L(pooled_features)
        tbp_lv_Lext = self.linear_tbp_lv_Lext(pooled_features)
        tbp_lv_minorAxisMM = self.linear_tbp_lv_minorAxisMM(pooled_features)
        tbp_lv_norm_border = self.linear_tbp_lv_norm_border(pooled_features)
        tbp_lv_norm_color = self.linear_tbp_lv_norm_color(pooled_features)
        tbp_lv_perimeterMM = self.linear_tbp_lv_perimeterMM(pooled_features)
        tbp_lv_stdL = self.linear_tbp_lv_stdL(pooled_features)
        tbp_lv_stdLExt = self.linear_tbp_lv_stdLExt(pooled_features)
        tbp_lv_symm_2axis_angle = self.linear_tbp_lv_symm_2axis_angle(pooled_features)
        tbp_lv_dnn_lesion_confidence = self.linear_tbp_lv_dnn_lesion_confidence(
            pooled_features
        )
        return {
            "malignant": malignant,
            "sex": sex,
            "age_approx": age,
            "anatom_site_general": site,
            "clin_size_long_diam_mm": clin_size_long_diam_mm,
            "tbp_lv_A": tbp_lv_A,
            "tbp_lv_Aext": tbp_lv_Aext,
            "tbp_lv_area_perim_ratio": tbp_lv_area_perim_ratio,
            "tbp_lv_areaMM2": tbp_lv_areaMM2,
            "tbp_lv_B": tbp_lv_B,
            "tbp_lv_Bext": tbp_lv_Bext,
            "tbp_lv_C": tbp_lv_C,
            "tbp_lv_Cext": tbp_lv_Cext,
            "tbp_lv_color_std_mean": tbp_lv_color_std_mean,
            "tbp_lv_deltaA": tbp_lv_deltaA,
            "tbp_lv_deltaB": tbp_lv_deltaB,
            "tbp_lv_deltaL": tbp_lv_deltaL,
            "tbp_lv_deltaLBnorm": tbp_lv_deltaLBnorm,
            "tbp_lv_eccentricity": tbp_lv_eccentricity,
            "tbp_lv_H": tbp_lv_H,
            "tbp_lv_Hext": tbp_lv_Hext,
            "tbp_lv_L": tbp_lv_L,
            "tbp_lv_Lext": tbp_lv_Lext,
            "tbp_lv_minorAxisMM": tbp_lv_minorAxisMM,
            "tbp_lv_norm_border": tbp_lv_norm_border,
            "tbp_lv_norm_color": tbp_lv_norm_color,
            "tbp_lv_perimeterMM": tbp_lv_perimeterMM,
            "tbp_lv_stdL": tbp_lv_stdL,
            "tbp_lv_stdLExt": tbp_lv_stdLExt,
            "tbp_lv_symm_2axis_angle": tbp_lv_symm_2axis_angle,
            "tbp_lv_dnn_lesion_confidence": tbp_lv_dnn_lesion_confidence,
        }
