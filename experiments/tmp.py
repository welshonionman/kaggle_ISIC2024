from pathlib import Path

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from src.pipeline import get_train_pipeline


class CFG:
    wandb_mode = "disabled"
    exp_name = Path(__file__).stem

    seed = 46
    epochs = 30
    img_size = 384
    n_fold = 6

    pipeline = "auxv2"
    preprocess = "base_negative_sampling"
    dataset = "base_equal_sampling"
    auxtargets = [
        "sex",
        "age_approx",
        "anatom_site_general",
        "clin_size_long_diam_mm",
        "tbp_lv_A",
        "tbp_lv_Aext",
        "tbp_lv_area_perim_ratio",
        "tbp_lv_areaMM2",
        "tbp_lv_B",
        "tbp_lv_Bext",
        "tbp_lv_C",
        "tbp_lv_Cext",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaA",
        "tbp_lv_deltaB",
        "tbp_lv_deltaL",
        "tbp_lv_deltaLBnorm",
        "tbp_lv_eccentricity",
        "tbp_lv_H",
        "tbp_lv_Hext",
        "tbp_lv_L",
        "tbp_lv_Lext",
        "tbp_lv_minorAxisMM",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_perimeterMM",
        "tbp_lv_stdL",
        "tbp_lv_stdLExt",
        "tbp_lv_symm_2axis_angle",
        "tbp_lv_dnn_lesion_confidence",
    ]
    model_name: str = "auxv2"
    encoder_name: str = "tf_efficientnet_b1_ns"

    # image_dir = "/kaggle/input/improved_dataset/train_image"
    # meta_path = "/kaggle/input/improved_dataset/metadata.csv"

    train_batch_size = 32
    valid_batch_size = 64
    learning_rate = 1e-4

    lossfn = "BCEWithLogitsLoss"
    sampling_factor = 20
    loss_weight = 1
    optimizer = "AdamW"
    scheduler = "CosineAnnealingLR"

    min_lr = 1e-7
    weight_decay = 1e-6

    train_transform = A.Compose(
        [
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(limit=0.2, p=0.75),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 30.0)),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.0),
                    A.ElasticTransform(alpha=3),
                ],
                p=0.7,
            ),
            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85
            ),
            A.CoarseDropout(
                # max_h_size=int(img_size * 0.375),
                # max_w_size=int(img_size * 0.375),
                # num_holes=1,
                p=0.7,
            ),
            A.Resize(img_size, img_size, cv2.INTER_LANCZOS4),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0,
    )

    valid_transform = A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0,
    )


if __name__ == "__main__":
    pipeline = get_train_pipeline(CFG)
    pipeline(CFG)
