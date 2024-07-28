from src.constants import TRAIN_DIR


def get_train_file_path(image_id, cfg):
    image_dir = getattr(cfg, "image_dir", TRAIN_DIR)
    return f"{image_dir}/{image_id}.jpg"


def impute_missing_values(df):
    df.fillna(
        {
            "sex": df["sex"].mode()[0],
            "anatom_site_general": df["anatom_site_general"].mode()[0],
            "age_approx": df["age_approx"].mode()[0],
        },
        inplace=True,
    )
    return df
