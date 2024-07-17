import numpy as np
from sklearn.metrics import roc_auc_score


def calc_score(
    solution: np.ndarray, submission: np.ndarray, min_tpr: float = 0.80
) -> float:
    v_gt = abs(solution - 1)
    v_pred = np.array([1.0 - x for x in submission])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (
        partial_auc_scaled - 0.5
    )
    return partial_auc


def pauc_80(preds, data):  # for lightgbm.cv(feval=pauc_80)
    score_value = calc_score(data.get_label(), preds, min_tpr=0.8)
    return "pauc_80", score_value, True
