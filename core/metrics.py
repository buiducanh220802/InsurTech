"""Chỉ số đánh giá dùng chung cho cả hai nhánh (copy nguyên cho notebook của MoE)."""
import numpy as np


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float((err ** 2).sum()); ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                r2=1 - ss_res / ss_tot, wape=float(np.abs(err).sum() / np.abs(y_true).sum()))


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42, alpha=0.05):
    """Chỉ số trên test + khoảng tin cậy bootstrap (resample hồ sơ test).
    Trả về {metric: {"value", "ci_low", "ci_high"}}."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    rng = np.random.default_rng(seed); n = len(y_true)
    point = compute_metrics(y_true, y_pred)
    draws = [compute_metrics(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    out = {}
    for k, v in point.items():
        s = np.array([d[k] for d in draws])
        out[k] = dict(value=v, ci_low=float(np.quantile(s, alpha / 2)), ci_high=float(np.quantile(s, 1 - alpha / 2)))
    return out
