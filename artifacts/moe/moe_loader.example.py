"""VÍ DỤ hợp đồng giữa UI và mô hình MoE. Đổi tên file thành moe_loader.py để UI nạp.

Chỉ cần: hàm load(artifacts_dir) trả về một object có các thuộc tính/hàm dưới đây.
Ví dụ này dùng 3 expert tuyến tính để minh họa (chạy được, không cần xgboost). MoE thật thay phần thân bằng
mô hình của nhóm; nếu expert là XGBoost thì expert_contribs dùng booster.predict(DMatrix, pred_contribs=True)."""
import numpy as np
import pandas as pd


class ToyMoE:
    expert_names = ["Nhóm A", "Nhóm B", "Nhóm C"]
    expert_profiles = {"Nhóm A": "Mô tả nghiệp vụ nhóm A", "Nhóm B": "Mô tả nghiệp vụ nhóm B", "Nhóm C": "Mô tả nghiệp vụ nhóm C"}
    meta = {"test_metrics": {m: {"value": 1.0, "ci_low": 0.9, "ci_high": 1.1} for m in ("mae", "rmse", "r2", "wape")}, "note": "ví dụ"}

    def __init__(self, feature_names, seed=0):
        rng = np.random.default_rng(seed)
        F, K = len(feature_names), 3
        self.feature_names = list(feature_names)
        self.mu = np.zeros(F)                                   # dùng để điền NaN và để căn tâm
        self.w = rng.normal(0, 300, (K, F))
        self.b = np.array([9000.0, 15000.0, 24000.0])
        self.gw = rng.normal(0, 0.3, (K, F))

    def _x(self, X):
        return X[self.feature_names].to_numpy(float)

    def _fill(self, x):
        return np.where(np.isnan(x), self.mu, x)

    def gating_proba(self, X):                                   # (n, K), mỗi dòng cộng bằng 1
        z = self._fill(self._x(X)) @ self.gw.T
        z = np.exp(z - z.max(axis=1, keepdims=True)); return z / z.sum(axis=1, keepdims=True)

    def expert_contribs(self, X):                                # (K, n, F+1), cột cuối là bias của expert
        x = self._fill(self._x(X)); K = len(self.b)
        phi = np.einsum("kf,nf->knf", self.w, x - self.mu)
        bias = np.broadcast_to((self.b + self.w @ self.mu)[:, None, None], (K, len(x), 1))
        return np.concatenate([phi, bias], axis=2)

    def predict(self, X):                                        # PHẢI bằng sum_k g_k * f_k, với f_k = tổng contribs của expert k
        return (self.gating_proba(X) * self.expert_contribs(X).sum(axis=2).T).sum(axis=1)


def load(artifacts_dir):
    from core.features import DEFAULT_FEATURES
    return ToyMoE(DEFAULT_FEATURES)
