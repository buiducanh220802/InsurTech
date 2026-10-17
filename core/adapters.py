"""Giao diện chung giữa UI và các mô hình.

Mọi mô hình trong demo (XGBoost, MoE) đều phải trả về cùng một kiểu Explanation:
    prediction = base_value + sum(contribs)       (đơn vị: đơn vị gốc của claim)
UI chỉ làm việc với Explanation nên thêm mô hình mới không cần sửa app.py.
"""
from __future__ import annotations
import importlib.util, json, os, sys
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .features import default_groups, DEFAULT_FEATURES

TOL = 0.05   # sai số cho phép khi kiểm tra tính cộng của SHAP (đơn vị claim)


@dataclass
class Explanation:
    prediction: float
    base_value: float
    contribs: pd.Series                      # index = tên feature
    extras: dict = field(default_factory=dict)

    def check(self):
        gap = abs(self.base_value + float(self.contribs.sum()) - self.prediction)
        assert gap < TOL, f"SHAP không cộng đủ: lệch {gap:.4f}"


class BaseAdapter:
    name: str = "model"
    feature_names: list
    groups: dict
    meta: dict

    def predict(self, X: pd.DataFrame) -> np.ndarray: raise NotImplementedError
    def explain(self, X1: pd.DataFrame) -> Explanation: raise NotImplementedError


# ----------------------------------------------------------------------------- XGBoost
def contrib_matrix(booster, X: pd.DataFrame) -> np.ndarray:
    """SHAP gốc của XGBoost (TreeSHAP): mảng (n, F+1), cột cuối là bias. Xử lý NaN nguyên bản."""
    import xgboost as xgb
    names = booster.feature_names
    dm = xgb.DMatrix(X[names], feature_names=names)
    return booster.predict(dm, pred_contribs=True)


class XGBAdapter(BaseAdapter):
    name = "XGBoost + SHAP"

    def __init__(self, directory: str):
        import xgboost as xgb
        self.dir = directory
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.booster = xgb.Booster()          # dùng Booster thuần: không cần scikit-learn khi chạy demo
        self.booster.load_model(os.path.join(directory, "model.json"))
        self.feature_names = self.meta.get("feature_names") or list(self.booster.feature_names)
        self.groups = self.meta.get("feature_groups") or default_groups(self.feature_names)

    def predict(self, X):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(X[self.feature_names], feature_names=self.feature_names))

    def explain(self, X1):
        c = contrib_matrix(self.booster, X1[self.feature_names])[0]
        pred = float(self.predict(X1)[0])
        e = Explanation(pred, float(c[-1]), pd.Series(c[:-1], index=self.feature_names))
        e.check()
        return e

    def global_files(self):
        out = {}
        for k, fn in {"summary": "shap_summary.png", "bar": "shap_bar.png", "groups": "shap_groups.png"}.items():
            p = os.path.join(self.dir, fn)
            if os.path.exists(p): out[k] = p
        return out

    def global_table(self):
        p = os.path.join(self.dir, "shap_importance.csv")
        return pd.read_csv(p) if os.path.exists(p) else None


# ----------------------------------------------------------------------------- MoE
def combine_expert_contribs(gating: np.ndarray, contribs: np.ndarray):
    """Gộp SHAP của K expert thành SHAP cho MoE.

    MoE dự đoán: y(x) = sum_k g_k(x) * f_k(x), với f_k(x) = bias_k + sum_j phi_kj(x)  (mỗi expert là cây).
    => y(x) = sum_k g_k*bias_k + sum_j sum_k g_k*phi_kj. Phân rã này chính xác, không cần xấp xỉ:
       base(x)      = sum_k g_k * bias_k            (mức nền phụ thuộc nhóm mà hồ sơ được xếp vào)
       contrib_j(x) = sum_k g_k * phi_kj
    gating:  (n, K)    xác suất gating, mỗi dòng cộng bằng 1
    contribs:(K, n, F+1)  SHAP của từng expert, cột cuối là bias
    """
    g = np.asarray(gating, dtype=float)
    c = np.asarray(contribs, dtype=float)
    assert c.ndim == 3 and c.shape[0] == g.shape[1] and c.shape[1] == g.shape[0], (g.shape, c.shape)
    phi = np.einsum("nk,knf->nf", g, c[:, :, :-1])
    base = np.einsum("nk,kn->n", g, c[:, :, -1])
    return base, phi


class MoEAdapter(BaseAdapter):
    """Bọc object MoE của nhóm. Object cần có (xem README, mục 'Gắn MoE'):
        expert_names: list[str]
        feature_names: list[str]                       (35 cột, đúng thứ tự)
        predict(X) -> (n,)                             đơn vị gốc của claim
        gating_proba(X) -> (n, K)
        expert_contribs(X) -> (K, n, F+1)              SHAP từng expert, cột cuối là bias
        expert_profiles: dict[str, str]  (tùy chọn)    mô tả nghiệp vụ của từng nhóm
        meta: dict (tùy chọn)                          test_metrics, best_params, ...
    """
    name = "MoE (nhiều chuyên gia)"

    def __init__(self, moe, label: str | None = None):
        self.moe = moe
        if label: self.name = label
        self.feature_names = list(getattr(moe, "feature_names", DEFAULT_FEATURES))
        self.groups = getattr(moe, "feature_groups", None) or default_groups(self.feature_names)
        self.meta = getattr(moe, "meta", {}) or {}
        self.expert_names = list(moe.expert_names)
        self.expert_profiles = getattr(moe, "expert_profiles", {}) or {}

    def predict(self, X):
        return np.asarray(self.moe.predict(X[self.feature_names]), dtype=float)

    def explain(self, X1):
        X1 = X1[self.feature_names]
        g = np.asarray(self.moe.gating_proba(X1), dtype=float)
        c = np.asarray(self.moe.expert_contribs(X1), dtype=float)
        base, phi = combine_expert_contribs(g, c)
        pred = float(self.predict(X1)[0])
        e = Explanation(pred, float(base[0]), pd.Series(phi[0], index=self.feature_names),
                        extras={"gating": pd.Series(g[0], index=self.expert_names),
                                "expert_profiles": self.expert_profiles})
        e.check()   # nếu lỗi ở đây: MoE.predict không bằng sum_k g_k * f_k, hoặc SHAP expert không cộng đủ
        return e

    def global_files(self):
        d = getattr(self.moe, "artifacts_dir", None)
        out = {}
        if d:
            for k, fn in {"summary": "shap_summary.png", "bar": "shap_bar.png", "groups": "shap_groups.png"}.items():
                p = os.path.join(d, fn)
                if os.path.exists(p): out[k] = p
        return out

    def global_table(self):
        d = getattr(self.moe, "artifacts_dir", None)
        p = os.path.join(d, "shap_importance.csv") if d else None
        return pd.read_csv(p) if p and os.path.exists(p) else None


def _load_moe_from_dir(directory: str):
    """artifacts/moe/moe_loader.py phải có hàm load(artifacts_dir) -> object MoE."""
    path = os.path.join(directory, "moe_loader.py")
    if directory not in sys.path:
        sys.path.insert(0, directory)          # để moe_loader.py import được các file .py cùng thư mục (VD moe_model.py)
    spec = importlib.util.spec_from_file_location("moe_loader", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    moe = mod.load(directory)
    if not hasattr(moe, "artifacts_dir"): moe.artifacts_dir = directory
    return moe


def load_models(artifacts_dir: str):
    """Trả về (dict tên->adapter, list thông báo). Mô hình nào chưa có file thì bỏ qua, UI vẫn chạy."""
    models, notes = {}, []
    xdir = os.path.join(artifacts_dir, "xgb")
    if os.path.exists(os.path.join(xdir, "model.json")):
        try:
            a = XGBAdapter(xdir); models[a.name] = a
        except Exception as ex:
            notes.append(f"Không nạp được XGBoost: {ex!r}")
    else:
        notes.append("Chưa có artifacts/xgb/model.json: chạy notebook train rồi giải nén xgb_artifacts.zip vào artifacts/xgb/.")
    mdir = os.path.join(artifacts_dir, "moe")
    if os.path.exists(os.path.join(mdir, "moe_loader.py")):
        try:
            a = MoEAdapter(_load_moe_from_dir(mdir)); models[a.name] = a
        except Exception as ex:
            notes.append(f"Không nạp được MoE: {ex!r}")
    elif os.environ.get("DEMO_MOCK_MOE") == "1" and "XGBoost + SHAP" in models:
        from .mock_moe import MockMoE
        a = MoEAdapter(MockMoE(models["XGBoost + SHAP"]), label="MoE (MOCK — chỉ để thử giao diện)")
        models[a.name] = a
        notes.append("Đang dùng MoE GIẢ LẬP (DEMO_MOCK_MOE=1). Kết quả không phải của mô hình MoE thật.")
    else:
        notes.append("Chưa có artifacts/moe/moe_loader.py: nhánh MoE chưa được gắn (xem README, mục 'Gắn MoE').")
    return models, notes