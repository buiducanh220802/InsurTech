"""Portable loader and inference API for the exported raw-claim MoE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


class MoEModel:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = Path(artifacts_dir).resolve()
        with open(self.artifacts_dir / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.feature_names = list(self.meta["feature_names"])
        self.expert_names = list(self.meta["expert_names"])
        self.K = int(self.meta["n_experts"])
        self.experts = []
        for k in range(self.K):
            booster = xgb.Booster()
            booster.load_model(str(self.artifacts_dir / f"expert_{k}.json"))
            self.experts.append(booster)
        # Booster thuần (không cần scikit-learn); gate.json lưu objective multi:softprob
        self.gate = xgb.Booster()
        self.gate.load_model(str(self.artifacts_dir / "gate.json"))
        self.expert_profiles = self.meta.get("expert_profiles", {})

    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"Missing model feature(s): {missing}")
        # Work on a copy and preserve the caller's DataFrame.
        out = X[self.feature_names].copy()
        for c in self.feature_names:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def gating_proba(self, X: pd.DataFrame) -> np.ndarray:
        out = self._X(X)
        D = xgb.DMatrix(out, feature_names=self.feature_names)
        p = np.asarray(self.gate.predict(D), dtype=float)
        if p.ndim == 1:                      # trường hợp gate nhị phân: p = xác suất lớp 1
            p = np.column_stack([1.0 - p, p])
        return p

    def expert_predictions(self, X: pd.DataFrame) -> np.ndarray:
        out = self._X(X)
        D = xgb.DMatrix(out, feature_names=self.feature_names)
        return np.vstack([booster.predict(D) for booster in self.experts]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Tính từ chính SHAP của expert (cộng bằng float64) để đảm bảo predict = Σ g_k·f_k đúng tuyệt đối;
        # booster.predict dùng float32 nên lệch cỡ 1e-6 * giá trị so với tổng SHAP.
        G = self.gating_proba(X)
        mu = self.expert_contribs(X).astype(np.float64).sum(axis=2).T
        return np.sum(G.astype(np.float64) * mu, axis=1)

    def expert_contribs(self, X: pd.DataFrame) -> np.ndarray:
        """Return (K, n, F+1), with the last column equal to each expert bias."""
        out = self._X(X)
        D = xgb.DMatrix(out, feature_names=self.feature_names)
        return np.stack([booster.predict(D, pred_contribs=True) for booster in self.experts])

    def explain(self, X: pd.DataFrame, row=0) -> dict:
        out = self._X(X)
        G = self.gating_proba(out)
        phi = self.expert_contribs(out)
        weights = G[row]
        base = float(np.sum(weights * phi[:, row, -1]))
        contributions = np.sum(weights[:, None] * phi[:, row, :-1], axis=0)
        prediction = float(self.predict(out)[row])
        recombined = base + float(contributions.sum())
        table = pd.DataFrame({
            "feature": self.feature_names,
            "contribution_usd": contributions,
            "abs_contribution_usd": np.abs(contributions),
        }).sort_values("abs_contribution_usd", ascending=False, ignore_index=True)
        return {
            "prediction_usd": prediction,
            "base_usd": base,
            "contributions": table,
            "gate_weights": dict(zip(self.expert_names, weights.tolist())),
            "dominant_expert": self.expert_names[int(np.argmax(weights))],
            "additivity_difference": prediction - recombined,
        }


def load(artifacts_dir):
    return MoEModel(artifacts_dir)