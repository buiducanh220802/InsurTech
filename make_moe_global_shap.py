"""Tạo SHAP toàn cục cho gói MoE (cùng định dạng file như gói XGBoost).

    python make_moe_global_shap.py --data D:\\duong_dan\\train.csv

Dùng đúng moe_loader.py trong artifacts/moe: đóng góp của đặc trưng j = Σ_k g_k · SHAP_kj (như UI).
Sinh ra trong artifacts/moe/:  shap_bar.png, shap_groups.png, shap_importance.csv, shap_experts.png
--data có thể là file đã engineering (35 cột) hoặc dữ liệu thô 12 cột; cột claim (nếu có) bị bỏ qua.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from core.features import DEFAULT_FEATURES, engineer, group_of
from core.adapters import _load_moe_from_dir, combine_expert_contribs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV train (35 cột đã engineering, hoặc 12 cột thô)")
    ap.add_argument("--dir", default=os.path.join(ROOT, "artifacts", "moe"))
    ap.add_argument("--n", type=int, default=2000, help="số hồ sơ lấy mẫu")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    X = df[DEFAULT_FEATURES] if set(DEFAULT_FEATURES) <= set(df.columns) else engineer(df)
    X = X.sample(min(a.n, len(X)), random_state=a.seed).reset_index(drop=True)

    moe = _load_moe_from_dir(a.dir)
    feats = list(moe.feature_names)
    g = np.asarray(moe.gating_proba(X), float)
    c = np.asarray(moe.expert_contribs(X), float)
    base, phi = combine_expert_contribs(g, c)
    pred = np.asarray(moe.predict(X), float)
    gap = np.abs(base + phi.sum(axis=1) - pred).max()
    assert gap < 0.05 + 2e-6 * np.abs(pred).max(), f"SHAP không cộng đủ (lệch {gap:.4f})"
    print(f"Kiểm tra: base + tổng SHAP = dự đoán (lệch lớn nhất {gap:.2e}) trên {len(X)} hồ sơ")

    names = list(moe.expert_names)
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(phi).mean(axis=0)})
    for k, nm in enumerate(names):     # tầm quan trọng trong từng expert (không nhân gate)
        imp[f"mean_abs_shap_{nm}"] = np.abs(c[k, :, :-1]).mean(axis=0)
    imp["group"] = imp.feature.map(group_of)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(a.dir, "shap_importance.csv"), index=False)

    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(top.feature, top.mean_abs_shap, color="#1f3864")
    ax.set_xlabel("mean |SHAP| (USD) sau khi trộn theo gate"); ax.set_title("MoE: top-15 đặc trưng")
    fig.tight_layout(); fig.savefig(os.path.join(a.dir, "shap_bar.png"), dpi=150); plt.close(fig)

    grp = pd.DataFrame(phi, columns=feats).T.groupby(lambda f: group_of(f)).sum().T.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.5)); ax.barh(grp.index, grp.values, color="#c0392b")
    ax.set_xlabel("mean |tổng SHAP của nhóm| (USD)"); ax.set_title("MoE: ảnh hưởng theo nhóm")
    fig.tight_layout(); fig.savefig(os.path.join(a.dir, "shap_groups.png"), dpi=150); plt.close(fig)

    # So sánh giữa các expert: đặc trưng nào mỗi expert dựa vào nhiều nhất
    top_f = imp.head(10).feature.tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8, 5)); h = 0.8 / len(names); y = np.arange(len(top_f))
    for k, nm in enumerate(names):
        v = imp.set_index("feature").loc[top_f, f"mean_abs_shap_{nm}"].values
        ax.barh(y + k * h, v, height=h, label=nm)
    ax.set_yticks(y + 0.4 - h / 2); ax.set_yticklabels(top_f); ax.legend()
    ax.set_xlabel("mean |SHAP| trong từng expert (USD)"); ax.set_title("Khác biệt giữa các expert")
    fig.tight_layout(); fig.savefig(os.path.join(a.dir, "shap_experts.png"), dpi=150); plt.close(fig)

    print("Gate trung bình:", dict(zip(names, g.mean(axis=0).round(3))))
    print(imp[["feature", "mean_abs_shap"]].head(10).round(0).to_string(index=False))
    print("Đã ghi ảnh/bảng vào", a.dir)


if __name__ == "__main__":
    main()