"""Chuyển Explanation thành 'hóa đơn' (Base + từng khoản = Tổng) và biểu đồ thác nước."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import pretty_value

BASE_LABEL, REST_LABEL, TOTAL_LABEL = "Mức nền (trung bình của mô hình)", "Các yếu tố còn lại", "TỔNG DỰ ĐOÁN"


def make_bill(expl, x_row: pd.DataFrame, groups: dict, grouped: bool = True, top_n: int = 8) -> pd.DataFrame:
    """Trả về DataFrame: Khoản mục | Giá trị nhập | Ảnh hưởng. Tổng cột 'Ảnh hưởng' = dự đoán (đã kiểm tra)."""
    c = expl.contribs
    items = []
    if grouped:
        for gname, idx in c.groupby(pd.Series(groups).reindex(c.index).fillna("Khác")).groups.items():
            feats = list(idx)
            shown = [f"{f}={pretty_value(x_row.iloc[0][f])}" for f in feats
                     if not (f.startswith(("hereditary_", "job_title_")) and x_row.iloc[0][f] == 0)]
            items.append((gname, "; ".join(shown), float(c[feats].sum())))
    else:
        for f, v in c.items():
            items.append((f, pretty_value(x_row.iloc[0][f]), float(v)))
    items.sort(key=lambda t: -abs(t[2]))
    keep, rest = items[:top_n], items[top_n:]
    rows = [(BASE_LABEL, "", float(expl.base_value))] + keep
    if rest: rows.append((REST_LABEL, f"{len(rest)} mục nhỏ", float(sum(t[2] for t in rest))))
    rows.append((TOTAL_LABEL, "", float(expl.prediction)))
    df = pd.DataFrame(rows, columns=["Khoản mục", "Giá trị nhập", "Ảnh hưởng"])
    assert abs(df["Ảnh hưởng"].iloc[:-1].sum() - expl.prediction) < 0.05, "Hóa đơn không cộng đủ"
    return df


def waterfall(bill: pd.DataFrame, unit: str = "USD"):
    import matplotlib
    import matplotlib.pyplot as plt
    labels = bill["Khoản mục"].tolist(); vals = bill["Ảnh hưởng"].to_numpy(float)
    n = len(vals); starts = np.zeros(n); widths = np.zeros(n); colors = []
    cum = 0.0
    for i in range(n - 1):
        if i == 0:
            starts[i], widths[i] = 0.0, vals[i]; cum = vals[i]; colors.append("#7f8c8d")
        else:
            starts[i], widths[i] = cum, vals[i]; cum += vals[i]; colors.append("#c0392b" if vals[i] > 0 else "#2e7d32")
    starts[-1], widths[-1] = 0.0, vals[-1]; colors.append("#1f3864")
    fig, ax = plt.subplots(figsize=(8, 0.5 * n + 1.2))
    y = np.arange(n)[::-1]
    ax.barh(y, widths, left=starts, color=colors, height=0.6)
    for yi, s, w, v in zip(y, starts, widths, vals):
        ax.text(max(s, s + w) + abs(vals).max() * 0.01, yi, f"{v:+,.0f}" if yi != y[0] and yi != y[-1] else f"{v:,.0f}",
                va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([l if len(l) < 38 else l[:36] + "…" for l in labels], fontsize=8)
    ax.set_xlabel(f"{unit}"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(starts + np.maximum(widths, 0)) * 1.12)
    fig.tight_layout()
    return fig


def _selftest():
    import numpy as np
    from .adapters import Explanation, combine_expert_contribs
    from .features import default_groups, build_row, PRESETS
    rng = np.random.default_rng(1)
    feats = list(build_row(PRESETS["Phí cao"]).columns)
    X1 = build_row(PRESETS["Phí cao"])
    phi = rng.normal(0, 800, len(feats)); base = 15000.0
    e = Explanation(base + phi.sum(), base, pd.Series(phi, index=feats)); e.check()
    for grouped in (True, False):
        b = make_bill(e, X1, default_groups(feats), grouped=grouped, top_n=5)
        assert abs(b["Ảnh hưởng"].iloc[:-1].sum() - e.prediction) < 1e-6
    # MoE: gộp K expert phải cộng đúng bằng sum_k g_k * f_k
    K, n, F = 3, 4, len(feats)
    g = rng.dirichlet(np.ones(K), size=n); c = rng.normal(0, 500, (K, n, F + 1)); c[:, :, -1] += 12000
    base_m, phi_m = combine_expert_contribs(g, c)
    f_k = c.sum(axis=2)                                   # f_k(x) = bias + sum phi
    y = (g * f_k.T).sum(axis=1)
    assert np.allclose(base_m + phi_m.sum(axis=1), y)
    print("explain selftest OK")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.explain import _selftest as t; t()
