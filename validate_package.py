"""Nghiệm thu một gói mô hình trước khi giao cho UI.

    python validate_package.py artifacts/moe      # gói MoE (có moe_loader.py)
    python validate_package.py artifacts/xgb      # gói XGBoost (có model.json)

Mã thoát 0 nếu không có mục FAIL. Mục WARN không chặn nhưng nên đọc.
"""
import os, sys, time, traceback
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from core.features import DEFAULT_FEATURES, PRESETS, HEREDITARY, JOBS, build_row
from core.adapters import TOL, TOL_REL, within_tol, MoEAdapter, XGBAdapter, _load_moe_from_dir, contrib_matrix

RES = []


def rec(status, name, detail=""):
    RES.append((status, name, detail))
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""))


def make_test_rows(n_random=300, seed=7):
    """4 hồ sơ mẫu + hồ sơ ngẫu nhiên, trong đó khoảng 15% không rõ tuổi, BMI, huyết áp (NaN)."""
    rng = np.random.default_rng(seed)
    rows = [build_row(p) for p in PRESETS.values()]
    for _ in range(n_random):
        unk = lambda v: None if rng.random() < 0.15 else v
        rows.append(build_row(dict(
            age=unk(int(rng.integers(18, 65))), sex=int(rng.integers(0, 2)), weight=int(rng.integers(34, 96)),
            bmi=unk(round(float(rng.uniform(16, 53)), 1)), no_of_dependents=int(rng.integers(0, 6)),
            smoker=int(rng.random() < 0.36), bloodpressure=unk(int(rng.integers(40, 123))),
            diabetes=int(rng.random() < 0.56), regular_ex=int(rng.random() < 0.43),
            hereditary=str(rng.choice(HEREDITARY)), job=str(rng.choice(JOBS)))))
    return pd.concat(rows, ignore_index=True)


def unit_check(pred):
    m = float(np.mean(pred))
    if m < 50:
        rec("FAIL", "Đơn vị đầu ra", f"dự đoán trung bình {m:.2f}: có vẻ là log1p(claim). Hợp đồng yêu cầu đơn vị GỐC của claim (mẫu: 1.100 đến 63.800)")
    elif not (1000 <= m <= 60000):
        rec("WARN", "Đơn vị đầu ra", f"dự đoán trung bình {m:,.0f} lệch xa khoảng claim thật (1.100 đến 63.800)")
    else:
        rec("PASS", "Đơn vị đầu ra", f"dự đoán trung bình {m:,.0f}")


def check_test_metrics(meta):
    tm = (meta or {}).get("test_metrics") if isinstance(meta, dict) else None
    ok = isinstance(tm, dict) and all(k in tm and isinstance(tm[k], dict) and {"value", "ci_low", "ci_high"} <= set(tm[k]) for k in ("mae", "rmse", "r2"))
    rec("PASS" if ok else "WARN", "meta['test_metrics'] (mae, rmse, r2 với value/ci_low/ci_high)", "tab 'Về mô hình' sẽ trống nếu thiếu")
    if ok:
        mae = tm["mae"]["value"]
        if not (0 < mae < 20000): rec("WARN", "MAE test có hợp lý không", f"MAE={mae:,.1f}: kiểm tra đơn vị")


def validate_moe(directory):
    t0 = time.time()
    try:
        moe = _load_moe_from_dir(directory)
    except Exception:
        rec("FAIL", "Nạp moe_loader.load(artifacts_dir)", "\n" + traceback.format_exc()); return
    rec("PASS", "Nạp moe_loader.load(artifacts_dir)", f"{time.time()-t0:.1f}s" + ("" if time.time() - t0 < 30 else " (QUÁ CHẬM, yêu cầu < 30s)"))

    need = ["expert_names", "feature_names", "predict", "gating_proba", "expert_contribs"]
    miss = [a for a in need if not hasattr(moe, a)]
    if miss: rec("FAIL", "Có đủ thuộc tính/hàm bắt buộc", f"thiếu: {miss}"); return
    names = list(moe.expert_names); K = len(names); feats = list(moe.feature_names)
    rec("PASS" if K >= 2 and len(set(names)) == K and all(isinstance(n, str) for n in names) else "FAIL",
        "expert_names", f"K={K}: {names}")
    ok_f = len(feats) == len(set(feats)) and set(feats) == set(DEFAULT_FEATURES)
    rec("PASS" if ok_f else "FAIL", "feature_names = đúng 35 tên của dữ liệu",
        "" if ok_f else f"thiếu {sorted(set(DEFAULT_FEATURES)-set(feats))}, thừa {sorted(set(feats)-set(DEFAULT_FEATURES))}")
    if not ok_f: return

    X = make_test_rows(); n = len(X); F = len(feats); X_before = X.copy()
    try:
        pred = np.asarray(moe.predict(X), dtype=float)
        g = np.asarray(moe.gating_proba(X), dtype=float)
        c = np.asarray(moe.expert_contribs(X), dtype=float)
    except Exception:
        rec("FAIL", "Gọi predict/gating_proba/expert_contribs trên hồ sơ có NaN", "\n" + traceback.format_exc()); return
    rec("PASS", "Gọi được trên hồ sơ có NaN", f"{n} hồ sơ, trong đó {int(X.isna().any(axis=1).sum())} hồ sơ có giá trị không rõ")
    rec("PASS" if X.equals(X_before) else "FAIL", "Không sửa đổi DataFrame đầu vào")

    rec("PASS" if pred.shape == (n,) and np.isfinite(pred).all() else "FAIL", "predict: shape (n,) và hữu hạn", f"shape={pred.shape}")
    rec("PASS" if g.shape == (n, K) else "FAIL", "gating_proba: shape (n, K)", f"shape={g.shape}, cần {(n, K)}")
    rec("PASS" if c.shape == (K, n, F + 1) else "FAIL", "expert_contribs: shape (K, n, F+1)", f"shape={c.shape}, cần {(K, n, F + 1)}")
    if g.shape != (n, K) or c.shape != (K, n, F + 1) or pred.shape != (n,): return
    if not (np.isfinite(g).all() and np.isfinite(c).all()):
        rec("FAIL", "gating_proba và expert_contribs hữu hạn (không NaN/inf)"); return
    rec("PASS" if g.min() >= -1e-9 and np.abs(g.sum(axis=1) - 1).max() < 1e-6 else "FAIL",
        "gating: không âm và mỗi dòng cộng bằng 1", f"min={g.min():.3g}, lệch tổng lớn nhất={np.abs(g.sum(axis=1) - 1).max():.2g}")

    f_k = c.sum(axis=2)                      # (K, n): dự đoán của từng expert = bias + tổng SHAP
    gap = np.abs(pred - (g * f_k.T).sum(axis=1))
    ok_sum = within_tol(gap, pred)
    rec("PASS" if ok_sum else "FAIL", "predict = Σ_k g_k · f_k (điều kiện để SHAP gộp đúng)",
        f"lệch lớn nhất {gap.max():.4g} (cho phép {TOL} + {TOL_REL}·|dự đoán|); tại hồ sơ số {int(gap.argmax())}")
    if not ok_sum:
        rec("INFO", "Gợi ý", "kiểm tra: predict có clip/làm mượt/chuẩn hóa lại gating không? gating dùng để dự đoán có trùng gating_proba không (hard vs soft)? "
                             "expert có objective log-link không? hồ sơ lệch nhất ở trên là nơi nên debug trước")

    idx = list(range(0, n, max(1, n // 12)))[:12]
    d1 = max(np.abs(np.asarray(moe.predict(X.iloc[[i]]), float)[0] - pred[i]) for i in idx)
    d2 = max(np.abs(np.asarray(moe.gating_proba(X.iloc[[i]]), float)[0] - g[i]).max() for i in idx)
    rec("PASS" if d1 < 1e-3 and d2 < 1e-6 else "FAIL", "Kết quả 1 dòng = kết quả khi chạy theo lô", f"lệch predict {d1:.2g}, lệch gating {d2:.2g}")

    unit_check(pred)

    try:
        ad = MoEAdapter(moe); t0 = time.time(); worst = 0.0
        for p in PRESETS.values():
            e = ad.explain(build_row(p)); worst = max(worst, abs(e.base_value + e.contribs.sum() - e.prediction))
        dt = (time.time() - t0) / len(PRESETS)
        rec("PASS", "MoEAdapter.explain trên hồ sơ mẫu (UI chạy được)", f"{dt*1000:.0f} ms/hồ sơ" + ("" if dt < 1.5 else "  [WARN: chậm, UI sẽ giật]"))
    except Exception:
        rec("FAIL", "MoEAdapter.explain trên hồ sơ mẫu", "\n" + traceback.format_exc())

    meta = getattr(moe, "meta", None)
    check_test_metrics(meta)
    prof = getattr(moe, "expert_profiles", None)
    rec("PASS" if isinstance(prof, dict) and set(prof) == set(names) else "WARN", "expert_profiles (mô tả nghiệp vụ từng nhóm)", "nên có đủ K mục")
    adir = getattr(moe, "artifacts_dir", directory)
    have = [f for f in ("shap_summary.png", "shap_bar.png", "shap_groups.png", "shap_importance.csv") if os.path.exists(os.path.join(adir, f))]
    rec("PASS" if len(have) == 4 else "WARN", "Ảnh/bảng SHAP toàn cục", f"có {have}")


def validate_xgb(directory):
    try:
        ad = XGBAdapter(directory)
    except Exception:
        rec("FAIL", "Nạp XGBAdapter (model.json + meta.json)", "\n" + traceback.format_exc()); return
    rec("PASS", "Nạp XGBAdapter (model.json + meta.json)")
    ok_f = set(ad.feature_names) == set(DEFAULT_FEATURES)
    rec("PASS" if ok_f else "FAIL", "feature_names = đúng 35 tên của dữ liệu")
    X = make_test_rows(); pred = ad.predict(X); unit_check(pred)
    C = contrib_matrix(ad.booster, X)
    gap = np.abs(C.sum(axis=1) - pred)
    ok_x = within_tol(gap, pred)
    rec("PASS" if ok_x else "FAIL", "Base + Σ SHAP = dự đoán", f"lệch lớn nhất {gap.max():.4g}"
        + ("" if ok_x else "; objective phải là reg:squarederror (không dùng log-link/tweedie/gamma)"))
    for p in PRESETS.values(): ad.explain(build_row(p))
    rec("PASS", "XGBAdapter.explain trên hồ sơ mẫu")
    check_test_metrics(ad.meta)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    d = os.path.abspath(sys.argv[1])
    print(f"Nghiệm thu: {d}\n")
    if os.path.exists(os.path.join(d, "moe_loader.py")): validate_moe(d)
    elif os.path.exists(os.path.join(d, "model.json")): validate_xgb(d)
    else:
        rec("FAIL", "Nhận diện gói", "không thấy moe_loader.py (gói MoE) hoặc model.json (gói XGBoost) trong thư mục")
    nf = sum(1 for s, *_ in RES if s == "FAIL"); nw = sum(1 for s, *_ in RES if s == "WARN")
    print(f"\nKẾT QUẢ: {'ĐẠT' if nf == 0 else 'CHƯA ĐẠT'} ({nf} FAIL, {nw} WARN)")
    sys.exit(1 if nf else 0)