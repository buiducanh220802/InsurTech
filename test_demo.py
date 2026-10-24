"""Kiểm thử tự động cho bản demo (không cần mở giao diện).

    python test_demo.py                      # chạy bộ test chính
    python test_demo.py --test D:\\...\\test.csv   # thêm: tính lại MAE/R2 trên test và so với meta.json

Mã thoát 0 nếu không có FAIL. WARN là điều nên đọc nhưng không chặn.
"""
import argparse, os, sys, time, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from core.features import (DEFAULT_FEATURES, HEREDITARY, JOBS, PRESETS, build_row, engineer, KEPT_JOBS)
from core.adapters import load_models, within_tol
from core.explain import make_bill, waterfall
from core.metrics import compute_metrics

R = []
def rec(s, name, detail=""):
    R.append(s); print(f"[{s}] {name}" + (f": {detail}" if detail else ""))
def check(cond, name, detail="", warn=False):
    rec("PASS" if cond else ("WARN" if warn else "FAIL"), name, detail)

def rand_profile(rng, unk=0.15):
    u = lambda v: None if rng.random() < unk else v
    return dict(age=u(int(rng.integers(18, 65))), sex=int(rng.integers(0, 2)), weight=int(rng.integers(34, 96)),
                bmi=u(round(float(rng.uniform(16, 53)), 1)), no_of_dependents=int(rng.integers(0, 6)),
                smoker=int(rng.random() < .4), bloodpressure=u(int(rng.integers(40, 123))), diabetes=int(rng.random() < .5),
                regular_ex=int(rng.random() < .5), hereditary=str(rng.choice(HEREDITARY)), job=str(rng.choice(JOBS)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--test", default=None, help="CSV test (35 cột + claim) để đối chiếu điểm với meta.json")
    a = ap.parse_args(); rng = np.random.default_rng(2026)

    print("== A. Engineering (build_row)")
    X = build_row(PRESETS["45 tuổi, hút thuốc, bệnh nền"]); check(list(X.columns) == DEFAULT_FEATURES and X.shape == (1, 35), "35 cột đúng tên và thứ tự")
    check(X.loc[0, "is_smoker_obese"] == 1 and X.loc[0, "is_smoker_senior"] == 1, "Hút thuốc + BMI 31 + tuổi 45: hai cờ = 1")
    edge = build_row(dict(PRESETS["Trẻ khỏe"], age=45, bmi=30.0, smoker=1)); check(edge.loc[0, "is_smoker_obese"] == 1 and edge.loc[0, "is_smoker_senior"] == 1, "Biên bmi=30, age=45 tính là đạt (>=)")
    edge = build_row(dict(PRESETS["Trẻ khỏe"], age=44, bmi=29.9, smoker=1)); check(edge.loc[0, "is_smoker_obese"] == 0 and edge.loc[0, "is_smoker_senior"] == 0, "Sát biên bmi=29.9, age=44: hai cờ = 0")
    nu = build_row(dict(PRESETS["Trẻ khỏe"], age=None, bmi=None, bloodpressure=None, smoker=1, diabetes=1))
    check(all(np.isnan(nu.loc[0, c]) for c in ["age", "bmi", "bloodpressure", "Age_x_Smoker", "Age_x_Diabetic", "BMI_x_Smoker", "BP_per_Age"]), "Không rõ age/bmi/bp: các cột liên quan = NaN")
    check(nu.loc[0, "is_smoker_obese"] == 0 and nu.loc[0, "is_smoker_senior"] == 0, "Không rõ age/bmi: hai cờ = 0 (như dữ liệu thật)")
    nh = build_row(dict(PRESETS["Trẻ khỏe"], hereditary="Alzheimer (baseline)", job="Actor (baseline)"))
    check(nh.filter(like="hereditary_").sum(axis=1).iloc[0] == 0 and nh.filter(like="job_title_").sum(axis=1).iloc[0] == 0, "Alzheimer/Actor (baseline): toàn bộ cột one-hot = 0")
    jo = build_row(dict(PRESETS["Trẻ khỏe"], job="Other (25 nghề còn lại)")); check(jo.loc[0, "job_title_Other"] == 1 and jo.filter(like="job_title_").sum(axis=1).iloc[0] == 1, "Other: chỉ bật job_title_Other")
    raws = []
    for _ in range(500):
        p = rand_profile(rng); raws.append(dict(age=np.nan if p["age"] is None else p["age"], sex="male" if p["sex"] else "female", weight=p["weight"],
            bmi=np.nan if p["bmi"] is None else p["bmi"], hereditary_diseases=p["hereditary"].split(" (")[0], no_of_dependents=p["no_of_dependents"], smoker=p["smoker"],
            bloodpressure=np.nan if p["bloodpressure"] is None else p["bloodpressure"], diabetes=p["diabetes"], regular_ex=p["regular_ex"], job_title=p["job"].split(" (")[0].replace("Other", "Plumber")))
        raws[-1]["_p"] = p
    E = engineer(pd.DataFrame([{k: v for k, v in r.items() if k != "_p"} for r in raws]))
    B = pd.concat([build_row(r["_p"]) for r in raws], ignore_index=True)
    check(np.allclose(E.to_numpy(float), B.to_numpy(float), equal_nan=True), "engineer(thô) trùng build_row trên 500 hồ sơ ngẫu nhiên")

    print("== B. Nạp mô hình")
    models, notes = load_models(a.artifacts)
    for n in notes: print("   ghi chú:", n)
    check(len(models) >= 1, "Nạp được ít nhất 1 mô hình", ", ".join(models) or "không có")
    check(len(models) == 2, "Nạp được cả XGBoost và MoE", f"đang có {len(models)}", warn=True)

    profiles = [rand_profile(rng) for _ in range(300)]
    extremes = {
        "tất cả không rõ": dict(PRESETS["Trẻ khỏe"], age=None, bmi=None, bloodpressure=None),
        "tuổi nhỏ nhất 18": dict(PRESETS["Trẻ khỏe"], age=18), "tuổi lớn nhất 64": dict(PRESETS["Trẻ khỏe"], age=64),
        "BMI nhỏ nhất 16": dict(PRESETS["Trẻ khỏe"], bmi=16.0), "BMI lớn nhất 53.1": dict(PRESETS["Trẻ khỏe"], bmi=53.1),
        "người phụ thuộc 0": dict(PRESETS["Trẻ khỏe"], no_of_dependents=0), "người phụ thuộc 5": dict(PRESETS["Trẻ khỏe"], no_of_dependents=5),
        "huyết áp 40": dict(PRESETS["Trẻ khỏe"], bloodpressure=40), "huyết áp 122": dict(PRESETS["Trẻ khỏe"], bloodpressure=122),
        "nặng nhất + hút thuốc + tiểu đường": dict(PRESETS["Phí cao"], weight=95, bmi=53.1, age=64),
        "NGOÀI khoảng: tuổi 100": dict(PRESETS["Trẻ khỏe"], age=100), "NGOÀI khoảng: BMI 80": dict(PRESETS["Trẻ khỏe"], bmi=80.0),
        "NGOÀI khoảng: cân nặng 150": dict(PRESETS["Trẻ khỏe"], weight=150)}
    for name, m in models.items():
        print(f"== C. {name}")
        try:
            Xb = pd.concat([build_row(p) for p in profiles], ignore_index=True)
            pred = np.asarray(m.predict(Xb), float)
            check(np.isfinite(pred).all() and pred.shape == (len(Xb),), "Dự đoán hữu hạn trên 300 hồ sơ ngẫu nhiên (có NaN)")
            check(pred.min() > 0, "Mọi dự đoán dương", f"nhỏ nhất {pred.min():,.0f}", warn=True)
            check(1000 < pred.min() and pred.max() < 80000, "Dự đoán nằm trong khoảng hợp lý của claim", f"{pred.min():,.0f} đến {pred.max():,.0f}", warn=True)
            bad = 0; worst = 0.0; t0 = time.time(); fig_err = 0
            for i, p in enumerate(profiles[:100]):
                e = m.explain(build_row(p))
                gap = abs(e.base_value + e.contribs.sum() - e.prediction); worst = max(worst, gap); bad += not within_tol(gap, e.prediction)
                for grouped in (True, False):
                    for top in (3, 8, 15):
                        try:
                            b = make_bill(e, build_row(p), m.groups, grouped=grouped, top_n=top)
                            if i < 6: fig = waterfall(b, "USD"); plt.close(fig)
                        except Exception: fig_err += 1
            dt = (time.time() - t0) / 100
            check(bad == 0, "Base + Σ đóng góp = dự đoán trên 100 hồ sơ", f"lệch lớn nhất {worst:.3g}")
            check(fig_err == 0, "Hóa đơn (gộp/không gộp, top 3/8/15) và waterfall không lỗi", f"{fig_err} lỗi")
            check(dt < 1.5, "Thời gian giải thích mỗi hồ sơ (kể cả dựng hóa đơn)", f"{dt*1000:.0f} ms", warn=True)
            x1 = build_row(profiles[0]); check(np.allclose(m.predict(x1), m.predict(x1)) and np.isclose(m.predict(x1)[0], pred[0], rtol=1e-6), "Xác định: chạy lại hoặc chạy theo lô cho cùng kết quả")
            for nm, p in extremes.items():
                try:
                    e = m.explain(build_row(p)); ok = np.isfinite(e.prediction) and within_tol(abs(e.base_value + e.contribs.sum() - e.prediction), e.prediction)
                    check(ok, f"Biên: {nm}", f"{e.prediction:,.0f} USD")
                except Exception as ex:
                    check(False, f"Biên: {nm}", repr(ex))
            ps = [m.predict(build_row(dict(p, smoker=1)))[0] - m.predict(build_row(dict(p, smoker=0)))[0] for p in profiles[:100]]
            frac = float(np.mean(np.array(ps) > 0)); check(frac >= 0.85, "Hợp lý: bật hút thuốc làm tăng dự đoán ở phần lớn hồ sơ", f"{frac:.0%} hồ sơ tăng", warn=True)
            pa = [m.predict(build_row(dict(p, age=60)))[0] - m.predict(build_row(dict(p, age=22)))[0] for p in profiles[:100]]
            frac = float(np.mean(np.array(pa) > 0)); check(frac >= 0.85, "Hợp lý: tuổi 60 cao hơn tuổi 22 ở phần lớn hồ sơ", f"{frac:.0%} hồ sơ tăng", warn=True)
            if "gating" in m.explain(build_row(profiles[0])).extras:
                gs = np.array([m.explain(build_row(p)).extras["gating"].to_numpy() for p in profiles[:50]])
                check(np.allclose(gs.sum(axis=1), 1, atol=1e-5) and (gs >= -1e-9).all(), "Trọng số gate: không âm, cộng bằng 1")
                check(gs.mean(axis=0).max() < 0.99, "Gate không bị suy biến về một expert duy nhất", str(gs.mean(axis=0).round(2)), warn=True)
            if a.test:
                df = pd.read_csv(a.test); Xt = df[DEFAULT_FEATURES] if set(DEFAULT_FEATURES) <= set(df.columns) else engineer(df)
                met = compute_metrics(df["claim"].to_numpy(float), np.asarray(m.predict(Xt), float)); ref = ((m.meta or {}).get("test_metrics") or {}).get("mae", {}).get("value")
                print(f"   MAE test tính lại = {met['mae']:,.1f} | R2 = {met['r2']:.4f} | meta: {ref}")
                if ref: check(abs(met["mae"] - ref) / ref < 0.01, "MAE tính lại khớp meta.json (cùng mô hình đã báo cáo)", f"lệch {abs(met['mae']-ref)/ref:.2%}")
        except Exception:
            check(False, f"{name}: lỗi bất ngờ", "\n" + traceback.format_exc())

    if len(models) == 2:
        print("== D. So sánh hai mô hình")
        names = list(models); d = [abs(models[names[0]].predict(build_row(p))[0] - models[names[1]].predict(build_row(p))[0]) for p in profiles[:100]]
        check(np.median(d) < 8000, "Hai mô hình cho kết quả cùng bậc (chênh lệch trung vị)", f"{np.median(d):,.0f} USD", warn=True)
    nf, nw = R.count("FAIL"), R.count("WARN"); print(f"\nKẾT QUẢ: {'ĐẠT' if nf == 0 else 'CHƯA ĐẠT'} ({R.count('PASS')} PASS, {nf} FAIL, {nw} WARN)"); sys.exit(1 if nf else 0)

if __name__ == "__main__":
    main()