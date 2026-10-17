# Sinh notebook train XGBoost cho Kaggle. Chạy: python train/build_notebook.py
import json, inspect, sys, os, ast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import features

HERE = os.path.dirname(os.path.abspath(__file__))
from core import metrics as _metrics
GROUP_SRC = "SMOKER_GROUP = " + repr(sorted(features.SMOKER_GROUP)) + "\n\n" + inspect.getsource(features.group_of)
METRIC_SRC = "import numpy as np\n\n" + inspect.getsource(_metrics.compute_metrics) + "\n\n" + inspect.getsource(_metrics.bootstrap_ci)

cells = []
def md(s): cells.append(("md", s))
def code(s): cells.append(("code", s))

md("# 05_xgb_final: huấn luyện, tinh chỉnh và lưu mô hình XGBoost cho bản demo\n"
   "**Giao thức holdout** (thống nhất cả nhóm): file train 80% → cắt validation (20% của train, seed 42) để tune và chọn số cây "
   "(early stopping) → fit lại **toàn bộ train** với cấu hình đã chốt → chấm điểm **đúng một lần** trên file test 20% "
   "(kèm khoảng tin cậy bootstrap) → SHAP toàn cục → lưu `xgb_artifacts.zip`.\n\n"
   "Mô hình dùng **NaN gốc** (không impute) và mục tiêu `claim` gốc (`reg:squarederror`), nên SHAP cộng trực tiếp ra đơn vị claim. "
   "Không được quay lại chỉnh tham số sau khi đã xem điểm test.")

code("""import os, glob, json, time, shutil, warnings
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, train_test_split
warnings.filterwarnings("ignore")
print("xgboost", xgb.__version__)

# ====== CẤU HÌNH: sửa 4 dòng này cho khớp dataset trên Kaggle ======
TRAIN_NAME, TEST_NAME = "train", "test"   # tên file CSV chứa chuỗi này (không phân biệt hoa thường)
TARGET = "claim"
DROP = ["Unnamed: 0"]                      # cột index thừa nếu có
N_ITER = 40                                # số cấu hình thử ngẫu nhiên (tăng 80 nếu còn quota)
SEED, VAL_FRAC = 42, 0.2                   # PHẢI giống thành viên làm MoE
# ===================================================================

def find_csv(key):
    hits = [f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True) if key in os.path.basename(f).lower()]
    assert len(hits) == 1, f"Cần đúng 1 file chứa '{key}', thấy: {hits}"
    return hits[0]
def load(path):
    df = pd.read_csv(path).drop(columns=[c for c in DROP if c in pd.read_csv(path, nrows=1).columns])
    return df.drop(columns=[TARGET]), df[TARGET].to_numpy(float)
Xtr, ytr = load(find_csv(TRAIN_NAME)); Xte, yte = load(find_csv(TEST_NAME))
assert list(Xtr.columns) == list(Xte.columns), "train và test khác cột"
assert Xtr.shape[1] == 35, f"Kỳ vọng 35 feature sau engineering, thực tế {Xtr.shape[1]}"
print("train", Xtr.shape, "| test", Xte.shape, "| claim train mean %.0f" % ytr.mean())

WORK = "/kaggle/working"; ART = f"{WORK}/artifacts/xgb"; os.makedirs(ART, exist_ok=True)
XGB_FIXED = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)  # mốc chưa tune
""" + GROUP_SRC + "\n\n" + METRIC_SRC)

md("## 1. Chia train/validation và tinh chỉnh tham số")
code("""Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=VAL_FRAC, random_state=SEED, shuffle=True)
print("train-con", Xa.shape, "| validation", Xv.shape)

# RandomizedSearchCV với PredefinedSplit = đúng MỘT lần chia (train-con / validation), không K-fold
Xs = pd.concat([Xa, Xv]); ys = np.concatenate([ya, yv])
ps = PredefinedSplit(np.r_[np.full(len(Xa), -1), np.zeros(len(Xv), int)])
space = dict(
    n_estimators=[300, 500, 800, 1200], max_depth=[3, 4, 5, 6, 8], learning_rate=[0.01, 0.03, 0.05, 0.1],
    subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.5, 0.7, 0.9, 1.0], min_child_weight=[1, 3, 5, 10],
    reg_lambda=[0.5, 1, 5, 10], reg_alpha=[0, 0.1, 1], gamma=[0, 0.1, 0.5])
base = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=-1)
search = RandomizedSearchCV(base, space, n_iter=N_ITER, scoring="neg_mean_absolute_error", cv=ps,
                            random_state=SEED, n_jobs=1, refit=False, verbose=1)
t0 = time.time(); search.fit(Xs, ys)
best = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
print(f"Xong sau {time.time()-t0:.0f}s | MAE validation tốt nhất: {-search.best_score_:,.1f}")
print(json.dumps(best, indent=1))

# Chọn số cây bằng early stopping trên validation (với tham số đã chọn)
es_params = {k: v for k, v in best.items() if k != "n_estimators"}
es = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=-1,
                      n_estimators=3000, early_stopping_rounds=50, eval_metric="mae", **es_params)
es.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
best_n = int(es.best_iteration) + 1
best["n_estimators"] = best_n
val_metrics = compute_metrics(yv, es.predict(Xv))
print("Số cây chốt (early stopping):", best_n, "| validation:", {k: round(v, 4) for k, v in val_metrics.items()})
pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(10)[["rank_test_score", "mean_test_score", "params"]]""")

md("## 2. Fit lại toàn bộ train với cấu hình đã chốt và chấm điểm trên test (một lần)")
code("""final = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=-1, **best).fit(Xtr, ytr)
fixed = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=-1, **XGB_FIXED).fit(Xtr, ytr)
test_metrics = bootstrap_ci(yte, final.predict(Xte))
fixed_metrics = bootstrap_ci(yte, fixed.predict(Xte))
tbl = pd.concat({"XGB tuned": pd.DataFrame(test_metrics).T, "XGB_FIXED (chưa tune)": pd.DataFrame(fixed_metrics).T}, axis=1).round(4)
print(tbl.to_string())""")

md("## 3. Lưu mô hình tốt nhất")
code("""final.save_model(f"{ART}/model.json")
booster = final.get_booster()
rng_cols = ["age", "weight", "bmi", "no_of_dependents", "bloodpressure"]
bias = float(booster.predict(xgb.DMatrix(Xtr.iloc[:1], feature_names=list(Xtr.columns)), pred_contribs=True)[0, -1])
meta = dict(
    model="XGBRegressor (objective reg:squarederror, NaN gốc)", xgboost_version=xgb.__version__, n_rows=int(len(ytr)),
    target_unit="đơn vị gốc của claim", feature_names=list(Xtr.columns),
    feature_groups={f: group_of(f) for f in Xtr.columns}, best_params=best,
    protocol=dict(type="holdout", split="train/val %.0f%% seed %d để tune; fit lại toàn bộ train; chấm trên test" % (VAL_FRAC * 100, SEED),
                  n_train=int(len(ytr)), n_test=int(len(yte)), n_iter=N_ITER, scoring="MAE", ci="bootstrap 1000 lần, 95%"),
    test_metrics=test_metrics, test_metrics_xgb_fixed=fixed_metrics, val_metrics=val_metrics,
    ranges={c: dict(min=float(Xtr[c].min()), max=float(Xtr[c].max()), median=float(Xtr[c].median())) for c in rng_cols if c in Xtr},
    claim=dict(mean=float(ytr.mean()), median=float(np.median(ytr)), min=float(ytr.min()), max=float(ytr.max())),
    expected_value=bias)
json.dump(meta, open(f"{ART}/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Đã lưu:", os.listdir(ART), "| mức nền (bias):", round(bias, 1))""")

md("## 4. SHAP toàn cục (TreeSHAP gốc của XGBoost, 2.000 dòng mẫu của train)")
code("""import matplotlib; import matplotlib.pyplot as plt
X = Xtr; Xs2 = X.sample(min(2000, len(X)), random_state=SEED)
contrib = booster.predict(xgb.DMatrix(Xs2, feature_names=list(X.columns)), pred_contribs=True)
sv, pr = contrib[:, :-1], final.predict(Xs2)
assert np.abs(contrib.sum(axis=1) - pr).max() < 0.05, "SHAP không cộng đủ"
print("Kiểm tra: base + tổng SHAP = dự đoán (sai số lớn nhất %.4f)" % np.abs(contrib.sum(axis=1) - pr).max())

imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
imp["group"] = imp.feature.map(meta["feature_groups"]); imp = imp.sort_values("mean_abs_shap", ascending=False)
imp.to_csv(f"{ART}/shap_importance.csv", index=False)

top = imp.head(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(top.feature, top.mean_abs_shap, color="#1f3864")
ax.set_xlabel("mean |SHAP| (đơn vị claim)"); ax.set_title("Top-15 đặc trưng"); fig.tight_layout(); fig.savefig(f"{ART}/shap_bar.png", dpi=150); plt.show()

grp = pd.DataFrame(sv, columns=X.columns).T.groupby(meta["feature_groups"]).sum().T.abs().mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 3.5)); ax.barh(grp.index, grp.values, color="#c0392b")
ax.set_xlabel("mean |tổng SHAP của nhóm|"); ax.set_title("Ảnh hưởng theo nhóm"); fig.tight_layout(); fig.savefig(f"{ART}/shap_groups.png", dpi=150); plt.show()

try:
    import shap
    shap.summary_plot(sv, Xs2, show=False, max_display=15); plt.tight_layout(); plt.savefig(f"{ART}/shap_summary.png", dpi=150, bbox_inches="tight"); plt.show()
except Exception as ex:
    print("Bỏ qua beeswarm (shap lỗi hoặc chưa cài):", repr(ex))
print(imp.head(10).round(1).to_string(index=False))""")

md("## 5. Kiểm tra nạp lại mô hình và đóng gói")
code("""chk = xgb.XGBRegressor(); chk.load_model(f"{ART}/model.json")
assert np.allclose(chk.predict(Xte.iloc[:200]), final.predict(Xte.iloc[:200])), "Mô hình nạp lại cho kết quả khác"
print("OK: mô hình nạp lại cho kết quả giống hệt")
zp = shutil.make_archive(f"{WORK}/xgb_artifacts", "zip", ART)
print("Đã tạo:", zp, "| file:", sorted(os.listdir(ART)))
print("Save Version, rồi tải xgb_artifacts.zip từ tab Output và giải nén vào demo/artifacts/xgb/")""")

def src(s):
    L = s.split("\n"); return [l + "\n" for l in L[:-1]] + ([L[-1]] if L[-1] else [])

out = []
for k, s in cells:
    if k == "code": ast.parse(s)
    out.append({"cell_type": "markdown" if k == "md" else "code", "metadata": {}, "source": src(s),
                **({} if k == "md" else {"execution_count": None, "outputs": []})})
nb = {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                  "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train", "05_xgb_final.ipynb")
json.dump(nb, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("wrote", dest, len(out), "cells")
