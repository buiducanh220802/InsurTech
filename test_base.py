import json, numpy as np, xgboost as xgb

with open("artifacts/xgb/meta.json", encoding="utf-8") as f:
    print("XGB base:", json.load(f)["expected_value"])

for k in (0, 1):
    b = xgb.Booster()
    b.load_model(f"artifacts/moe/expert_{k}.json")
    d = xgb.DMatrix(np.full((1, 35), np.nan))
    print(f"Expert {k} bias:", b.predict(d, pred_contribs=True, validate_features=False)[0, -1])