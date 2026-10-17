"""Dựng 35 feature từ các ô nhập của UI, đúng quy tắc của health_insurance.csv.

Quy tắc quan trọng (đã kiểm tra trên dữ liệu thật):
- Cột tương tác được tính SAU khi gán NaN: age NaN -> Age_x_Smoker, Age_x_Diabetic, BP_per_Age đều NaN.
- is_smoker_obese / is_smoker_senior: so sánh với NaN cho False -> bằng 0 (dữ liệu thật cũng như vậy).
- one-hot hereditary_diseases_* : 'Alzheimer' là baseline (tất cả = 0).
- one-hot job_title_* : 10 nghề phổ biến giữ riêng, 25 nghề còn lại gom thành 'Other', baseline bị bỏ là 'Actor'
  (tất cả cột job_title_* = 0 nghĩa là Actor). Theo description_health_data.md.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HEREDITARY = ["Alzheimer", "Arthritis", "Cancer", "Diabetes", "Epilepsy", "EyeDisease",
              "HeartDisease", "High BP", "NoDisease", "Obesity"]
JOBS = ["Actor", "Architect", "Dancer", "Engineer", "FilmMaker", "HomeMakers",
        "HouseKeeper", "Photographer", "Singer", "Student", "Other"]
KEPT_JOBS = ["Actor", "Architect", "Dancer", "Engineer", "FilmMaker", "HomeMakers", "HouseKeeper",
             "Photographer", "Singer", "Student"]          # các nghề còn lại trong dữ liệu thô -> "Other"

# Thứ tự cột mặc định (trùng profile). Khi có meta.json, thứ tự trong meta được ưu tiên.
DEFAULT_FEATURES = (
    ["age", "sex", "weight", "bmi", "no_of_dependents", "smoker", "bloodpressure", "diabetes", "regular_ex",
     "Age_x_Smoker", "Age_x_Diabetic", "BMI_x_Smoker", "BP_per_Age", "Dependents_x_Smoker",
     "is_smoker_obese", "is_smoker_senior"]
    + [f"hereditary_diseases_{h}" for h in ["Arthritis", "Cancer", "Diabetes", "Epilepsy", "EyeDisease",
                                              "HeartDisease", "High BP", "NoDisease", "Obesity"]]
    + [f"job_title_{j}" for j in ["Architect", "Dancer", "Engineer", "FilmMaker", "HomeMakers", "HouseKeeper",
                                    "Other", "Photographer", "Singer", "Student"]]
)

SMOKER_GROUP = {"smoker", "Age_x_Smoker", "BMI_x_Smoker", "Dependents_x_Smoker", "is_smoker_obese", "is_smoker_senior"}


def group_of(feature: str) -> str:
    """Nhóm hiển thị cho 'bill' gộp. Chỉ là lựa chọn trình bày; các cột hút thuốc đa cộng tuyến nên gộp lại."""
    if feature in SMOKER_GROUP: return "Hút thuốc (và các cột tương tác)"
    if feature in {"age", "bloodpressure", "BP_per_Age"}: return "Tuổi và huyết áp"
    if feature in {"weight", "bmi", "sex"}: return "Thể trạng"
    if feature in {"diabetes", "Age_x_Diabetic", "regular_ex"}: return "Tiểu đường và lối sống"
    if feature == "no_of_dependents": return "Người phụ thuộc"
    if feature.startswith("hereditary_diseases_"): return "Bệnh di truyền gia đình"
    if feature.startswith("job_title_"): return "Nghề nghiệp"
    return "Khác"


def default_groups(features) -> dict:
    return {f: group_of(f) for f in features}


def _nan(v):
    return np.nan if v is None else float(v)


def build_row(inp: dict, feature_names=None) -> pd.DataFrame:
    """inp: age (None = không rõ), sex (1 nam / 0 nữ), weight, bmi (None), no_of_dependents, smoker (0/1),
    bloodpressure (None), diabetes (0/1), regular_ex (0/1), hereditary (tên trong HEREDITARY), job (tên trong JOBS)."""
    age, bmi, bp = _nan(inp.get("age")), _nan(inp.get("bmi")), _nan(inp.get("bloodpressure"))
    smoker, diab = float(inp["smoker"]), float(inp["diabetes"])
    dep = float(inp["no_of_dependents"])
    row = {
        "age": age, "sex": float(inp["sex"]), "weight": float(inp["weight"]), "bmi": bmi,
        "no_of_dependents": dep, "smoker": smoker, "bloodpressure": bp, "diabetes": diab,
        "regular_ex": float(inp["regular_ex"]),
        "Age_x_Smoker": age * smoker, "Age_x_Diabetic": age * diab, "BMI_x_Smoker": bmi * smoker,
        "BP_per_Age": bp / age if (age == age and bp == bp and age != 0) else np.nan,
        "Dependents_x_Smoker": dep * smoker,
        "is_smoker_obese": float(bool(bmi >= 30) and smoker == 1),   # NaN >= 30 là False
        "is_smoker_senior": float(bool(age >= 45) and smoker == 1),
    }
    for h in DEFAULT_FEATURES:
        if h.startswith("hereditary_diseases_"):
            row[h] = 1.0 if h.split("hereditary_diseases_")[1] == inp["hereditary"].split(" (")[0] else 0.0
        elif h.startswith("job_title_"):
            row[h] = 1.0 if h.split("job_title_")[1] == inp["job"].split(" (")[0] else 0.0
    cols = list(feature_names) if feature_names is not None else DEFAULT_FEATURES
    return pd.DataFrame([row]).reindex(columns=cols).astype(float)


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Dữ liệu thô (đúng 12 cột của health_data_official.csv, có thể bỏ claim) -> 35 feature.
    Dùng chung cho notebook huấn luyện và để đối chiếu với health_official_onehot.csv."""
    r = raw.copy()
    sex = r["sex"] if pd.api.types.is_numeric_dtype(r["sex"]) else r["sex"].map({"male": 1, "female": 0})
    age, bmi, bp = r["age"].astype(float), r["bmi"].astype(float), r["bloodpressure"].astype(float)
    sm, di, dep = r["smoker"].astype(float), r["diabetes"].astype(float), r["no_of_dependents"].astype(float)
    out = pd.DataFrame({"age": age, "sex": sex.astype(float), "weight": r["weight"].astype(float), "bmi": bmi,
                        "no_of_dependents": dep, "smoker": sm, "bloodpressure": bp, "diabetes": di,
                        "regular_ex": r["regular_ex"].astype(float)})
    out["Age_x_Smoker"] = age * sm; out["Age_x_Diabetic"] = age * di; out["BMI_x_Smoker"] = bmi * sm
    out["BP_per_Age"] = bp / age; out["Dependents_x_Smoker"] = dep * sm
    out["is_smoker_obese"] = ((sm == 1) & (bmi >= 30)).astype(float)
    out["is_smoker_senior"] = ((sm == 1) & (age >= 45)).astype(float)
    job = r["job_title"].where(r["job_title"].isin(KEPT_JOBS), "Other")
    for c in DEFAULT_FEATURES:
        if c.startswith("hereditary_diseases_"): out[c] = (r["hereditary_diseases"] == c.split("hereditary_diseases_")[1]).astype(float)
        elif c.startswith("job_title_"): out[c] = (job == c.split("job_title_")[1]).astype(float)
    return out[DEFAULT_FEATURES]


def pretty_value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "không rõ"
    if float(v) == int(v): return str(int(v))
    return f"{v:.2f}"


PRESETS = {
    "Trẻ khỏe": dict(age=25, sex=0, weight=58, bmi=21.5, no_of_dependents=0, smoker=0, bloodpressure=68,
                     diabetes=0, regular_ex=1, hereditary="NoDisease", job="Student"),
    "45 tuổi, hút thuốc, bệnh nền": dict(age=45, sex=1, weight=82, bmi=31.0, no_of_dependents=2, smoker=1,
                                          bloodpressure=85, diabetes=1, regular_ex=0, hereditary="HeartDisease", job="Engineer"),
    "Phí cao": dict(age=58, sex=1, weight=90, bmi=36.0, no_of_dependents=3, smoker=1, bloodpressure=95,
                    diabetes=1, regular_ex=0, hereditary="Cancer", job="HomeMakers"),
    "Thiếu thông tin (age, bmi)": dict(age=None, sex=0, weight=70, bmi=None, no_of_dependents=1, smoker=1,
                                        bloodpressure=75, diabetes=1, regular_ex=0, hereditary="NoDisease", job="Dancer"),
}


def _selftest():
    r = build_row(PRESETS["45 tuổi, hút thuốc, bệnh nền"])
    assert r.shape == (1, 35) and list(r.columns) == DEFAULT_FEATURES
    assert r.loc[0, "Age_x_Smoker"] == 45 and abs(r.loc[0, "BMI_x_Smoker"] - 31.0) < 1e-9
    assert r.loc[0, "is_smoker_obese"] == 1 and r.loc[0, "is_smoker_senior"] == 1
    assert abs(r.loc[0, "BP_per_Age"] - 85 / 45) < 1e-9
    assert r.loc[0, "hereditary_diseases_HeartDisease"] == 1 and r.filter(like="hereditary_").sum(axis=1).iloc[0] == 1
    r2 = build_row(PRESETS["Thiếu thông tin (age, bmi)"])
    for c in ["age", "bmi", "Age_x_Smoker", "Age_x_Diabetic", "BMI_x_Smoker", "BP_per_Age"]: assert np.isnan(r2.loc[0, c]), c
    assert r2.loc[0, "is_smoker_obese"] == 0 and r2.loc[0, "is_smoker_senior"] == 0   # cờ = 0 khi không rõ, như dữ liệu thật
    r3 = build_row(dict(PRESETS["Trẻ khỏe"], hereditary="Alzheimer (baseline)", job="Actor (baseline)"))
    assert r3.filter(like="hereditary_").sum(axis=1).iloc[0] == 0 and r3.filter(like="job_title_").sum(axis=1).iloc[0] == 0
    # engineer(thô) phải khớp build_row(từng hồ sơ)
    raw = pd.DataFrame([dict(age=45, sex="male", weight=82, bmi=31.0, hereditary_diseases="HeartDisease", no_of_dependents=2, smoker=1,
                             bloodpressure=85, diabetes=1, regular_ex=0, job_title="Engineer"),
                        dict(age=np.nan, sex="female", weight=70, bmi=np.nan, hereditary_diseases="Alzheimer", no_of_dependents=1, smoker=1,
                             bloodpressure=75, diabetes=1, regular_ex=0, job_title="Plumber")])
    e = engineer(raw)
    assert e.loc[0].equals(build_row(PRESETS["45 tuổi, hút thuốc, bệnh nền"]).iloc[0])
    assert e.loc[1, "job_title_Other"] == 1 and e.filter(like="hereditary_").loc[1].sum() == 0 and np.isnan(e.loc[1, "BP_per_Age"])
    print("features selftest OK")


if __name__ == "__main__":
    _selftest()
