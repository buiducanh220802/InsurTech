"""Demo UI: nhập thông số -> dự đoán chi phí bảo hiểm y tế + giải thích SHAP, cho XGBoost và MoE.
Chạy:  streamlit run app.py        (không cần deploy, chạy hoàn toàn local)"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core.features import HEREDITARY, JOBS, PRESETS, build_row, DEFAULT_FEATURES, pretty_value
from core.adapters import load_models
from core.explain import make_bill, waterfall

UNIT = "USD"     # đơn vị của claim trong dữ liệu
ART = os.environ.get("DEMO_ARTIFACTS", os.path.join(HERE, "artifacts"))

st.set_page_config(page_title="Dự đoán chi phí bảo hiểm y tế", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def get_models():
    return load_models(ART)


models, notes = get_models()

# ------------------------------------------------------------------ giá trị mặc định / khoảng nhập
RANGES = dict(age=(18, 64, 38), weight=(34, 95, 65), bmi=(16.0, 53.1, 29.0), no_of_dependents=(0, 5, 1), bloodpressure=(40, 122, 73))
for m in models.values():
    r = (m.meta or {}).get("ranges")
    if r:
        for k in RANGES:
            if k in r: RANGES[k] = (r[k]["min"], r[k]["max"], r[k]["median"])
        break


def _apply(p: dict):
    s = st.session_state
    s["age_unknown"] = p["age"] is None;  s["age"] = int(p["age"] if p["age"] is not None else RANGES["age"][2])
    s["bmi_unknown"] = p["bmi"] is None;  s["bmi"] = float(p["bmi"] if p["bmi"] is not None else RANGES["bmi"][2])
    s["bp_unknown"] = p["bloodpressure"] is None; s["bp"] = int(p["bloodpressure"] if p["bloodpressure"] is not None else RANGES["bloodpressure"][2])
    s["sex"] = "Nam" if p["sex"] == 1 else "Nữ"; s["weight"] = int(p["weight"]); s["dep"] = int(p["no_of_dependents"])
    s["smoker"] = bool(p["smoker"]); s["diabetes"] = bool(p["diabetes"]); s["ex"] = bool(p["regular_ex"])
    s["hered"] = p["hereditary"]; s["job"] = p["job"]


def preset_cb(name):
    _apply(PRESETS[name])


if "age" not in st.session_state:
    _apply(PRESETS["Trẻ khỏe"])

# ------------------------------------------------------------------ sidebar: nhập thông số
sb = st.sidebar
sb.header("Thông tin hồ sơ")
sb.caption("Chọn nhanh một hồ sơ mẫu:")
for name in PRESETS:
    sb.button(name, key=f"p_{name}", on_click=preset_cb, args=(name,))
sb.divider()

ss = st.session_state
sb.checkbox("Không rõ tuổi", key="age_unknown")
sb.number_input("Tuổi", int(RANGES["age"][0]), int(RANGES["age"][1]), key="age", disabled=ss["age_unknown"])
sb.radio("Giới tính", ["Nữ", "Nam"], key="sex", horizontal=True)
sb.slider("Cân nặng (kg)", int(RANGES["weight"][0]), int(RANGES["weight"][1]), key="weight")
sb.checkbox("Không rõ BMI", key="bmi_unknown")
sb.number_input("BMI", float(RANGES["bmi"][0]), float(RANGES["bmi"][1]), step=0.1, key="bmi", disabled=ss["bmi_unknown"])
sb.slider("Số người phụ thuộc", int(RANGES["no_of_dependents"][0]), int(RANGES["no_of_dependents"][1]), key="dep")
sb.checkbox("Có hút thuốc", key="smoker")
sb.checkbox("Không rõ huyết áp", key="bp_unknown")
sb.number_input("Huyết áp tâm trương (mmHg)", int(RANGES["bloodpressure"][0]), int(RANGES["bloodpressure"][1]), key="bp", disabled=ss["bp_unknown"])
sb.checkbox("Có tiểu đường", key="diabetes")
sb.checkbox("Tập thể dục thường xuyên", key="ex")
sb.selectbox("Bệnh di truyền trong gia đình", HEREDITARY, key="hered")
sb.selectbox("Nghề nghiệp", JOBS, key="job")

inp = dict(age=None if ss["age_unknown"] else ss["age"], sex=1 if ss["sex"] == "Nam" else 0, weight=ss["weight"],
           bmi=None if ss["bmi_unknown"] else ss["bmi"], no_of_dependents=ss["dep"], smoker=int(ss["smoker"]),
           bloodpressure=None if ss["bp_unknown"] else ss["bp"], diabetes=int(ss["diabetes"]), regular_ex=int(ss["ex"]),
           hereditary=ss["hered"], job=ss["job"])
X1 = build_row(inp, DEFAULT_FEATURES)

# ------------------------------------------------------------------ nội dung chính
st.title("Dự đoán chi phí bảo hiểm y tế")
st.caption("Bản demo nội bộ. Mô hình huấn luyện trên bộ dữ liệu bảo hiểm để minh họa phương pháp, không dùng để định phí thực tế.")
for n in notes:
    (st.warning if "MOCK" in n or "GIẢ LẬP" in n else st.info)(n)
if not models:
    st.error("Chưa có mô hình nào được nạp. Xem README.md, mục 'Bước 1'.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Dự đoán và giải thích", "Giải thích toàn cục", "Về mô hình"])

with tab1:
    c1, c2, c3 = st.columns([2, 1, 1])
    chosen = c1.multiselect("Mô hình hiển thị", list(models), default=list(models))
    grouped = c2.toggle("Gộp theo nhóm", value=True, help="Gộp các cột liên quan (VD các cột hút thuốc) thành một khoản để dễ đọc")
    top_n = c3.slider("Số khoản hiển thị", 3, 15, 8)
    if not chosen:
        st.info("Chọn ít nhất một mô hình.")
    else:
        cols = st.columns(len(chosen))
        preds = {}
        for col, name in zip(cols, chosen):
            m = models[name]
            with col:
                st.subheader(name)
                try:
                    e = m.explain(X1)
                except Exception as ex:
                    st.error(f"Lỗi khi dự đoán/giải thích: {ex}"); continue
                preds[name] = e.prediction
                st.metric("Chi phí dự đoán", f"{e.prediction:,.0f} {UNIT}",
                          delta=f"{e.prediction - e.base_value:+,.0f} so với mức nền", delta_color="off")
                if "gating" in e.extras:
                    g = e.extras["gating"].sort_values(ascending=False)
                    st.markdown(f"**Nhóm chuyên gia chính:** {g.index[0]} ({g.iloc[0]:.0%})")
                    prof = e.extras.get("expert_profiles", {}).get(g.index[0])
                    if prof: st.caption(prof)
                    st.bar_chart(g)
                bill = make_bill(e, X1, m.groups, grouped=grouped, top_n=top_n)
                fig = waterfall(bill, UNIT); st.pyplot(fig); plt.close(fig)
                disp = bill.copy()
                disp[f"Ảnh hưởng ({UNIT})"] = [f"{v:,.0f}" if i in (0, len(bill) - 1) else f"{v:+,.0f}" for i, v in enumerate(bill["Ảnh hưởng"])]
                st.dataframe(disp.drop(columns="Ảnh hưởng"), hide_index=True)
                st.caption("Mức nền + các khoản = tổng dự đoán (đã kiểm tra cộng đủ). Đỏ: làm tăng chi phí, xanh: làm giảm chi phí.")
        if len(preds) == 2:
            a, b = list(preds)
            st.info(f"Chênh lệch giữa hai mô hình: {preds[a] - preds[b]:+,.0f} {UNIT}")
    with st.expander("Giá trị 35 đặc trưng đưa vào mô hình (gồm các cột tương tác tự tính)"):
        t = X1.T.rename(columns={0: "Giá trị"}); t["Giá trị"] = [pretty_value(v) for v in t["Giá trị"]]
        st.dataframe(t)

with tab2:
    shown = False
    for name in models:
        m = models[name]
        files = m.global_files() if hasattr(m, "global_files") else {}
        tbl = m.global_table() if hasattr(m, "global_table") else None
        if not files and tbl is None: continue
        shown = True
        st.subheader(name)
        a, b = st.columns(2)
        if "summary" in files: a.image(files["summary"], caption="SHAP beeswarm (mỗi chấm là một hồ sơ)")
        if "bar" in files: b.image(files["bar"], caption="Mức quan trọng trung bình")
        if "groups" in files: st.image(files["groups"], caption="Ảnh hưởng theo nhóm đặc trưng")
        if tbl is not None: st.dataframe(tbl.head(20), hide_index=True)
    if not shown:
        st.info("Chưa có ảnh SHAP toàn cục. Chạy notebook train và giải nén đầy đủ xgb_artifacts.zip.")
    st.caption("Các cột hút thuốc (smoker, Age_x_Smoker, BMI_x_Smoker, is_smoker_*) rất tương quan nhau nên điểm SHAP bị chia giữa chúng; hãy đọc theo nhóm.")

with tab3:
    for name, m in models.items():
        st.subheader(name)
        meta = m.meta or {}
        tm = meta.get("test_metrics")
        if tm:
            rows = {k: {"Giá trị": v["value"], "CI 95% thấp": v["ci_low"], "CI 95% cao": v["ci_high"]} for k, v in tm.items()}
            st.write("Kết quả trên tập test (fit trên toàn bộ train, CI bootstrap 95%):")
            st.dataframe(pd.DataFrame(rows).T.round(4))
            fx = meta.get("test_metrics_xgb_fixed")
            if fx: st.caption("Mốc chưa tune (XGB_FIXED), MAE test: %s" % format(fx["mae"]["value"], ",.1f"))
        elif meta.get("cv_metrics"):
            df = pd.DataFrame(meta["cv_metrics"]).T
            st.write("Kết quả đánh giá (mô hình cung cấp):"); st.dataframe(df.round(4))
        if meta.get("best_params"): st.write("Tham số tốt nhất:"); st.json(meta["best_params"])
        extra = {k: meta[k] for k in ("model", "xgboost_version", "n_rows", "protocol", "claim") if k in meta}
        if extra: st.json(extra)
        if not meta: st.caption("Mô hình này chưa cung cấp meta.")
    st.markdown("**Lưu ý:** dữ liệu có nhiều cột thưa (bệnh di truyền hiếm, nghề nghiệp) và các cột hút thuốc đa cộng tuyến; "
                "đọc giải thích theo nhóm. Khi không rõ tuổi/BMI, mô hình xử lý giá trị thiếu nguyên bản (không điền trước).")
