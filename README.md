# Demo dự đoán chi phí bảo hiểm y tế (XGBoost + SHAP, MoE)

Giao diện chạy **local** (Streamlit), nhập thông số hồ sơ, nhận dự đoán và "hóa đơn" SHAP
(Mức nền + từng khoản = Tổng) cho cả hai mô hình.

```
demo/
├─ app.py                      giao diện Streamlit
├─ core/                       features (dựng 35 cột), adapters (XGB, MoE), explain (hóa đơn, thác nước)
├─ train/05_xgb_final.ipynb    notebook Kaggle: tune train/val, fit lại train, chấm test (CI bootstrap), SHAP, lưu mô hình
├─ artifacts/xgb/              model.json, meta.json, ảnh SHAP (giải nén từ xgb_artifacts.zip)
├─ artifacts/moe/              moe_loader.py của nhóm MoE (có moe_loader.example.py làm mẫu)
└─ selftest.py                 kiểm tra logic không cần xgboost/streamlit
```

## Bước 1. Huấn luyện XGBoost trên Kaggle (làm một lần)
1. Kaggle: File → Import Notebook → chọn `train/05_xgb_final.ipynb`, Add Input dataset `insurancedata` (có `health_insurance.csv`, `config.py`, `utils.py`).
2. Save Version → Save & Run All. Đợi tuning (vài phút đến vài chục phút tùy `N_ITER`).
3. Tab Output → tải `xgb_artifacts.zip`, giải nén vào `artifacts/xgb/` (phải có `model.json`, `meta.json`).

Quy trình đánh giá (holdout): cắt validation từ file train để tune, fit lại toàn bộ train, chấm một lần trên file test kèm khoảng tin cậy bootstrap.

## Bước 2. Chạy giao diện (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Trình duyệt tự mở http://localhost:8501.

## Bước 3. Gắn MoE (thành viên làm nhánh MoE)
Đặt vào `artifacts/moe/` file **`moe_loader.py`** có hàm `load(artifacts_dir)` trả về object có:

| Thành phần | Kiểu | Ý nghĩa |
|---|---|---|
| `expert_names` | list[str] | Tên K nhóm chuyên gia |
| `feature_names` | list[str] | 35 cột, đúng thứ tự của `core.features.DEFAULT_FEATURES` |
| `predict(X)` | (n,) | Dự đoán theo đơn vị gốc của claim, **bằng Σ_k g_k·f_k** |
| `gating_proba(X)` | (n, K) | Xác suất gating, mỗi dòng cộng bằng 1 |
| `expert_contribs(X)` | (K, n, F+1) | SHAP của từng expert; cột cuối là bias của expert |
| `expert_profiles` (tùy chọn) | dict | Mô tả nghiệp vụ từng nhóm |
| `meta` (tùy chọn) | dict | `test_metrics`, `best_params`... hiển thị ở tab "Về mô hình" |
| `feature_groups` (tùy chọn) | dict | Ánh xạ feature → nhóm hiển thị |

UI gộp SHAP của các expert theo công thức chính xác `contrib_j = Σ_k g_k·φ_kj`, `base = Σ_k g_k·bias_k`
(xem `core/adapters.combine_expert_contribs`), rồi **kiểm tra base + tổng contrib = dự đoán**. Nếu MoE của nhóm không thỏa
(VD gating có thêm bước làm mượt), app báo lỗi ngay tại đó để sửa.
Input `X` là DataFrame một hoặc nhiều dòng, NaN giữ nguyên (không điền trước).
Xem `artifacts/moe/moe_loader.example.py` làm mẫu chạy được.

## Thử giao diện khi chưa có MoE
- Windows cmd: `set DEMO_MOCK_MOE=1` rồi `streamlit run app.py` (MoE **giả lập**, có nhãn MOCK, không dùng để báo cáo).
- Hoặc sao chép `moe_loader.example.py` thành `moe_loader.py` để xem giao diện với MoE ví dụ (số liệu ngẫu nhiên).

## Lưu ý khi trình bày
- Khi không rõ tuổi/BMI/huyết áp, mô hình xử lý giá trị thiếu nguyên bản; các cột tương tác tự tính theo cùng quy tắc của dữ liệu (NaN lan truyền, cờ `is_smoker_*` = 0).
- Các cột hút thuốc đa cộng tuyến: bật "Gộp theo nhóm" khi diễn giải.
- Bộ dữ liệu có hút thuốc là yếu tố chi phối; đây là demo phương pháp, không phải công cụ định phí thật.

## Hợp đồng tích hợp & nghiệm thu

Chi tiết tại `MODEL_CONTRACT.md`. Trước khi giao gói mô hình:

```
python validate_package.py artifacts/xgb
python validate_package.py artifacts/moe
```
