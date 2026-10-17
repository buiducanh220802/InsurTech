# HỢP ĐỒNG TÍCH HỢP MÔ HÌNH ↔ UI DEMO

Hai người, hai mô hình, một UI. Mỗi người tự huấn luyện, tự lưu, tự nghiệm thu bằng
`validate_package.py`, rồi bỏ vào thư mục của mình. UI không sửa gì khi ghép.

| Nhánh | Người làm | Thư mục giao | Nghiệm thu |
|---|---|---|---|
| XGBoost + SHAP | (bạn) | `artifacts/xgb/` | `python validate_package.py artifacts/xgb` |
| MoE (expert là cây) | (thành viên kia) | `artifacts/moe/` | `python validate_package.py artifacts/moe` |

Điều kiện giao: validator in `KẾT QUẢ: ĐẠT (0 FAIL ...)`. WARN không chặn nhưng nên xử lý.

---------------------------------------------------------------------------

## 1. Quy tắc chung (cả hai nhánh)

> Dữ liệu train/test là dữ liệu SAU engineering (35 cột). UI nhận thông tin thô từ người dùng và tự engineering
> bằng `core/features.py::build_row` trước khi gọi mô hình; mô hình luôn nhận 35 cột. Quy tắc engineering theo
> `description_health_data.md` (sex male=1; baseline hereditary = Alzheimer; baseline job = Actor; 25 nghề ít gặp gom thành Other).
> `core/features.py::engineer(raw_df)` dựng 35 cột từ dữ liệu thô 12 cột, dùng để đối chiếu với `health_official_onehot.csv`.


1. **Dữ liệu vào**: DataFrame pandas đúng **35 cột** như `DEFAULT_FEATURES` trong `core/features.py`
   (cùng tên, kiểu số thực, không cần đúng thứ tự nhưng nên đúng). Giá trị không rõ là `NaN` gốc
   (age, bmi, bloodpressure và các cột dẫn xuất từ chúng). **Mô hình phải tự xử lý NaN**
   (cây XGBoost/LightGBM xử lý native; nếu dùng sklearn thì imputer phải nằm TRONG gói và được lưu).
   UI không impute hộ.
2. **Dữ liệu ra**: dự đoán theo **đơn vị gốc của `claim`** (USD, khoảng 1.100 đến 63.800).
   Cấm xuất ra log1p(claim). Nếu huấn luyện trên log thì phải đổi lại trong `predict` VÀ trong SHAP
   sẽ không còn cộng tính, nên **không dùng log-target** cho bản demo. (Xem mục 7, câu hỏi mở.)
3. **Không clip, không làm mượt, không làm tròn** trong `predict`. Giá trị UI hiển thị phải bằng
   Base + tổng đóng góp SHAP, sai lệch tối đa 0,05.
4. **Giao thức huấn luyện thống nhất (holdout)**: cả hai dùng CÙNG file train 80% và file test 20%.
   - Cắt validation từ file train: `train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)`.
   - Tune tham số (và chọn số cây/early stopping, số epoch...) trên train-con / validation.
   - Chốt cấu hình, **fit lại toàn bộ file train** (cả phần validation) rồi chấm **đúng một lần** trên test.
     Không chỉnh gì thêm sau khi đã xem điểm test.
   - Mọi bước học từ dữ liệu (scaler, encoder, imputer, gating...) chỉ fit trên train.
   - Báo cáo test bằng `core/metrics.py::bootstrap_ci(y_test, y_pred)` (MAE, RMSE, R2, WAPE, CI bootstrap 95%),
     ghi vào `meta["test_metrics"]` (định dạng ở mục 2). Cùng một hàm tính cho cả hai nhánh.
5. **Lưu bản tốt nhất**: lưu mô hình đã fit lại trên toàn bộ file train với cấu hình đã chốt (chính mô hình đã chấm trên test). Không lưu mô hình của
   vòng thử.
6. **Pin phiên bản**: ghi version thư viện vào meta (`xgboost_version`, `lightgbm_version`...).
   Máy chạy demo cài đúng version đó (ghi vào `requirements.txt` nếu thêm thư viện).
7. **Không phụ thuộc đường dẫn tuyệt đối hay thư mục làm việc**: mọi file đọc phải tính từ
   `artifacts_dir` được truyền vào. Không đọc `/kaggle/...`, không đọc CSV gốc khi chạy demo.
8. **Không sửa DataFrame đầu vào** (làm trên bản sao).
9. **Tốc độ**: nạp < 30 s; `explain` một hồ sơ < 1,5 s.

---------------------------------------------------------------------------

## 2. Gói XGBoost (`artifacts/xgb/`)

Notebook `train/05_xgb_final.ipynb` đã tạo sẵn đúng gói này trên Kaggle (tải `xgb_artifacts.zip`
về, giải nén vào `artifacts/xgb/`).

| File | Bắt buộc | Nội dung |
|---|---|---|
| `model.json` | có | `booster.save_model("model.json")`; objective **`reg:squarederror`** trên claim gốc |
| `meta.json` | có | `feature_names` (35), `feature_groups`, `best_params`, `protocol`, `test_metrics`, `test_metrics_xgb_fixed`, `ranges`, thống kê claim, `expected_value`, `xgboost_version` |
| `shap_bar.png`, `shap_groups.png`, `shap_summary.png`, `shap_importance.csv` | nên có | SHAP toàn cục cho tab "Giải thích toàn cục" |

Định dạng `meta["test_metrics"]` (cả hai nhánh): 
`{"mae": {"value": ..., "ci_low": ..., "ci_high": ...}, "rmse": {...}, "r2": {...}, "wape": {...}}`
(sinh bằng `core/metrics.bootstrap_ci`).

SHAP cho từng hồ sơ do **UI tự tính** bằng `booster.predict(DMatrix, pred_contribs=True)`
(TreeSHAP chính xác, cột cuối là bias). Người làm nhánh này không phải viết code SHAP.

Cấm: objective log-link (`reg:tweedie`, `reg:gamma`, `count:poisson`), `reg:squaredlogerror`,
hậu xử lý predict.

---------------------------------------------------------------------------

## 3. Gói MoE (`artifacts/moe/`), expert là cây

### 3.1 Cấu trúc thư mục

```
artifacts/moe/
  moe_loader.py        # BẮT BUỘC: có hàm load(artifacts_dir) trả về đối tượng MoE
  ...                  # mọi file của bạn: gating.pkl, expert_0.json, expert_1.json...
  shap_*.png, shap_importance.csv   # nên có
```
`moe_loader.py` được import với thư mục gói nằm trong `sys.path`, nên import được file anh em.
Mẫu chạy được: `artifacts/moe/moe_loader.example.py` (ToyMoE), copy rồi thay ruột.

### 3.2 Giao diện đối tượng trả về từ `load(artifacts_dir)`

| Thành viên | Kiểu | Ý nghĩa |
|---|---|---|
| `expert_names` | `list[str]`, K ≥ 2, không trùng | Tên hiển thị từng expert ("Nhóm hút thuốc", ...) |
| `feature_names` | `list[str]`, đúng 35 tên | Thứ tự cột của trục F trong `expert_contribs` |
| `predict(X)` | → `ndarray (n,)` | Dự đoán claim, đơn vị gốc |
| `gating_proba(X)` | → `ndarray (n, K)` | Trọng số g_k, không âm, mỗi dòng cộng = 1 |
| `expert_contribs(X)` | → `ndarray (K, n, F+1)` | Với từng expert: SHAP từng feature (F cột) + **bias ở cột cuối** |
| `expert_profiles` | `dict[str,str]` (tùy chọn) | Mô tả nghiệp vụ mỗi expert, tab "Về mô hình" dùng |
| `meta` | `dict` (tùy chọn) | `test_metrics`, `best_params`, `protocol`... |
| `feature_groups` | `dict` (tùy chọn) | Gộp feature để vẽ waterfall gọn |
| `artifacts_dir` | `str` (tùy chọn) | Nơi UI tìm ảnh SHAP toàn cục |

`X` là DataFrame n hàng. `n` có thể là 1 (UI) hoặc vài trăm (validator).

### 3.3 Điều kiện toán học (cốt lõi)

Gọi f_k(x) = bias_k + Σ_j φ_kj(x) là dự đoán của expert k. Hợp đồng đòi hỏi

```
predict(x) = Σ_k g_k(x) · f_k(x)         (sai lệch ≤ 0,05)
```
với **cùng g_k** mà `gating_proba` trả về. Khi đó UI gộp chính xác:

```
Base   = Σ_k g_k · bias_k
contrib_j = Σ_k g_k · φ_kj
Base + Σ_j contrib_j = predict(x)
```
Ví dụ số: g = (0,7; 0,3); expert 1: bias 10.000, φ_tuổi = +2.000, φ_smoker = +1.000 ⇒ f_1 = 13.000;
expert 2: bias 8.000, φ_tuổi = 0, φ_smoker = +500 ⇒ f_2 = 8.500. predict = 0,7·13.000 + 0,3·8.500 = 11.650.
Base = 0,7·10.000 + 0,3·8.000 = 9.400; contrib_tuổi = 1.400; contrib_smoker = 0,7·1.000+0,3·500 = 850.
9.400 + 1.400 + 850 = 11.650 ✔.

Hệ quả:
- **Soft routing**: gating_proba là xác suất mềm, predict dùng đúng nó.
- **Hard routing** (chọn 1 expert): `gating_proba` trả one-hot, predict dùng expert được chọn.
  Đừng trả xác suất mềm trong khi predict lại hard, validator sẽ báo lệch.
- Expert phải **cộng tính trong claim gốc**: không log-link, không clip, không nhân hệ số sau.
- Nếu có bước hậu xử lý trên tổng (ví dụ làm tròn, giới hạn tối thiểu), phải bỏ.

### 3.4 Lấy φ_kj từ expert là cây (copy dùng)

```python
# XGBoost expert (reg:squarederror, claim gốc)
import xgboost as xgb
d = xgb.DMatrix(X[feature_names], feature_names=feature_names)
phi = booster.predict(d, pred_contribs=True)          # (n, F+1), cột cuối = bias

# LightGBM expert
phi = lgbm_model.predict(X[feature_names], pred_contrib=True)   # (n, F+1), cột cuối = bias

# sklearn tree/forest/GBM (cần shap, tắt kiểm tra additivity nếu cần)
import shap
ex = shap.TreeExplainer(model)
phi = np.c_[ex.shap_values(X[feature_names]), np.full(len(X), ex.expected_value)]
```
Rồi `expert_contribs = np.stack([phi_0, phi_1, ..., phi_{K-1}])` có shape (K, n, F+1).

Lưu ý: nếu expert chỉ nhìn một **tập con feature**, vẫn phải trả đủ F cột, cột không dùng bằng 0.
Nếu expert train trên dòng con (cụm), vẫn chạy được trên mọi hồ sơ; g_k mới quyết định trọng số.

### 3.5 Khung `moe_loader.py` tối thiểu

```python
import os, json, numpy as np
import xgboost as xgb

class MoE:
    def __init__(self, d):
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        self.feature_names = meta["feature_names"]
        self.expert_names = meta["expert_names"]
        self.experts = []
        for i in range(len(self.expert_names)):
            b = xgb.Booster(); b.load_model(os.path.join(d, f"expert_{i}.json")); self.experts.append(b)
        self.gate = ...            # nạp bộ gating của bạn (cây, sklearn, tự viết)
        self.meta = {"test_metrics": meta.get("test_metrics"), "best_params": meta.get("best_params")}
        self.expert_profiles = meta.get("expert_profiles", {})
        self.artifacts_dir = d

    def _X(self, X): return X[self.feature_names]
    def gating_proba(self, X): ...                      # (n, K), cộng = 1
    def expert_contribs(self, X):
        D = xgb.DMatrix(self._X(X), feature_names=self.feature_names)
        return np.stack([b.predict(D, pred_contribs=True) for b in self.experts])
    def predict(self, X):
        c = self.expert_contribs(X)                     # (K, n, F+1)
        return (self.gating_proba(X) * c.sum(axis=2).T).sum(axis=1)   # đảm bảo đúng hợp đồng

def load(artifacts_dir):
    return MoE(artifacts_dir)
```
Cách an toàn nhất để thỏa 3.3: **viết `predict` đúng như dòng cuối ở trên**, kể cả khi lúc huấn
luyện bạn dùng đường khác. Khi đó đẳng thức luôn đúng, và việc còn lại là điểm số CV phải tính
bằng chính hàm này.

### 3.6 Điểm số phải khớp mô hình được demo
Chấm test bằng **đúng** `predict` ở trên (mô hình đã fit lại trên toàn bộ train), ghi vào `meta["test_metrics"]`.
Đừng báo điểm của một biến thể khác với cái được lưu.

---------------------------------------------------------------------------

## 4. Việc UI làm với gói của bạn (để biết giới hạn)

- XGBoost: UI gọi `pred_contribs` rồi dựng hóa đơn: Base + từng feature (hoặc nhóm) + "Các yếu tố còn lại" + TỔNG; có kiểm tra additivity (assert).
- MoE: UI gọi `predict`, `gating_proba`, `expert_contribs`, gộp theo 3.3, vẽ thêm biểu đồ trọng số gating.
- UI không đọc dữ liệu huấn luyện, không tự impute, không tự đổi đơn vị.

---------------------------------------------------------------------------

## 5. Nghiệm thu bằng `validate_package.py`

```
python validate_package.py artifacts/moe
python validate_package.py artifacts/xgb
```
Mã thoát 0 = đạt. Các kiểm tra chính:

| Kiểm tra | Mức |
|---|---|
| Nạp được, < 30 s | FAIL |
| Đủ thuộc tính/hàm; `expert_names` K≥2 không trùng | FAIL |
| `feature_names` đúng 35 tên | FAIL |
| Chạy được trên ~300 hồ sơ có 15% NaN; không sửa DataFrame vào | FAIL |
| Shape: predict (n,), gating (n,K), contribs (K,n,F+1), hữu hạn | FAIL |
| Gating không âm, mỗi dòng cộng 1 | FAIL |
| `predict = Σ g_k f_k` (≤ 0,05) | FAIL |
| Chạy 1 dòng = chạy theo lô | FAIL |
| Đơn vị đầu ra (phát hiện log1p) | FAIL / WARN |
| `MoEAdapter.explain` trên 4 hồ sơ mẫu, tốc độ | FAIL / WARN |
| `meta['test_metrics']`, `expert_profiles`, ảnh SHAP toàn cục | WARN |

Đã thử: gói mẫu → ĐẠT; predict bị clip → FAIL ở đẳng thức; đầu ra log-scale → FAIL đơn vị;
gating sai shape → FAIL.

---------------------------------------------------------------------------

## 6. Lỗi thường gặp

| Triệu chứng | Nguyên nhân hay gặp | Cách sửa |
|---|---|---|
| FAIL đẳng thức, lệch hàng trăm | predict có clip/max/min hoặc hard vs soft không khớp | Dùng `predict` như 3.5 |
| FAIL đơn vị, trung bình ≈ 10 | huấn luyện trên log1p | Train lại trên claim gốc (`reg:squarederror`) |
| `KeyError`/thiếu cột | `feature_names` khác dữ liệu | Dùng `core.features.DEFAULT_FEATURES` |
| Lỗi trên hồ sơ NaN | sklearn không nhận NaN | Đưa imputer vào trong gói |
| Import lỗi khi UI nạp | loader dùng đường dẫn tương đối/cwd | Dùng `os.path.join(artifacts_dir, ...)` |
| Lệch phiên bản | model lưu bằng xgboost khác bản | Ghi và cài đúng version |
| Contribs thiếu cột bias | quên cột cuối | `pred_contribs`/`pred_contrib` đã có sẵn; shap thì tự nối |

---------------------------------------------------------------------------

## 7. Câu hỏi mở (thống nhất với nhóm)

1. **Log-target**: nếu MoE cần log để tốt hơn thì SHAP không còn cộng tính trong USD. Phương án:
   báo cáo SHAP trong không gian log và UI đổi nhãn đơn vị. Chưa hỗ trợ, cần quyết định trước khi tune.
2. **Giải thích gating**: hiện UI chỉ hiển thị trọng số g_k. Nếu gating cũng là cây, có thể
   thêm SHAP cho gating ("vì sao chọn expert này"), cần mở rộng giao diện.
3. **Mô hình hộp đen (mức C)**: nếu một expert không phải cây, UI cần SHAP độc lập mô hình
   (chậm hơn). Chưa cài; chỉ làm nếu thực sự cần.
4. **Mô phỏng MNAR** cho bài stress-test: chưa quyết định.

---------------------------------------------------------------------------

## 8. Checklist giao hàng

- [ ] Đúng giao thức holdout: cùng train/test, cùng cách cắt validation (seed 42), chấm test một lần
- [ ] Mục tiêu là claim gốc, không log, không clip
- [ ] Gói nằm đúng thư mục, không phụ thuộc đường dẫn ngoài
- [ ] `meta` có điểm CV và version thư viện
- [ ] Có ảnh SHAP toàn cục
- [ ] `python validate_package.py <thư mục>` → ĐẠT
- [ ] Chạy thử `streamlit run app.py` thấy mô hình xuất hiện và hóa đơn khớp TỔNG
