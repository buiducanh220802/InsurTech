"""MoE GIẢ LẬP, chỉ để thử giao diện khi nhóm chưa giao mô hình thật. Bật bằng DEMO_MOCK_MOE=1.
3 'expert' đều là cùng một mô hình XGBoost; gating chỉ là luật tay. Không dùng để báo cáo kết quả."""
import numpy as np
from .adapters import contrib_matrix


class MockMoE:
    expert_names = ["Nhóm trẻ khỏe", "Nhóm rủi ro chuyển hóa", "Nhóm hút thuốc rủi ro cao"]
    expert_profiles = {"Nhóm trẻ khỏe": "(MOCK) tuổi thấp, không hút thuốc",
                       "Nhóm rủi ro chuyển hóa": "(MOCK) bmi cao hoặc tiểu đường, không hút thuốc",
                       "Nhóm hút thuốc rủi ro cao": "(MOCK) hút thuốc"}

    def __init__(self, xgb_adapter):
        self.xa = xgb_adapter
        self.feature_names = xgb_adapter.feature_names
        self.meta = {"note": "MOCK"}

    def gating_proba(self, X):
        s = (X["smoker"] == 1).to_numpy(float)
        m = ((X["diabetes"] == 1) | (X["bmi"].fillna(0) >= 30)).to_numpy(float) * (1 - s)
        y = 1 - s - m
        z = np.stack([0.2 + 2 * y, 0.2 + 2 * m, 0.2 + 2 * s], axis=1)
        return z / z.sum(axis=1, keepdims=True)

    def expert_contribs(self, X):
        c = contrib_matrix(self.xa.booster, X)
        return np.stack([c, c, c], axis=0)

    def predict(self, X):
        g = self.gating_proba(X)
        p = self.xa.predict(X)
        return (g * p[:, None]).sum(axis=1)
