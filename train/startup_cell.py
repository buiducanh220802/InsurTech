# CELL KHỞI ĐỘNG (chạy đầu mỗi notebook)
import os, shutil, sys, importlib
IN = None
for root, dirs, files in os.walk("/kaggle/input"):
    if {"config.py", "utils.py"}.issubset(files) and any(f.endswith(".csv") for f in files):
        IN = root; break
assert IN is not None, "Không thấy config.py/utils.py/CSV: kiểm tra Add Input dataset"
print("Dataset nằm ở:", IN, "->", sorted(os.listdir(IN)))

WORK = "/kaggle/working"
for f in ["config.py", "utils.py"]:
    shutil.copy(os.path.join(IN, f), os.path.join(WORK, f))
sys.path.insert(0, WORK)
import config; importlib.reload(config)          # reload để không dính bản cũ trong bộ nhớ
assert config.TARGET == "claim", f"config.py đang là bản cũ (TARGET={config.TARGET}): upload lại config.py mới"
assert os.path.exists(os.path.join(IN, config.INPUT_CSV)), f"Thiếu {config.INPUT_CSV} trong dataset"
shutil.copy(os.path.join(IN, config.INPUT_CSV), os.path.join(WORK, config.INPUT_CSV))
if os.path.exists(os.path.join(IN, "preproc.py")):
    shutil.copy(os.path.join(IN, "preproc.py"), os.path.join(WORK, "preproc.py"))
os.makedirs(f"{WORK}/figs", exist_ok=True); os.makedirs(f"{WORK}/tables", exist_ok=True)
import utils; importlib.reload(utils)
X, y = utils.load_data(WORK)
cv = utils.make_cv(len(y))
print("SẴN SÀNG:", X.shape, len(cv), sorted({c["seed"] for c in cv}))
assert X.shape[1] == 35, f"Kỳ vọng 35 feature, thực tế {X.shape[1]} (còn cột index hoặc claim?)"
