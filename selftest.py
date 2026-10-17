"""Kiểm tra phần logic không cần xgboost/streamlit:  python selftest.py"""
import os, sys, importlib.util
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import features, explain
from core.adapters import MoEAdapter
from core.features import PRESETS, build_row, default_groups

features._selftest(); explain._selftest()

# Hợp đồng MoE: nạp ví dụ, explain phải cộng đủ cho mọi hồ sơ mẫu, có gating
spec = importlib.util.spec_from_file_location("ex", os.path.join("artifacts", "moe", "moe_loader.example.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
a = MoEAdapter(mod.load("."))
for name, p in PRESETS.items():
    X1 = build_row(p); e = a.explain(X1)
    assert abs(e.prediction - a.predict(X1)[0]) < 1e-6 and abs(e.extras["gating"].sum() - 1) < 1e-9
    bill = explain.make_bill(e, X1, a.groups, grouped=True, top_n=6)
    fig = explain.waterfall(bill); fig.savefig(os.devnull, format="png"); 
print("MoE adapter + bill + waterfall OK trên", len(PRESETS), "hồ sơ mẫu")
