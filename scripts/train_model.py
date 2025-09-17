# -*- coding: utf-8 -*-
"""
train_model.py — Chọn mô hình tốt nhất & lưu cho dashboard
- Target: log(GiaVND)  → khi đánh giá/predict sẽ expm1 về VND
- Features: DienTich + one-hot Phuong_* (khớp dashboard)
- Ứng viên: HistGradientBoosting, XGBoost*, LightGBM*, CatBoost*, RandomForest, GradientBoosting
  (*): nếu thư viện không có, sẽ tự bỏ qua.
- Lưu:
  models/best_model.pkl
  models/feature_columns.json
  models/metrics.json  (có khóa "target": "log_price_vnd")

Chạy:
  source /mnt/d/PhanTichDuLieu/.venv/bin/activate
  python /mnt/d/PhanTichDuLieu/scripts/train_model.py
"""
import os, json, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Path helpers ----------
def pick_project_root():
    cand = ["/mnt/d/PhanTichDuLieu", "/mnt/d/PHANTICHDULIEU", os.getcwd()]
    for p in cand:
        if os.path.exists(p):
            return Path(p)
    return Path.cwd()

ROOT = pick_project_root()
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
CSV_INPUT  = DATA_DIR / "cleaned_rental_data.csv"
MODEL_OUT  = MODEL_DIR / "best_model.pkl"              # giữ tên này để dashboard không phải đổi
FEATS_JSON = MODEL_DIR / "feature_columns.json"
METRICS_JS = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Utils ----------
def human_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    v = float(v)
    if abs(v) >= 1_000_000_000: return f"{v/1_000_000_000:.2f} tỷ"
    if abs(v) >= 1_000_000:     return f"{v/1_000_000:.1f} triệu"
    if abs(v) >= 1_000:         return f"{v/1_000:.0f} nghìn"
    return f"{v:.0f} đ"

def safe_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def winsorize_series(s, p_low=1, p_high=99):
    if len(s) == 0: return s
    lo, hi = np.nanpercentile(s, [p_low, p_high])
    return s.clip(lo, hi)

# ---------- Data & features ----------
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"⚠️ Không thấy file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)

    # chuẩn hoá tên cột
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "tieude":   rename[c] = "TieuDe"
        if lc == "giavnd":   rename[c] = "GiaVND"
        if lc == "dientich": rename[c] = "DienTich"
        if lc == "phuong":   rename[c] = "Phuong"
        if lc == "quan":     rename[c] = "Quan"
        if lc == "link":     rename[c] = "Link"
    if rename: df = df.rename(columns=rename)

    keep = [c for c in ["TieuDe","GiaVND","DienTich","Phuong","Quan","Link"] if c in df.columns]
    if not keep: return pd.DataFrame()
    df = df[keep].copy()

    # ép kiểu
    df["GiaVND"]   = pd.to_numeric(df.get("GiaVND"), errors="coerce")
    df["DienTich"] = pd.to_numeric(df.get("DienTich"), errors="coerce")

    # lọc cơ bản
    df = df.dropna(subset=["GiaVND","DienTich"])
    df = df[(df["GiaVND"] > 0) & (df["DienTich"] > 0)]

    # chuẩn hoá text
    for c in ["Phuong","Quan"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("unknown").str.strip().str.lower()
        else:
            df[c] = "unknown"

    # khử ngoại lai giá (an toàn)
    df["GiaVND"] = winsorize_series(df["GiaVND"], 1, 99)
    return df.reset_index(drop=True)

def build_features(df: pd.DataFrame):
    """
    Giữ đúng schema đơn giản mà dashboard hỗ trợ:
      - DienTich (numeric)
      - one-hot Phuong_*
    """
    X = pd.DataFrame({"DienTich": pd.to_numeric(df["DienTich"], errors="coerce")})
    phuong = df["Phuong"].astype(str).fillna("unknown")
    dummies = pd.get_dummies(phuong, prefix="Phuong", dtype=float)
    X = pd.concat([X, dummies], axis=1)
    return X, X.columns.tolist()

# ---------- CV scoring ----------
def cv_score_rmse_vnd_log_target(X: pd.DataFrame, y_log: np.ndarray, model_builder, n_splits: int) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmses = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]
        mdl = model_builder()
        mdl.fit(X_tr, y_tr)
        yhat_vnd = np.expm1(mdl.predict(X_va))
        ytrue_vnd = np.expm1(y_va)
        rmses.append(safe_rmse(ytrue_vnd, yhat_vnd))
    return float(np.mean(rmses))

# ---------- Model zoo (builders) ----------
def get_model_builders():
    builders = []

    # HistGradientBoosting — mạnh & nhanh
    hgb_grid = [
        dict(learning_rate=0.05, max_depth=6, max_iter=600, l2_regularization=0.0, min_samples_leaf=10),
        dict(learning_rate=0.06, max_depth=7, max_iter=800, l2_regularization=0.1, min_samples_leaf=20),
        dict(learning_rate=0.08, max_depth=7, max_iter=900, l2_regularization=0.1, min_samples_leaf=10),
    ]
    for p in hgb_grid:
        def _b(p=p):
            return HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=p["learning_rate"],
                max_depth=p["max_depth"],
                max_iter=p["max_iter"],
                l2_regularization=p["l2_regularization"],
                min_samples_leaf=p["min_samples_leaf"],
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_STATE
            )
        builders.append(("HistGB", p, _b))

    # RandomForest — baseline ổn định
    rf_grid = [
        dict(n_estimators=800, max_depth=None, min_samples_leaf=1),
        dict(n_estimators=1000, max_depth=20, min_samples_leaf=2),
    ]
    for p in rf_grid:
        def _b(p=p):
            return RandomForestRegressor(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        builders.append(("RandomForest", p, _b))

    # GradientBoosting — truyền thống
    gbr_grid = [
        dict(learning_rate=0.05, n_estimators=700, max_depth=3),
        dict(learning_rate=0.08, n_estimators=600, max_depth=3),
    ]
    for p in gbr_grid:
        def _b(p=p):
            return GradientBoostingRegressor(
                learning_rate=p["learning_rate"],
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                random_state=RANDOM_STATE
            )
        builders.append(("GBR", p, _b))

    # XGBoost — nếu có
    try:
        from xgboost import XGBRegressor  # type: ignore
        xgb_grid = [
            dict(n_estimators=900, max_depth=7, learning_rate=0.05,
                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0),
            dict(n_estimators=1100, max_depth=8, learning_rate=0.05,
                 subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5),
        ]
        for p in xgb_grid:
            def _b(p=p):
                return XGBRegressor(
                    **p,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            builders.append(("XGBoost", p, _b))
    except Exception:
        pass

    # LightGBM — nếu có
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        lgb_grid = [
            dict(n_estimators=1200, learning_rate=0.05, max_depth=-1,
                 num_leaves=63, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0),
            dict(n_estimators=1500, learning_rate=0.05, max_depth=-1,
                 num_leaves=95, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5),
        ]
        for p in lgb_grid:
            def _b(p=p):
                return LGBMRegressor(
                    **p,
                    objective="regression",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            builders.append(("LightGBM", p, _b))
    except Exception:
        pass

    # CatBoost — nếu có (tắt stdout)
    try:
        from catboost import CatBoostRegressor  # type: ignore
        cb_grid = [
            dict(iterations=1200, learning_rate=0.05, depth=6, l2_leaf_reg=3.0),
            dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=5.0),
        ]
        for p in cb_grid:
            def _b(p=p):
                return CatBoostRegressor(
                    **p, loss_function="RMSE", random_state=RANDOM_STATE,
                    verbose=False
                )
            builders.append(("CatBoost", p, _b))
    except Exception:
        pass

    return builders

# ---------- Main ----------
def main():
    df = load_data(CSV_INPUT)
    n = len(df)
    if n < 40:
        print(f"⚠️ cleaned_rental_data quá ít ({n} dòng). Vẫn train nhưng CV k=3.")
    else:
        print(f"📦 Dữ liệu train: {n} dòng.")

    # features & target
    X_all, feat_cols = build_features(df)
    y_log = np.log1p(df["GiaVND"].values)

    # hold-out test
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_log, test_size=0.2, random_state=RANDOM_STATE)
    kfold = 3 if n < 120 else 5

    # chọn model tốt nhất theo CV RMSE (VND)
    print(f"🔎 Tìm mô hình tốt nhất (k={kfold}) — đơn vị RMSE = VND")
    best = {"name": None, "params": None, "cv_rmse": float("inf")}
    for name, params, builder in get_model_builders():
        try:
            cv_rmse = cv_score_rmse_vnd_log_target(X_tr, y_tr, builder, n_splits=kfold)
        except Exception as e:
            print(f"  {name} {params}  ->  SKIP ({e})")
            continue
        print(f"  {name} {params}  ->  CV_RMSE = {human_money(cv_rmse)}")
        if cv_rmse < best["cv_rmse"]:
            best.update(name=name, params=params, cv_rmse=cv_rmse)

    # fallback nếu không có ứng viên nào chạy được
    if best["name"] is None:
        print("⚠️ Không có mô hình nào huấn luyện được — dùng HistGB mặc định.")
        def _fallback():
            return HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.06, max_depth=7, max_iter=800,
                l2_regularization=0.1, min_samples_leaf=20, early_stopping=True,
                validation_fraction=0.15, random_state=RANDOM_STATE
            )
        best = {"name": "HistGB(fallback)", "params": {}, "cv_rmse": float("nan")}
        model = _fallback()
    else:
        # build model theo best
        for name, params, builder in get_model_builders():
            if name == best["name"] and params == best["params"]:
                model = builder()
                break

    model.fit(X_tr, y_tr)

    # đánh giá test (quy về VND)
    yhat_vnd_te  = np.expm1(model.predict(X_te))
    ytrue_vnd_te = np.expm1(y_te)
    mae  = mean_absolute_error(ytrue_vnd_te, yhat_vnd_te)
    rmse = safe_rmse(ytrue_vnd_te, yhat_vnd_te)
    r2   = r2_score(ytrue_vnd_te, yhat_vnd_te)

    print("\n✅ Train xong. Metrics (test):")
    print(f"- MAE : {human_money(mae)}")
    print(f"- RMSE: {human_money(rmse)}")
    print(f"- R^2 : {r2:.3f}")
    print(f"🏆 Best by CV: {best['name']} {best['params']}  |  CV_RMSE ≈ {human_money(best['cv_rmse'])}")

    # lưu model + feature cols + metrics
    joblib.dump(model, MODEL_OUT)
    with open(FEATS_JSON, "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)
    with open(METRICS_JS, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best["name"],
            "best_params": best["params"],
            "cv_rmse_vnd": float(best["cv_rmse"]) if best["cv_rmse"] == best["cv_rmse"] else None,
            "test_mae_vnd": float(mae),
            "test_rmse_vnd": float(rmse),
            "test_r2": float(r2),
            "target": "log_price_vnd",
            "n_rows": int(n),
            "kfold": int(kfold)
        }, f, indent=2)

    print(f"\n💾 Model  → {MODEL_OUT}")
    print(f"💾 Feats  → {FEATS_JSON}  (DienTich + Phuong_*)")
    print(f"💾 Metrics→ {METRICS_JS}")

if __name__ == "__main__":
    main()
