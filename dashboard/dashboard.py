# -*- coding: utf-8 -*-
# Dashboard phân tích & dự đoán giá thuê Q7 (Streamlit)
# - Đọc data từ cleaned_rental_data.csv (hoặc DB SQLite read-only)
# - Dự đoán bằng models/rf_model.pkl; TỰ KHỚP cột từ model
# - Quy đổi dự đoán về VND theo 'target' trong models/metrics.json

import os, re, json
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import sqlite3
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# Path resolver (WSL/Windows)
# ======================
def pick_root():
    for p in ("/mnt/d/PhanTichDuLieu", "/mnt/d/PHANTICHDULIEU", os.getcwd()):
        if os.path.exists(p):
            return p
    return os.getcwd()

ROOT = pick_root()
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

CSV_PATH_DEFAULT = os.path.join(DATA_DIR,  "cleaned_rental_data.csv")
DB_PATH_DEFAULT  = os.path.join(DATA_DIR,  "master.db")
MODEL_PATH       = os.path.join(MODEL_DIR, "best_model.pkl")
FEATURES_JSON    = os.path.join(MODEL_DIR, "feature_columns.json")
METRICS_JSON     = os.path.join(MODEL_DIR, "metrics.json")

CSV_PATH = os.getenv("BDS_CSV", CSV_PATH_DEFAULT)
DB_PATH  = os.getenv("BDS_DB_SQLITE", DB_PATH_DEFAULT)

# ================
# Page config
# ================
st.set_page_config(page_title="BĐS Q7 Dashboard", page_icon="🏙️", layout="wide")
st.title("🏙️ Dashboard – Giá thuê căn hộ Quận 7")
st.caption(f"CSV: `{CSV_PATH}`  •  DB fallback: `{DB_PATH}`  •  Model: `{MODEL_PATH}`")

# ===================
# Helpers
# ===================
def human_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    v = float(v)
    if abs(v) >= 1_000_000_000: return f"{v/1_000_000_000:.2f} tỷ"
    if abs(v) >= 1_000_000:     return f"{v/1_000_000:.1f} triệu"
    if abs(v) >= 1_000:         return f"{v/1_000:.0f} nghìn"
    return f"{v:.0f} đ"

def file_mtime(path: str) -> int:
    try: return int(os.path.getmtime(path))
    except Exception: return 0

@st.cache_data(show_spinner=False)
def load_csv(path: str, sig: int) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["TieuDe","GiaVND","DienTich","Phuong","Quan","Link","GiaTheoMetVuong"])
    df = pd.read_csv(path)
    # chuẩn hoá tên cột
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "tieude": ren[c] = "TieuDe"
        if cl == "giavnd": ren[c] = "GiaVND"
        if cl == "dientich": ren[c] = "DienTich"
        if cl == "phuong": ren[c] = "Phuong"
        if cl == "quan": ren[c] = "Quan"
        if cl == "link": ren[c] = "Link"
        if cl in ("giatheometvuong","gia_m2","gia_m²"): ren[c] = "GiaTheoMetVuong"
    if ren: df = df.rename(columns=ren)
    # ép kiểu
    for c in ("GiaVND","DienTich","GiaTheoMetVuong"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("Phuong","Quan","TieuDe","Link"):
        if c in df.columns: df[c] = df[c].astype("string").fillna("").str.strip()
    if "GiaTheoMetVuong" not in df.columns and set(["GiaVND","DienTich"]).issubset(df.columns):
        df["GiaTheoMetVuong"] = df["GiaVND"]/df["DienTich"]
    return df

def load_db_sqlite(path: str) -> pd.DataFrame:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000;")
        q = """
        SELECT
            title     AS TieuDe,
            price_vnd AS GiaVND,
            area      AS DienTich,
            ward      AS Phuong,
            district  AS Quan,
            url       AS Link,
            address   AS DiaChi
        FROM listings
        WHERE is_active=1 AND price_vnd IS NOT NULL AND area IS NOT NULL
        """
        return pd.read_sql_query(q, conn)
    finally:
        conn.close()

def safe_prepare(df: pd.DataFrame) -> pd.DataFrame:
    want = ["TieuDe","GiaVND","DienTich","Phuong","Quan","Link","DiaChi","GiaTheoMetVuong"]
    # map lower->proper
    ren = {}
    for w in want:
        if w not in df.columns:
            for x in df.columns:
                if x.lower() == w.lower():
                    ren[x] = w
    if ren: df = df.rename(columns=ren)

    # numeric
    df["GiaVND"] = pd.to_numeric(df.get("GiaVND"), errors="coerce")
    df["DienTich"] = pd.to_numeric(df.get("DienTich"), errors="coerce")
    if "GiaTheoMetVuong" not in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df["GiaTheoMetVuong"] = df["GiaVND"]/df["DienTich"]

    for c in ("Phuong","Quan"):
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("unknown").str.strip().str.lower()

    df = df.dropna(subset=["GiaVND","DienTich"])
    df = df[(df["GiaVND"]>0) & (df["DienTich"]>0)].reset_index(drop=True)

    for w in want:
        if w not in df.columns: df[w] = pd.NA
    return df

def get_data(prefer_db: bool=False):
    if not prefer_db and os.path.exists(CSV_PATH):
        sig = file_mtime(CSV_PATH)
        try:
            df = load_csv(CSV_PATH, sig)
            if len(df)>0: return safe_prepare(df), "csv"
        except Exception as e:
            st.warning(f"Đọc CSV lỗi: {e}")
    try:
        if os.path.exists(DB_PATH):
            df = load_db_sqlite(DB_PATH)
            if len(df)>0: return safe_prepare(df), "db"
    except Exception as e:
        st.error(f"Đọc DB lỗi: {e}")
    return pd.DataFrame(columns=["TieuDe","GiaVND","DienTich","Phuong","Quan","Link","DiaChi","GiaTheoMetVuong"]), "empty"

def load_model_and_features():
    model, feat_cols = None, None
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Không load được model: {e}")

    # ưu tiên cột từ model
    if model is not None and hasattr(model, "feature_names_in_"):
        try:
            feat_cols = list(model.feature_names_in_)
        except Exception:
            feat_cols = None

    # fallback: JSON
    if feat_cols is None and os.path.exists(FEATURES_JSON):
        try:
            with open(FEATURES_JSON, "r", encoding="utf-8") as f:
                feat_cols = json.load(f)
        except Exception:
            feat_cols = None
    return model, feat_cols

def load_model_meta():
    try:
        if os.path.exists(METRICS_JSON):
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

MODEL_META = load_model_meta()

def preds_to_vnd(yhat, df_ref=None):
    """
    Quy đổi dự đoán về VND/tháng theo 'target' khi train:
      - 'log_price_vnd' : expm1(yhat)
      - 'log_price_pm2' : expm1(yhat) * DienTich
      - mặc định        : giữ nguyên
    """
    tgt = str(MODEL_META.get("target","")).lower()
    y = np.asarray(yhat, dtype="float64")
    if "log" in tgt:
        y = np.expm1(y)
    if "pm2" in tgt and df_ref is not None and "DienTich" in df_ref.columns:
        area = pd.to_numeric(df_ref["DienTich"], errors="coerce").values
        y = y * area
    return y

def make_design_matrix_single(area_value: float, phuong_value: str, feat_cols: list[str], df_like: pd.DataFrame) -> pd.DataFrame:
    if feat_cols:
        X = pd.DataFrame(0.0, index=[0], columns=feat_cols, dtype=float)
        if "DienTich" in X.columns:
            X.at[0, "DienTich"] = float(area_value)
        col = f"Phuong_{str(phuong_value)}"
        if col in X.columns:
            X.at[0, col] = 1.0
        return X

    # Fallback tự suy ra từ df_like
    X = pd.DataFrame({"DienTich":[float(area_value)]})
    if "Phuong" in df_like.columns:
        for p in sorted(df_like["Phuong"].dropna().astype(str).unique().tolist()):
            X[f"Phuong_{p}"] = 1.0 if str(phuong_value)==p else 0.0
    return X

# =========================
# Sidebar – nguồn dữ liệu & bộ lọc
# =========================
with st.sidebar:
    st.header("Nguồn dữ liệu")
    prefer_db = st.toggle("Dùng DB (bỏ qua CSV)", value=False)
    df, source = get_data(prefer_db=prefer_db)

    if source == "csv":
        mt = file_mtime(CSV_PATH)
        if mt: st.caption(f"🗓 CSV cập nhật: {datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M:%S')}")
    elif source == "db":
        st.caption("Nguồn: Database (read-only)")
    else:
        st.caption("Chưa có dữ liệu. Hãy chạy pipeline trước.")

    st.divider()
    st.header("Bộ lọc")

    if len(df)>0:
        p5,p95 = np.nanpercentile(df["GiaVND"], [5,95])
        a5,a95 = np.nanpercentile(df["DienTich"], [5,95])
        price_range = st.slider("Giá thuê/tháng (VND)", 0, int(max(df["GiaVND"].max(), p95)),
                                value=(int(p5), int(p95)), step=500_000)
        area_range  = st.slider("Diện tích (m²)", 0, int(max(df["DienTich"].max(), a95)),
                                value=(int(a5), int(a95)), step=5)

        quans = ["(Tất cả)"] + sorted([x for x in df["Quan"].dropna().astype(str).unique().tolist() if x])
        sel_quan = st.selectbox("Quận", quans, index=0)

        if sel_quan != "(Tất cả)":
            phuong_opts = ["(Tất cả)"] + sorted([x for x in df[df["Quan"]==sel_quan]["Phuong"].dropna().astype(str).unique().tolist() if x])
        else:
            phuong_opts = ["(Tất cả)"] + sorted([x for x in df["Phuong"].dropna().astype(str).unique().tolist() if x])
        sel_phuong = st.selectbox("Phường", phuong_opts, index=0)

        keyword  = st.text_input("Từ khoá tiêu đề (cách nhau bằng dấu cách)", "", placeholder="sunrise river ...")
        adv_regex= st.text_input("Tìm kiếm nâng cao (regex)", "", placeholder="(sunrise|happy valley|eco green)")
        top_n    = st.selectbox("Số dòng hiển thị", [50,100,200,500,1000], index=1)

        st.divider()
        st.header("Clustering & Dự đoán")
        k_cluster = st.slider("Số cluster (KMeans)", 2, 6, 3)
        use_log_y = st.toggle("Log-scale trục Y (Giá)", value=False)

        ph_all = ["unknown"] + sorted([x for x in df["Phuong"].dropna().astype(str).unique().tolist() if x])
        default_ph = sel_phuong if sel_phuong!="(Tất cả)" else (ph_all[1] if len(ph_all)>1 else "unknown")
        pred_area  = st.number_input("Diện tích dự đoán (m²)", 10.0, 500.0, 70.0, 1.0)
        pred_phuong= st.selectbox("Phường (dự đoán)", ph_all, index=(ph_all.index(default_ph) if default_ph in ph_all else 0))
    else:
        price_range=(0,1_000_000_000); area_range=(0,500)
        sel_quan=sel_phuong="(Tất cả)"; keyword=adv_regex=""; top_n=100
        k_cluster=3; use_log_y=False; pred_area=70.0; pred_phuong="unknown"

# =========================
# Áp bộ lọc
# =========================
df_show = df.copy()
if len(df_show)>0:
    df_show = df_show[(df_show["GiaVND"]>=price_range[0]) & (df_show["GiaVND"]<=price_range[1])]
    df_show = df_show[(df_show["DienTich"]>=area_range[0]) & (df_show["DienTich"]<=area_range[1])]
    if sel_quan!="(Tất cả)":   df_show = df_show[df_show["Quan"]==sel_quan]
    if sel_phuong!="(Tất cả)": df_show = df_show[df_show["Phuong"]==sel_phuong]
    if keyword.strip():
        for tok in keyword.lower().split():
            df_show = df_show[df_show["TieuDe"].fillna("").str.lower().str.contains(re.escape(tok))]
    if adv_regex.strip():
        try:
            rx = re.compile(adv_regex, flags=re.IGNORECASE)
            df_show = df_show[df_show["TieuDe"].fillna("").str.contains(rx)]
        except re.error as e:
            st.warning(f"Regex không hợp lệ: {e}")

# =========================
# KPI
# =========================
n_rows = len(df_show)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Số bản ghi", f"{n_rows:,}")
c2.metric("Giá trung vị / tháng", human_money(float(df_show["GiaVND"].median())) if n_rows else "—")
c3.metric("Giá trung vị / m²", human_money(float(df_show["GiaTheoMetVuong"].median()))+"/m²" if n_rows else "—")
c4.metric("Diện tích TB", f"{float(df_show['DienTich'].mean()):.1f} m²" if n_rows else "—")

# =========================
# Biểu đồ
# =========================
if n_rows>0:
    st.subheader("Quan hệ Giá thuê – Diện tích (scatter)")
    vis = df_show.dropna(subset=["GiaVND","DienTich"]).copy()
    vis["Giá (triệu)"] = vis["GiaVND"]/1_000_000
    fig_scatter = px.scatter(
        vis, x="DienTich", y="Giá (triệu)", color="Phuong",
        hover_data=["TieuDe","Phuong","Quan","DienTich","GiaVND","Link"],
        title="Giá (triệu VND) theo Diện tích"
    )
    if use_log_y:
        fig_scatter.update_yaxes(type="log")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Giá/m² trung vị theo Phường (Top 20)")
    med = (df_show.dropna(subset=["Phuong","GiaTheoMetVuong"])
           .groupby("Phuong", as_index=False)["GiaTheoMetVuong"].median()
           .sort_values("GiaTheoMetVuong", ascending=False).head(20))
    med["Giá/m² (nghìn)"] = (med["GiaTheoMetVuong"]/1000).round(1)
    fig_bar = px.bar(med, x="Giá/m² (nghìn)", y="Phuong", orientation="h",
                     title="Giá/m² trung vị (nghìn VND)", text="Giá/m² (nghìn)")
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Phân bố Giá thuê (histogram)")
    fig_hist = px.histogram(df_show.assign(GiaTrieu=df_show["GiaVND"]/1_000_000),
                            x="GiaTrieu", nbins=40, title="Phân bố giá (triệu VND/tháng)",
                            labels={"GiaTrieu":"Giá (triệu VND)"})
    st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# Clustering (KMeans)
# =========================
if n_rows>0:
    st.subheader("Clustering KMeans (DiệnTích & GiáVND)")
    cl = df_show.dropna(subset=["DienTich","GiaVND"]).copy()
    if len(cl) >= 10:
        scaler = StandardScaler()
        X = scaler.fit_transform(cl[["DienTich", "GiaVND"]].values)
        km = KMeans(n_clusters=k_cluster, n_init="auto", random_state=42)
        cl["cluster"] = km.fit_predict(X).astype(str)
        fig_cl = px.scatter(
            cl.assign(GiaTrieu=cl["GiaVND"]/1_000_000),
            x="DienTich",
            y="GiaTrieu",
            color="cluster",
            hover_data=["TieuDe", "Phuong", "Quan", "Link"],
            title="KMeans (trục Y = triệu VND)",
        )
        st.plotly_chart(fig_cl, use_container_width=True)
    else:
        st.info("Cần ≥ 10 dòng để chạy clustering.")

# ==========================
# Dự đoán 1 điểm (inference) — ĐÃ SỬA QUY ĐỔI VND
# ==========================
st.header("🔮 Dự đoán giá thuê (dùng model đã train)")
model, feat_cols = load_model_and_features()
if model is None:
    st.warning("Chưa tìm thấy model đã train. Hãy chạy pipeline để tạo models/rf_model.pkl")
else:
    X1 = make_design_matrix_single(pred_area, pred_phuong, feat_cols, df if len(df)>0 else pd.DataFrame({"Phuong":[]}))
    try:
        yhat_raw = model.predict(X1)
        yhat_vnd = preds_to_vnd(yhat_raw, df_ref=pd.DataFrame({"DienTich":[pred_area]}))
        per_m2 = float(yhat_vnd[0])/max(float(pred_area), 1e-6)
        st.success(f"Giá dự đoán: **{human_money(float(yhat_vnd[0]))}** / tháng • **{human_money(per_m2)}**/m²")
    except Exception as e:
        st.error(f"Lỗi khi predict: {e}")

# =========================
# Dữ liệu chi tiết & Xuất CSV/Excel
# =========================
st.subheader("Bảng dữ liệu (đã lọc)")
show_cols = [c for c in ["TieuDe","Phuong","Quan","DienTich","GiaVND","GiaTheoMetVuong","Link","DiaChi"] if c in df_show.columns]
st.dataframe(df_show[show_cols].head(top_n), use_container_width=True)

csv_bytes = df_show[show_cols].to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Tải CSV đã lọc", data=csv_bytes, file_name="q7_filtered.csv", mime="text/csv")

def make_excel_bytes(df_full: pd.DataFrame) -> bytes:
    out = BytesIO()
    # ưu tiên xlsxwriter
    try:
        with pd.ExcelWriter(out, engine="xlsxwriter") as w:
            df_full.to_excel(w, sheet_name="Data", index=False)
            if {"Phuong","GiaTheoMetVuong"}.issubset(df_full.columns):
                g = (df_full.dropna(subset=["Phuong","GiaTheoMetVuong"])
                     .groupby("Phuong", as_index=False)
                     .agg(Count=("GiaVND","count"),
                          Median_Gia=("GiaVND","median"),
                          Median_Gia_m2=("GiaTheoMetVuong","median")))
                g.to_excel(w, sheet_name="Summary_Phuong", index=False)
        return out.getvalue()
    except Exception:
        pass
    # fallback openpyxl
    try:
        import openpyxl  # noqa
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            df_full.to_excel(w, sheet_name="Data", index=False)
        return out.getvalue()
    except Exception:
        st.warning("Không tạo được Excel (hãy cài xlsxwriter hoặc openpyxl).")
        return b""

xls = make_excel_bytes(df_show[show_cols])
if xls:
    st.download_button("⬇️ Tải Excel (đa sheet)", data=xls,
                       file_name=f"q7_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================
# Đánh giá model (bulk) — ĐÃ SỬA QUY ĐỔI VND
# =========================
with st.expander("Đánh giá model (nếu có ground truth & predict hàng loạt)"):
    if model is not None and len(df_show) >= 30:
        # dựng ma trận đặc trưng theo đúng cột model
        def build_matrix_from_df(df_in: pd.DataFrame):
            cols = None
            if hasattr(model, "feature_names_in_"):
                try: cols = list(model.feature_names_in_)
                except Exception: cols = None
            if cols is None and os.path.exists(FEATURES_JSON):
                try:
                    with open(FEATURES_JSON, "r", encoding="utf-8") as f:
                        cols = json.load(f)
                except Exception:
                    cols = None
            if cols:
                X = pd.DataFrame(0.0, index=df_in.index, columns=cols)
                if "DienTich" in X.columns: X["DienTich"] = pd.to_numeric(df_in["DienTich"], errors="coerce")
                if "Phuong" in df_in.columns:
                    ph = df_in["Phuong"].astype(str)
                    for i, p in enumerate(ph):
                        c = f"Phuong_{p}"
                        if c in X.columns: X.iat[i, X.columns.get_loc(c)] = 1.0
                return X
            # fallback suy diễn
            X = pd.DataFrame({"DienTich": pd.to_numeric(df_in["DienTich"], errors="coerce")}, index=df_in.index)
            if "Phuong" in df_in.columns:
                for p in sorted(df_in["Phuong"].dropna().astype(str).unique().tolist()):
                    X[f"Phuong_{p}"] = (df_in["Phuong"].astype(str) == p).astype(float)
            return X

        try:
            df_eval = df_show.dropna(subset=["GiaVND","DienTich","Phuong"]).copy()
            if len(df_eval) >= 30:
                X_eval = build_matrix_from_df(df_eval)
                y_true = pd.to_numeric(df_eval["GiaVND"], errors="coerce").values

                # ❗ Quy đổi dự đoán về VND theo target lúc train
                y_hat_raw = model.predict(X_eval)
                y_hat = preds_to_vnd(y_hat_raw, df_ref=df_eval)

                mae = mean_absolute_error(y_true, y_hat)
                try:
                    rmse = mean_squared_error(y_true, y_hat, squared=False)
                except TypeError:
                    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
                r2 = r2_score(y_true, y_hat)
                st.write(f"MAE: {human_money(mae)}  |  RMSE: {human_money(rmse)}  |  R²: {r2:.3f}")
            else:
                st.info("Chưa đủ mẫu để đánh giá.")
        except Exception as e:
            st.warning(f"Không thể đánh giá model: {e}")
    else:
        st.info("Cần ≥ 30 dòng đã lọc và có model để đánh giá.")

st.caption("© Data pipeline Q7 • Streamlit dashboard — auto target conversion (metrics.json) • feature names synced from model")
