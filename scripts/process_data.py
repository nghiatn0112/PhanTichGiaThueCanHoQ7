import os
import numpy as np
import pandas as pd

INPUT  = "/mnt/d/PHANTICHDULIEU/data/GiaThueCanHoQ7.csv"   # snapshot từ merge
OUTPUT = "/mnt/d/PHANTICHDULIEU/data/cleaned_rental_data.csv"

def log(msg): print(msg, flush=True)

def iqr_bounds(s: pd.Series, k: float = 1.5):
    s = s.dropna()
    if len(s) == 0:
        return None, None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return lo, hi

def process_rental_data():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"Không thấy file input: {INPUT}")
    log(f"Bắt đầu xử lý file '{INPUT}'...")

    df = pd.read_csv(INPUT)
    n0 = len(df)
    log(f"- Số dòng ban đầu: {n0}")

    # 1) Chuẩn hoá tên cột → lowercase, bỏ space
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 2) Map về schema chuẩn chữ thường
    rename_map = {
        "tieude":"tieu_de",
        "giavnd":"gia_vnd_thang", "gia_vnd":"gia_vnd_thang", "gia_vnd_thang":"gia_vnd_thang",
        "dientich":"dientich_m2", "dien_tich":"dientich_m2", "dien_tich_m2":"dientich_m2",
        "phuong":"phuong", "quan":"quan"
    }
    for k,v in list(rename_map.items()):
        if k in df.columns and v != k:
            df.rename(columns={k:v}, inplace=True)

    # 3) Bổ sung cột thiếu tối thiểu
    for c in ["tieu_de","gia_vnd_thang","dientich_m2","phuong","quan"]:
        if c not in df.columns:
            df[c] = pd.NA

    # 4) Ép kiểu số & chuẩn hoá text
    df["gia_vnd_thang"] = pd.to_numeric(df["gia_vnd_thang"], errors="coerce")
    df["dientich_m2"]   = pd.to_numeric(df["dientich_m2"],   errors="coerce")
    df["phuong"] = df["phuong"].astype("string").fillna("unknown").str.strip().str.lower()
    df["quan"]   = df["quan"].astype("string").fillna("unknown").str.strip().str.lower()

    # 5) Lọc null bắt buộc
    df = df.dropna(subset=["gia_vnd_thang","dientich_m2"]).copy()
    log(f"- Sau drop NaN price/area: {len(df)}")

    # 6) Bỏ các giá trị không hợp lệ / không dương
    df = df[(df["gia_vnd_thang"] > 0) & (df["dientich_m2"] > 0)]
    log(f"- Sau loại <=0: {len(df)}")

    # 7) Tính giá/m2 (luôn tạo cột để tránh KeyError)
    df["GiaTheoMetVuong"] = np.where(
        df["dientich_m2"] > 0, df["gia_vnd_thang"] / df["dientich_m2"], np.nan
    ).round(2)

    # 8) Lọc outlier bằng IQR (robust). Nếu lọc xong = 0, sẽ nới lỏng.
    def apply_iqr_filter(df_in: pd.DataFrame):
        df1 = df_in.copy()
        # Bounds cho price
        lo_p, hi_p = iqr_bounds(df1["gia_vnd_thang"], k=1.5)
        # Bounds cho area
        lo_a, hi_a = iqr_bounds(df1["dientich_m2"], k=1.5)
        # Bounds cho price/m2
        lo_m2, hi_m2 = iqr_bounds(df1["GiaTheoMetVuong"], k=1.5)

        # Áp dụng với ngưỡng mềm (kết hợp với chặn cứng an toàn)
        def within(val, lo, hi, hard_lo=None, hard_hi=None):
            if pd.isna(val): return False
            if hard_lo is not None and val < hard_lo: return False
            if hard_hi is not None and val > hard_hi: return False
            if lo is not None and val < lo: return False
            if hi is not None and val > hi: return False
            return True

        HARD = {
            "price_lo": 300_000,   # 0.3 triệu
            "price_hi": 300_000_000,  # 300 triệu
            "area_lo": 10,
            "area_hi": 400,
            "m2_lo": 10_000,       # 10k VND/m2 (~quá rẻ)
            "m2_hi": 1_000_000,    # 1 triệu VND/m2 (~quá đắt)
        }

        mask = df1.apply(lambda r: within(r["gia_vnd_thang"], lo_p, hi_p, HARD["price_lo"], HARD["price_hi"]) and
                                   within(r["dientich_m2"],   lo_a, hi_a, HARD["area_lo"], HARD["area_hi"]) and
                                   within(r["GiaTheoMetVuong"], lo_m2, hi_m2, HARD["m2_lo"], HARD["m2_hi"]), axis=1)
        return df1[mask].copy()

    df_filtered = apply_iqr_filter(df)
    log(f"- Sau IQR filter: {len(df_filtered)}")

    # 9) Nếu rỗng → fallback nới lỏng: chỉ dùng chặn cứng an toàn
    if len(df_filtered) == 0:
        log("⚠️ IQR filter loại hết → dùng ngưỡng cứng an toàn.")
        df_filtered = df[(df["gia_vnd_thang"].between(300_000, 300_000_000)) &
                         (df["dientich_m2"].between(10, 400)) &
                         (df["GiaTheoMetVuong"].between(10_000, 1_000_000))].copy()
        log(f"- Sau fallback hard-bounds: {len(df_filtered)}")

    # 10) Nếu vẫn 0 → chấp nhận dữ liệu đã chuẩn hoá nhưng không lọc (để không chặn pipeline)
    if len(df_filtered) == 0:
        log("⚠️ Dữ liệu vẫn 0 sau fallback → xuất file rỗng với schema chuẩn để pipeline không crash.")
        out = df.head(0).rename(columns={
            "tieu_de":"TieuDe",
            "gia_vnd_thang":"GiaVND",
            "dientich_m2":"DienTich",
            "phuong":"Phuong",
            "quan":"Quan",
        })[["TieuDe","GiaVND","DienTich","Phuong","Quan"]]
        out["GiaTheoMetVuong"] = pd.Series(dtype="float")
        out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
        log(f"✅ Đã lưu cleaned: '{OUTPUT}' với {len(out)} dòng.")
        return

    # 11) Đổi tên về in-hoa cho dashboard/model
    out = df_filtered.rename(columns={
        "tieu_de":"TieuDe",
        "gia_vnd_thang":"GiaVND",
        "dientich_m2":"DienTich",
        "phuong":"Phuong",
        "quan":"Quan",
    })[["TieuDe","GiaVND","DienTich","Phuong","Quan","GiaTheoMetVuong"]]

    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    log(f"✅ Đã lưu cleaned: '{OUTPUT}' với {len(out)} dòng. (ban đầu {n0})")

if __name__ == "__main__":
    process_rental_data()
