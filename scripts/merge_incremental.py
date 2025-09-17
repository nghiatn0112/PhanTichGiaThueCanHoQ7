# /mnt/d/PHANTICHDULIEU/scripts/merge_incremental.py
import os, sys, hashlib, sqlite3, re
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from itertools import zip_longest

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB     = os.path.join(ROOT, "data", "master.db")
TABLE  = "listings"
# RAW_CSV: optional CLI argument; if not provided we'll try sensible defaults below
RAW_CSV = sys.argv[1] if len(sys.argv) > 1 else None

# --- Chuẩn hoá/parse "raw" ---
def parse_price_vnd(s):
    if pd.isna(s): return None
    t = str(s).lower().replace(",", ".")
    if "thỏa thuận" in t or "thoả thuận" in t: return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", t)
    if not m: return None
    v = float(m.group(1))
    if any(u in t for u in ["tỷ","ty","billion","bn"]): v *= 1_000_000_000
    elif any(u in t for u in ["triệu","tr","m","mil"]): v *= 1_000_000
    elif any(u in t for u in ["nghìn","ngàn","k"]):      v *= 1_000
    if any(u in t for u in ["/tháng","/thang","tháng"]): return int(v)
    if any(u in t for u in ["/năm","/nam"]):             return int(v/12.0)
    return int(v)  # mặc định /tháng

def parse_area_m2(s):
    if pd.isna(s): return None
    t = str(s).lower().replace(",", ".")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", t)
    return float(m.group(1)) if m else None

def split_location(loc):
    if pd.isna(loc): return None, None, None
    parts = [p.strip() for p in str(loc).split(",") if p.strip()]
    if len(parts) >= 3: return parts[-3], parts[-2], parts[-1]
    if len(parts) == 2:  return parts[0], parts[1], None
    return None, None, None

# --- Map tên cột đa biến thể ---
# (ưu tiên cleaned; nếu thiếu sẽ thử raw)
VARIANTS = {
    "title":      ["title","tieu_de","tieude","TieuDe"],
    "address":    ["address","dia_chi","diachi","DiaChi"],
    "ward":       ["ward","phuong","Phuong"],
    "district":   ["district","quan","Quan"],
    "area":       ["area","dientich_m2","dien_tich_m2","DienTich","dientich","dien_tich"],
    "price_vnd":  ["price_vnd","gia_vnd_thang","GiaVND","gia_vnd"],
    "url":        ["url","link","Link"],
    "listing_id": ["listing_id","ListingID","listingid"],
    # raw fields:
    "gia_raw":        ["gia_raw","GiaRaw","gia"],
    "dientich_raw":   ["dientich_raw","DienTichRaw","dien_tich_raw"],
    "vitri_raw":      ["vitri_raw","ViTriRaw","vi_tri_raw","location","Location"],
}

STD_COLS = ["listing_id","url","title","address","ward","district","area","price_vnd"]

def norm(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    return " ".join(s.split())

def rid(row):
    key = (row.get("listing_id") or "") + \
          norm(row.get("url") or "") + norm(row.get("title") or "") + \
          norm(row.get("address") or "") + norm(row.get("ward") or "") + \
          str(row.get("area") or "")
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def hp(row):
    payload = "|".join([
        norm(row.get("title") or ""), norm(row.get("address") or ""),
        norm(row.get("ward") or ""), norm(row.get("district") or ""),
        str(row.get("area") or ""), str(row.get("price_vnd") or "")
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def ensure_table(conn):
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE}(
      record_id TEXT PRIMARY KEY,
      listing_id TEXT, url TEXT, title TEXT, address TEXT,
      ward TEXT, district TEXT, area REAL, price_vnd REAL,
      hash_payload TEXT, first_seen_at TEXT, last_seen_at TEXT, is_active INTEGER
    );""")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_hash ON {TABLE}(hash_payload);")
    conn.commit()

def find_first(df, names):
    for n in names:
        if n in df.columns: return n
        # hỗ trợ tên cột bỏ dấu cách/underscore & lowercase
        nl = n.lower().replace(" ", "").replace("_", "")
        for c in df.columns:
            cl = c.lower().replace(" ", "").replace("_", "")
            if cl == nl: return c
    return None

def normalize_df(df0):
    cols = {k: find_first(df0, v) for k, v in VARIANTS.items()}
    out = pd.DataFrame()

    # string cols: ensure we always create a Series of correct length
    for k in ["title","address","ward","district","url","listing_id"]:
        c = cols.get(k)
        if c:
            out[k] = df0[c]
        else:
            out[k] = pd.Series([pd.NA] * len(df0), dtype="string")

    # area/price: ưu tiên cleaned; nếu thiếu -> parse từ raw
    c_area  = cols.get("area")
    c_price = cols.get("price_vnd")
    if c_area is not None:
        out["area"] = pd.to_numeric(df0[c_area], errors="coerce")
    else:
        ca = cols.get("dientich_raw")
        out["area"] = df0[ca].map(parse_area_m2) if ca else np.nan

    if c_price is not None:
        out["price_vnd"] = pd.to_numeric(df0[c_price], errors="coerce")
    else:
        cp = cols.get("gia_raw")
        out["price_vnd"] = df0[cp].map(parse_price_vnd) if cp else np.nan

    # ward/district từ vitri_raw nếu chưa có
    # ward/district từ vitri_raw nếu chưa có
    need_ward = out.get("ward") is None or out["ward"].isna().all()
    need_district = out.get("district") is None or out["district"].isna().all()
    if need_ward or need_district:
        cv = cols.get("vitri_raw")
        if cv:
            locs = df0[cv].map(split_location).tolist()
            ph = [l[0] for l in locs]
            q = [l[1] for l in locs]
            if need_ward:
                out["ward"] = pd.Series(ph, dtype="string")
            if need_district:
                out["district"] = pd.Series(q, dtype="string")

    # fill missing columns
    for c in STD_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    # normalize text & types
    for c in ["listing_id","url","title","address","ward","district"]:
        out[c] = out[c].astype("string").map(norm)
    out["area"] = pd.to_numeric(out["area"], errors="coerce")
    out["price_vnd"] = pd.to_numeric(out["price_vnd"], errors="coerce")

    return out[STD_COLS]

def upsert(conn, df):
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    for _, r in df.iterrows():
        rid_ = r["record_id"]
        cur.execute(f"SELECT hash_payload FROM {TABLE} WHERE record_id=?", (rid_,))
        row = cur.fetchone()
        if row is None:
            cur.execute(f"""
            INSERT INTO {TABLE}
            (record_id, listing_id, url, title, address, ward, district, area, price_vnd,
             hash_payload, first_seen_at, last_seen_at, is_active)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rid_, r.get("listing_id"), r.get("url"), r.get("title"), r.get("address"),
             r.get("ward"), r.get("district"), r.get("area"), r.get("price_vnd"),
             r.get("hash_payload"), now, now, 1))
        else:
            if row[0] != r["hash_payload"]:
                cur.execute(f"""
                UPDATE {TABLE}
                SET url=?, title=?, address=?, ward=?, district=?, area=?, price_vnd=?,
                    hash_payload=?, last_seen_at=?, is_active=1
                WHERE record_id=?""",
                (r.get("url"), r.get("title"), r.get("address"), r.get("ward"),
                 r.get("district"), r.get("area"), r.get("price_vnd"),
                 r.get("hash_payload"), now, rid_))
            else:
                cur.execute(f"UPDATE {TABLE} SET last_seen_at=?, is_active=1 WHERE record_id=?",
                            (now, rid_))
    conn.commit()

def main():
    # choose RAW_CSV: CLI arg overrides, otherwise try common default locations
    global RAW_CSV
    if RAW_CSV is None:
        candidates = [
            os.path.join(ROOT, "data", "raw", "bds_quan7_raw.csv"),
            os.path.join(ROOT, "data", "bds_quan7_raw.csv"),
            os.path.join(ROOT, "data", "GiaThueCanHoQ7.csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                RAW_CSV = p
                break
    if RAW_CSV is None:
        print("Missing input CSV. Pass a file path as first argument or place a default CSV in data/raw or data.")
        sys.exit(1)

    try:
        df0 = pd.read_csv(RAW_CSV)
    except Exception as e:
        print(f"Failed to read CSV '{RAW_CSV}':", e)
        sys.exit(1)
    df  = normalize_df(df0)

    # record_id & hash payload
    df["record_id"]   = df.apply(rid, axis=1)
    df["hash_payload"] = df.apply(hp, axis=1)
    df = df.drop_duplicates(subset=["record_id"], keep="last").reset_index(drop=True)

    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB)
    ensure_table(conn)
    upsert(conn, df)

    # Xuất snapshot chỉ gồm bản ghi có ĐỦ giá & diện tích cho bước sau
    snap = pd.read_sql_query(f"""
      SELECT title AS TieuDe, price_vnd AS GiaVND, area AS DienTich,
             ward AS Phuong, district AS Quan, address AS DiaChi,
             url AS Link, listing_id AS ListingID
      FROM {TABLE}
      WHERE is_active=1 AND price_vnd IS NOT NULL AND area IS NOT NULL
    """, conn)
    out_csv = os.path.join(ROOT, "data", "GiaThueCanHoQ7.csv")
    snap.to_csv(out_csv, index=False, encoding="utf-8-sig")
    conn.close()
    print("✅ Upsert OK →", DB)
    print("✅ Snapshot  →", out_csv, f"({len(snap)} dòng)")

if __name__ == "__main__":
    main()
