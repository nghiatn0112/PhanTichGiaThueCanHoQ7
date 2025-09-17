# -*- coding: utf-8 -*-
"""
Crawler BĐS Q7 (không lấy thời gian đăng/mô tả)
- Thu thập: tieu_de, gia_raw, dientich_raw, vitri_raw, link, listing_id
- Xuất:
    1) RAW  : /mnt/d/PHANTICHDULIEU/data/raw/bds_quan7_raw.csv
    2) CLEAN: /mnt/d/PHANTICHDULIEU/data/GiaThueCanHoQ7.csv
      (gồm: tieu_de, gia_vnd_thang, dientich_m2, phuong, quan, link, listing_id)
"""

# pip install cloudscraper beautifulsoup4 pandas numpy

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException
import os

# ========== CẤU HÌNH ==========
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81'
]

CONFIG = {
    "base_url": "https://batdongsan.com.vn/cho-thue-can-ho-chung-cu-quan-7",
    "max_pages": 70,
    "max_workers": 5,
    "output_raw_file": "/mnt/d/PHANTICHDULIEU/data/raw/bds_quan7_raw.csv",
    "output_cleaned_file": "/mnt/d/PHANTICHDULIEU/data/GiaThueCanHoQ7.csv",
    "request_timeout": 30,
    "sleep_min": 1.5,
    "sleep_max": 3.5
}

LOG_DIR = "/mnt/d/PHANTICHDULIEU/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "log_crawl_bds.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ========== HTTP HELPERS ==========
def create_scraper():
    s = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    s.headers.update({
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.6,en;q=0.5',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Referer': 'https://www.google.com/'
    })
    return s

def get_with_retry(page_number, max_retry=3):
    url = f'{CONFIG["base_url"]}/p{page_number}'
    for i in range(max_retry):
        try:
            s = create_scraper()  # session riêng cho mỗi attempt (an toàn với thread)
            s.headers['User-Agent'] = random.choice(USER_AGENTS)
            r = s.get(url, timeout=CONFIG["request_timeout"])
            if r.status_code == 403 or "Access Denied" in r.text:
                logging.warning(f"[{page_number}] 403/AccessDenied (thử {i+1}/{max_retry})")
                time.sleep(random.uniform(CONFIG["sleep_min"], CONFIG["sleep_max"]))
                continue
            r.raise_for_status()
            return r
        except RequestException as e:
            logging.warning(f"[{page_number}] Lỗi mạng: {e} (thử {i+1}/{max_retry})")
            time.sleep(random.uniform(CONFIG["sleep_min"], CONFIG["sleep_max"]))
    return None

# ========== PARSE HELPERS ==========
def extract_listing_id(link_tag):
    # ưu tiên data-product-id
    pid = link_tag.get('data-product-id')
    if pid: return pid.strip()
    # fallback từ href: ...-pr<digits>.htm
    href = link_tag.get('href', '') or ''
    m = re.search(r'-pr(\d+)\.htm', href)
    return m.group(1) if m else None

def parse_listing(item_soup):
    """Trả về 1 dict: tieu_de, gia_raw, dientich_raw, vitri_raw, link, listing_id"""
    try:
        # link
        link_tag = item_soup.find('a', class_='js__product-link-for-product-id')
        if not link_tag or not link_tag.has_attr('href'):
            link_tag = item_soup.select_one('a[href*="-pr"][href$=".htm"]')
            if not link_tag:
                return None

        # title (fallback)
        title = item_soup.find('span', class_='pr-title') \
             or item_soup.select_one('.re__card-title, .re__card-title.js__card-title')

        price = item_soup.find('span', class_='re__card-config-price')
        area  = item_soup.find('span', class_='re__card-config-area')
        location = item_soup.find('div', class_='re__card-location')

        return {
            'tieu_de': title.get_text(strip=True) if title else None,
            'gia_raw': price.get_text(strip=True) if price else None,
            'dientich_raw': area.get_text(strip=True) if area else None,
            'vitri_raw': location.get_text(strip=True) if location else None,
            'link': "https://batdongsan.com.vn" + link_tag['href'],
            'listing_id': extract_listing_id(link_tag)
        }
    except Exception as e:
        logging.warning(f"Lỗi parse: {e}")
        return None

def crawl_page(page_number):
    """Cào 1 trang → list dict; KHÔNG trả None để không dừng toàn bộ job"""
    logging.info(f"Đang cào trang {page_number}")
    resp = get_with_retry(page_number)
    if resp is None:
        logging.error(f"[{page_number}] Thất bại sau retry.")
        return []

    soup = BeautifulSoup(resp.content, 'html.parser')
    main_container = soup.find('div', id='product-lists-web')
    if not main_container:
        logging.warning(f"[{page_number}] Không có product-lists-web.")
        return []

    cards = main_container.select('div.re__card-full') or main_container.select('div.re__card')
    if not cards:
        logging.info(f"[{page_number}] Hết dữ liệu.")
        return []

    data = []
    for item in cards:
        d = parse_listing(item)
        if d: data.append(d)
    logging.info(f"[{page_number}] Tìm thấy {len(data)} tin.")
    return data

# ========== CLEAN HELPERS (KHÔNG dùng time_post/description) ==========
def convert_price(price_str):
    if not isinstance(price_str, str) or not price_str.strip():
        return np.nan
    t = price_str.lower().replace(',', '.')
    if 'thỏa thuận' in t or 'thoả thuận' in t:
        return np.nan

    m = re.search(r'([0-9]+(?:\.[0-9]+)?)', t)
    if not m: return np.nan
    val = float(m.group(1))

    if any(u in t for u in ['tỷ','ty','billion','bn']):     val *= 1_000_000_000
    elif any(u in t for u in ['triệu','tr','m','mil']):     val *= 1_000_000
    elif any(u in t for u in ['nghìn','ngàn','k']):         val *= 1_000

    if any(u in t for u in ['/tháng','/thang','tháng']):
        return val
    if any(u in t for u in ['/năm','/nam']):
        return val / 12.0
    return val  # mặc định /tháng

def parse_area(area_str):
    if not isinstance(area_str, str) or not area_str.strip():
        return np.nan
    t = area_str.lower().replace(',', '.')
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)', t)
    return float(m.group(1)) if m else np.nan

def split_location(loc):
    if not isinstance(loc, str) or not loc.strip():
        return None, None, None
    parts = [p.strip() for p in loc.split(',') if p.strip()]
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1], None
    return None, None, None

def clean_data(df):
    logging.info("Bắt đầu làm sạch dữ liệu...")
    df['gia_vnd_thang'] = df['gia_raw'].apply(convert_price)
    df['dientich_m2']   = df['dientich_raw'].apply(parse_area)

    ph, q, tp = zip(*df['vitri_raw'].map(split_location))
    df['phuong'] = pd.Series(ph, dtype='string').str.lower()
    df['quan']   = pd.Series(q, dtype='string').str.lower()
    df['thanh_pho'] = pd.Series(tp, dtype='string')

    # Schema tối thiểu cho pipeline (GIỮ NGUYÊN các trường khác)
    cols = ['tieu_de','gia_vnd_thang','dientich_m2','phuong','quan','link','listing_id']
    for c in cols:
        if c not in df.columns: df[c] = pd.NA
    out = df[cols].copy()
    logging.info("Hoàn tất làm sạch dữ liệu.")
    return out

def save_to_csv(df, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logging.info(f"Đã lưu {len(df)} dòng → {filename}")
    except PermissionError:
        alt_filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}.csv"
        df.to_csv(alt_filename, index=False, encoding='utf-8-sig')
        logging.warning(f"File đang bận, đã lưu tạm → {alt_filename}")

# ========== MAIN ==========
def main():
    logging.info("=== BẮT ĐẦU QUÁ TRÌNH CÀO DỮ LIỆU ===")
    all_data = []

    # Dùng ThreadPool; mỗi trang lỗi chỉ bỏ qua, không dừng job
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as ex:
        futures = {ex.submit(crawl_page, p): p for p in range(1, CONFIG["max_pages"] + 1)}
        for fut in as_completed(futures):
            page_num = futures[fut]
            try:
                page_data = fut.result()
                if page_data:
                    all_data.extend(page_data)
            except Exception as e:
                logging.error(f"Trang {page_num} exception: {e}")
            time.sleep(random.uniform(CONFIG["sleep_min"], CONFIG["sleep_max"]))  # throttle

    if not all_data:
        logging.warning("Không thu thập được dữ liệu nào. Kết thúc.")
        return

    logging.info(f"=== CÀO DỮ LIỆU THÔ HOÀN TẤT - {len(all_data)} TIN ===")
    df_raw = pd.DataFrame(all_data)
    save_to_csv(df_raw, CONFIG["output_raw_file"])

    df_clean = clean_data(df_raw.copy())
    save_to_csv(df_clean, CONFIG["output_cleaned_file"])

    logging.info("=== TOÀN BỘ QUÁ TRÌNH ĐÃ HOÀN TẤT ===")

if __name__ == '__main__':
    main()
