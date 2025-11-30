import cloudscraper
from bs4 import BeautifulSoup
import sqlite3
import time
import random
from tqdm import tqdm
import concurrent.futures
import threading
import sys
import argparse
from datetime import datetime

# --- CẤU HÌNH ---
def get_args():
    parser = argparse.ArgumentParser(description="Scrape posts by year and month range")
    parser.add_argument('--start-year', type=int, required=True, help='Start year (e.g. 2024)')
    parser.add_argument('--start-month', type=int, required=True, help='Start month (e.g. 3)')
    parser.add_argument('--end-year', type=int, required=True, help='End year (e.g. 2025)')
    parser.add_argument('--end-month', type=int, required=True, help='End month (e.g. 11)')
    args = parser.parse_args()
    return args

def generate_year_month_range(start_year, start_month, end_year, end_month):
    months = []
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    current = start
    while current <= end:
        months.append((current.year, f"{current.month:02d}"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return months

args = get_args()
month_ranges = generate_year_month_range(args.start_year, args.start_month, args.end_year, args.end_month)
DB_FILE = f"luatkhoa_{args.start_year}.db"
SITEMAP_INDEX = "https://luatkhoa.com/sitemap_index.xml"
MAX_WORKERS = 5  # Số luồng chạy song song

# Khóa an toàn để đồng bộ hóa việc ghi vào Database
db_lock = threading.Lock()

# Tạo scraper
base_scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
)

def init_db():
    """Khởi tạo Database với cấu trúc mới"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # SQLite không dùng SERIAL, dùng INTEGER PRIMARY KEY AUTOINCREMENT
    # Đặt label DEFAULT là 'Fake'
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            description TEXT,
            content TEXT,
            label TEXT DEFAULT 'Fake',
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            published_date TIMESTAMP,
            category TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ Đã khởi tạo database: {DB_FILE} với Label mặc định là 'Fake'")

def is_valid_post_url(url):
    url = url.lower()
    if not url:
        return False
    # Filter out common image extensions
    if url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg')):
        return False
    # Must contain YEAR_FILTER and not end with a file extension (except .html)
    if YEAR_FILTER not in url:
        return False
    # Optionally, only allow .html or no extension
    if '.' in url.split('/')[-1] and not url.endswith('.html'):
        return False
    return True

def save_to_db(data):
    """Lưu 1 bài viết vào DB (Thread-safe)"""
    if not data or not is_valid_post_url(str(data.get('url', ''))): return
    
    with db_lock:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            
            # Không insert cột 'label' và 'scraped_at' để DB tự lấy giá trị mặc định ('Fake' và giờ hiện tại)
            c.execute('''
                INSERT OR IGNORE INTO articles (url, title, description, content, published_date, category)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data['url'], 
                data['title'], 
                data['description'], 
                data['content'], 
                data['published_date'], 
                data['category']
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ Lỗi ghi DB: {e}")

def get_2025_urls():
    """Lấy danh sách URL từ Sitemap"""
    print("⏳ Đang lấy danh sách URL từ Sitemap...")
    target_urls = []
    try:
        res = base_scraper.get(SITEMAP_INDEX)
        if res.status_code != 200: return []
        
        soup_index = BeautifulSoup(res.content, 'xml')
        sitemaps = [sm.text for sm in soup_index.find_all('loc') if 'post-sitemap' in sm.text]
        
        for sm_url in sitemaps:
            r = base_scraper.get(sm_url)
            if r.status_code == 200:
                soup_sm = BeautifulSoup(r.content, 'xml')
                urls = soup_sm.find_all('loc')
                for url_tag in urls:
                    url_text = url_tag.text
                    if is_valid_post_url(url_text):
                        target_urls.append(url_text)
    except Exception as e:
        print(f"❌ Lỗi lấy sitemap: {e}")
    return list(set(target_urls))

def scrape_and_save(url):
    """Worker: Tải -> Bóc tách thêm Description/Category -> Lưu"""
    try:
        # Check trùng trước để tăng tốc
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
            exists = cursor.fetchone()
            conn.close()
        
        if exists: return "Skipped"

        # Request
        time.sleep(random.uniform(0.5, 1.5))
        response = base_scraper.get(url, timeout=10)
        if response.status_code != 200: return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Title
        title_tag = soup.find('h1', class_='jeg_post_title')
        if not title_tag:
            title_tag = soup.find('h1', class_='entry-title')
        title = title_tag.get_text(strip=True) if title_tag else ""
        
        # 2. Content
        content_div = soup.find('div', class_='entry-content')
        if content_div:
            for script in content_div(["script", "style", "div.sharedaddy", "div.jp-relatedposts", "div.wpcnt"]):
                script.extract()
            content = content_div.get_text(separator='\n', strip=True)
        else:
            content = ""
        
        # 3. Published Date
        date_tag = soup.find('div', class_='jeg_meta_date')
        published_date = None
        if date_tag:
            date_link = date_tag.find('a')
            if date_link:
                published_date = date_link.get_text(strip=True)
        if not published_date:
            date_tag = soup.find('time', class_='entry-date')
            published_date = date_tag['datetime'] if date_tag and date_tag.has_attr('datetime') else None
            
        # 4. Description (Lấy từ meta tag hoặc đoạn đầu bài viết)
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content', '').strip()
        else:
            # Fallback: Lấy 200 ký tự đầu của content làm mô tả
            description = content[:200] + "..." if content else ""

        # 5. Category
        category = "Uncategorized"
        breadcrumbs = soup.find('div', id='breadcrumbs')
        if breadcrumbs:
            links = breadcrumbs.find_all('a')
            if links:
                category = links[-1].get_text(strip=True)
        else:
            cat_tag = soup.find('a', attrs={'rel': 'category tag'})
            if cat_tag:
                category = cat_tag.get_text(strip=True)

        data = {
            "url": url,
            "title": title,
            "description": description,
            "content": content,
            "published_date": published_date,
            "category": category
        }
        
        save_to_db(data)
        return "Success"

    except Exception as e:
        return None

def main():
    print(f"--- TOOL CÀO DỮ LIỆU SQLITE (Schema mới) ---")
    # 1. Khởi tạo DB
    init_db()
    all_urls = []
    for year, month in month_ranges:
        print(f"\n🔎 Đang lấy link cho {year}/{month}...")
        global YEAR_FILTER
        YEAR_FILTER = f"/{year}/{month}/"
        urls = get_2025_urls()
        print(f"✅ Tìm thấy {len(urls)} bài viết cho {year}/{month}.")
        all_urls.extend(urls)
    all_urls = list(set(all_urls))
    total_urls = len(all_urls)
    print(f"\n✅ Tổng cộng {total_urls} bài viết cần xử lý.")
    if total_urls == 0: return
    # 3. Chạy đa luồng
    print("🚀 Đang chạy đa luồng...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(scrape_and_save, all_urls), total=total_urls))
    print(f"\n🎉 Hoàn tất! Kiểm tra file: {DB_FILE}")

if __name__ == "__main__":
    main()