import os
import time
import random
import requests
from lxml import html
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2

load_dotenv()

# ---------------------------------------
# Tạo session để giữ kết nối (rất nhanh)
# ---------------------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
})

def get_category(url, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                time.sleep(1 + attempt)
                continue

            # Use resp.text to get properly decoded Unicode string
            tree = html.fromstring(resp.text)

            # Try to find category from breadcrumb or article metadata first
            # Look for breadcrumb navigation
            breadcrumbs = tree.xpath('//ul[@class="breadcrumb"]//li/a/@title | //nav[@class="breadcrumb"]//a/@title')
            if breadcrumbs:
                # Filter out "Trang chủ" / "Home" and return the first valid category
                for crumb in breadcrumbs:
                    if crumb and crumb not in ["Trang chủ", "Home", "Trang chá»§"]:
                        return crumb
            
            # Try multiple selectors for the menu
            selectors = [
                'a[data-medium^="Menu-"]',  # Any <a> with data-medium starting with "Menu-"
                'div#menu-top a[data-medium^="Menu-"]',
                'header a[data-medium^="Menu-"]',
                'nav a[data-medium^="Menu-"]'
            ]
            
            for selector in selectors:
                elements = tree.cssselect(selector)
                if elements:
                    # Find the one that matches the URL path
                    for elem in elements:
                        href = elem.get("href", "")
                        # Skip "Trang chủ" / home links
                        title = elem.get("title", "").strip()
                        text = elem.text_content().strip()
                        if title in ["Trang chủ", "Home"] or text in ["Trang chủ", "Home"]:
                            continue
                        if href and href.strip('/') and href.strip('/') in url:
                            category = title or text
                            if category:
                                return category

            # Fallback: thử thẻ a trong menu chính
            elements = tree.cssselect('nav.main-nav a')
            if elements:
                # tìm cái nào có href khớp bài viết
                for a in elements:
                    href = a.get("href", "")
                    if href and href in url:
                        return a.text_content().strip()

            return None

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(1 + attempt)

    return None



# ---------------------------------------
# PostgreSQL
# ---------------------------------------
conn = psycopg2.connect(
    host=os.getenv("HOST"),
    database=os.getenv("DATABASE"),
    user=os.getenv("USER_DB"),
    password=os.getenv("PASSWORD")
)
cur = conn.cursor()
cur.execute("SELECT id, url FROM articles WHERE category IS NULL")
rows = cur.fetchall()

batch_updates = []
batch_size = 200   # có thể tăng

def worker(row):
    row_id, url = row
    print("Fetch:", url)

    cat = get_category(url)
    print("=>", cat)

    # Random delay để giảm bị block
    time.sleep(random.uniform(0.03, 0.07))

    return (cat, row_id) if cat else None


# ---------------------------------------
# Thread pool (giảm xuống 4 cho an toàn)
# ---------------------------------------
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(worker, row) for row in rows]

    for f in as_completed(futures):
        r = f.result()
        if r:
            batch_updates.append(r)

        if len(batch_updates) >= batch_size:
            cur.executemany(
                "UPDATE articles SET category = %s WHERE id = %s",
                batch_updates
            )
            conn.commit()
            batch_updates = []

# commit cuối
if batch_updates:
    cur.executemany(
        "UPDATE articles SET category = %s WHERE id = %s",
        batch_updates
    )
    conn.commit()

cur.close()
conn.close()
