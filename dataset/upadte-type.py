import requests
from bs4 import BeautifulSoup
import sqlite3
import time

import requests
from bs4 import BeautifulSoup

def get_category(url):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        # breadcrumb: category cha = link đầu tiên
        breadcrumb = soup.select("ul.breadcrumb li a")

        if breadcrumb and len(breadcrumb) > 0:
            return breadcrumb[0].get_text(strip=True)

        return None

    except Exception as e:
        print("Error:", e)
        return None


conn = sqlite3.connect("fn.db")
cur = conn.cursor()

cur.execute("SELECT id, url FROM articles WHERE category IS NULL")
rows = cur.fetchall()

for row_id, url in rows:
    print("Crawling:", url)
    cat = get_category(url)

    print(" => Category:", cat)

    if cat:
        cur.execute(
            "UPDATE articles SET category = ? WHERE id = ?",
            (cat, row_id)
        )
        conn.commit()

    time.sleep(0.3)

cur.close()
conn.close()
print(get_category("https://vnexpress.net/psg-thang-trong-ngay-vang-messi-4602373.html"))