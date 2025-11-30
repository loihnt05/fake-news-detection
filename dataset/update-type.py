import os
import requests
from bs4 import BeautifulSoup
import psycopg2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
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



# Update these connection parameters as needed
conn = psycopg2.connect(
    host=os.getenv("HOST"),
    database=os.getenv("DATABASE"),
    user=os.getenv("USER_DB"),
    password=os.getenv("PASSWORD")
)
cur = conn.cursor()


cur.execute("SELECT id, url FROM articles WHERE category IS NULL")
rows = cur.fetchall()




batch_size = 50
batch_updates = []

def fetch_category(row):
    row_id, url = row
    print(f"Crawling: {url}")
    cat = get_category(url)
    print(f" => Category: {cat}")
    time.sleep(0.05)
    return (cat, row_id) if cat else None

with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_row = {executor.submit(fetch_category, row): row for row in rows}
    for future in as_completed(future_to_row):
        result = future.result()
        if result:
            batch_updates.append(result)
        if len(batch_updates) >= batch_size:
            cur.executemany(
                "UPDATE articles SET category = %s WHERE id = %s",
                batch_updates
            )
            conn.commit()
            batch_updates = []

# Final commit for any remaining updates
if batch_updates:
    cur.executemany(
        "UPDATE articles SET category = %s WHERE id = %s",
        batch_updates
    )
    conn.commit()

cur.close()
conn.close()