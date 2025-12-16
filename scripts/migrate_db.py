import sqlite3
import psycopg2
import json
from sentence_transformers import SentenceTransformer
from fact_extractor import FactExtractor
from tqdm import tqdm # Thư viện thanh tiến trình
import os

# --- CẤU HÌNH ---
SQLITE_PATH = "data/news_dataset.db" # Đường dẫn file của bạn

# Cấu hình Postgres (Khớp .env)
PG_CONFIG = {
    "dbname": "airflow",
    "user": "airflow",
    "password": "airflow",
    "host": "localhost",
    "port": "5432"
}

def migrate():
    print("🚀 Bắt đầu di trú dữ liệu từ SQLite sang PostgreSQL...")

    # 1. Kết nối SQLite
    if not os.path.exists(SQLITE_PATH):
        print(f"❌ Không tìm thấy file {SQLITE_PATH}")
        return

    conn_sqlite = sqlite3.connect(SQLITE_PATH)
    cur_sqlite = conn_sqlite.cursor()
    
    # Đếm tổng số bài để hiển thị thanh loading
    cur_sqlite.execute("SELECT COUNT(*) FROM articles")
    total_rows = cur_sqlite.fetchone()[0]
    print(f"📦 Tổng số bài báo cần xử lý: {total_rows}")

    # 2. Load Models
    print("⏳ Đang load AI Models (SBERT + Extractor)...")
    sbert = SentenceTransformer('keepitreal/vietnamese-sbert')
    extractor = FactExtractor()

    # 3. Kết nối Postgres
    conn_pg = psycopg2.connect(**PG_CONFIG)
    cur_pg = conn_pg.cursor()

    # 4. Bắt đầu Loop
    # Lấy các trường cần thiết (map từ schema sqlite của bạn)
    # SQLite: content, url, published_date, label
    cur_sqlite.execute("SELECT content, url, published_date, label FROM articles")
    
    batch_size = 100
    batch_data = []
    
    # Dùng tqdm để hiện thanh %
    for row in tqdm(cur_sqlite, total=total_rows, desc="Processing"):
        content, url, date, label = row
        
        if not content or len(content.strip()) < 50:
            continue # Bỏ qua bài quá ngắn/rỗng

        try:
            # A. Trích xuất Fact
            facts = extractor.extract(content)
            fact_json = json.dumps(facts, ensure_ascii=False)
            
            # B. Vector hóa
            vector = sbert.encode(content).tolist()
            
            # C. Chuẩn bị data để insert
            # Lưu ý: Postgres schema của chúng ta có cột 'label' (1: Real, 0: Fake)
            # Dữ liệu của bạn 50/50, nên chúng ta import hết để làm giàu DB
            batch_data.append((content, url, date, fact_json, str(vector), label))

            # D. Batch Insert (Cứ đủ 100 bài thì ghi xuống DB 1 lần cho nhanh)
            if len(batch_data) >= batch_size:
                insert_batch(cur_pg, batch_data)
                conn_pg.commit()
                batch_data = [] # Reset batch

        except Exception as e:
            print(f"\n⚠️ Lỗi khi xử lý bài: {url} - {e}")
            continue

    # Insert nốt những bài còn sót lại trong batch cuối
    if batch_data:
        insert_batch(cur_pg, batch_data)
        conn_pg.commit()

    # 5. Dọn dẹp
    cur_sqlite.close()
    conn_sqlite.close()
    cur_pg.close()
    conn_pg.close()
    print("\n✅ HOÀN TẤT DI TRÚ! Database của bạn giờ đã cực mạnh.")

def insert_batch(cursor, data):
    sql = """
        INSERT INTO articles (content, source_url, publish_date, fact_metadata, embedding, label)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.executemany(sql, data)

if __name__ == "__main__":
    migrate()