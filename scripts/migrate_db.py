import sqlite3
import psycopg2
from psycopg2.extras import Json, execute_values
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from dotenv import load_dotenv
import torch
import multiprocessing as mp
from fact_extractor import FactExtractor

# Load environment variables
load_dotenv()

# --- CẤU HÌNH ---
SQLITE_PATH = os.getenv("SQLITE_PATH", "data/news_dataset.db")
BATCH_SIZE = 200 # Tăng batch lên vì xử lý đa nhân nhanh hơn
MAX_TEXT_LENGTH = 2000 # CHỈ LẤY 2000 KÝ TỰ ĐẦU (Đủ cho 5W1H)

PG_CONFIG = {
    "dbname": os.getenv("AIRFLOW_DB", "airflow"),
    "user": os.getenv("AIRFLOW_USER", "airflow"),
    "password": os.getenv("AIRFLOW_PASSWORD", "airflow"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432"))
}

# --- HÀM CHO WORKER PROCESS (CHẠY TRÊN CÁC NHÂN CPU KHÁC) ---
extractor = None

def worker_init():
    """Hàm khởi tạo chạy 1 lần trên mỗi nhân CPU con"""
    global extractor
    # Khởi tạo extractor riêng cho từng process để tránh conflict
    extractor = FactExtractor()

def worker_task(text):
    """Hàm xử lý extract fact"""
    global extractor
    if not text: return {}
    try:
        # Chỉ xử lý 1000 ký tự đầu -> Tốc độ tăng gấp 4 lần so với 4000
        return extractor.extract(text[:MAX_TEXT_LENGTH])
    except:
        return {}

def migrate():
    print("🚀 Bắt đầu di trú dữ liệu: MULTI-CORE SUPER MODE...")

    if not os.path.exists(SQLITE_PATH):
        print(f"❌ Không tìm thấy file {SQLITE_PATH}")
        return

    # 1. Setup DB & Count
    conn_sqlite = sqlite3.connect(SQLITE_PATH)
    cur_sqlite = conn_sqlite.cursor()
    conn_pg = psycopg2.connect(**PG_CONFIG)
    cur_pg = conn_pg.cursor()

    cur_sqlite.execute("SELECT COUNT(*) FROM articles")
    total_rows = cur_sqlite.fetchone()[0]
    print(f"📦 Tổng số bài: {total_rows}")

    # 2. Setup GPU Model (Main Process giữ SBERT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⏳ Main Process: Load SBERT trên {device.upper()}...")
    sbert = SentenceTransformer('keepitreal/vietnamese-sbert', device=device)

    # 3. Setup CPU Pool (Các nhân con giữ FactExtractor)
    num_cores = mp.cpu_count()
    print(f"🔥 Khởi động {num_cores} nhân CPU để đào Fact song song...")
    pool = mp.Pool(processes=num_cores, initializer=worker_init)

    # 4. Loop
    cur_sqlite.execute("SELECT content, url, published_date, label FROM articles")
    insert_sql = """
        INSERT INTO articles (content, source_url, publish_date, fact_metadata, embedding, label)
        VALUES %s
    """

    pbar = tqdm(total=total_rows, desc="Migrating")

    while True:
        rows = cur_sqlite.fetchmany(BATCH_SIZE)
        if not rows: break

        # Lọc dữ liệu đầu vào
        clean_rows = []
        batch_content_for_vector = [] # Full hoặc cắt vừa phải cho vector
        batch_content_for_extract = [] # Cắt ngắn cho extractor

        for row in rows:
            content, url, date, label = row
            if content and len(content.strip()) >= 50:
                # Vector cần ngữ cảnh rộng hơn chút (khoảng 2000 ký tự là đẹp cho SBERT)
                vec_text = content[:2000]
                
                clean_rows.append(row)
                batch_content_for_vector.append(vec_text)
                batch_content_for_extract.append(content) # Worker sẽ tự cắt 1000
            else:
                pbar.update(1)

        if not clean_rows: continue

        try:
            # BƯỚC 1: SBERT (GPU) - Chạy trên Main Process
            # Encode cả cục
            vectors = sbert.encode(batch_content_for_vector, batch_size=BATCH_SIZE, show_progress_bar=False)

            # BƯỚC 2: FACT EXTRACTOR (Multi-Core CPU) - Chạy song song
            # map: Phân phối list text cho các nhân xử lý cùng lúc
            facts_list = pool.map(worker_task, batch_content_for_extract)

            # BƯỚC 3: GOM DỮ LIỆU
            final_values = []
            for i, row in enumerate(clean_rows):
                full_content, url, date, label = row
                
                final_values.append((
                    full_content,
                    url,
                    date,
                    Json(facts_list[i]),   # Kết quả từ Pool
                    vectors[i].tolist(),   # Kết quả từ GPU
                    label
                ))

            # BƯỚC 4: INSERT
            execute_values(cur_pg, insert_sql, final_values)
            conn_pg.commit()
            
            pbar.update(len(rows))

        except Exception as e:
            print(f"⚠️ Lỗi Batch: {e}")
            conn_pg.rollback()

    # Cleanup
    pool.close()
    pool.join()
    cur_sqlite.close()
    conn_sqlite.close()
    cur_pg.close()
    conn_pg.close()
    print("\n✅ HOÀN TẤT DI TRÚ!")

if __name__ == "__main__":
    # Windows bắt buộc phải có if __name__ == "__main__" để dùng multiprocessing
    mp.set_start_method('spawn', force=True) # An toàn cho CUDA + Multiprocessing
    migrate()