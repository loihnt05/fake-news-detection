import psycopg2
from processor import NewsProcessor
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

BATCH_SIZE = 50 

def run_batch_processing():
    print("🔌 Đang kết nối Database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    processor = NewsProcessor()

    # --- FIX 1: Sửa điều kiện đếm ---
    # Chỉ đếm những bài chưa có vector VÀ chưa có extracted_facts (nghĩa là chưa đụng tới)
    cur.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NULL AND extracted_facts IS NULL;")
    total_remaining = cur.fetchone()[0]
    print(f"📊 Tổng số bài thực sự cần xử lý: {total_remaining}")
    
    if total_remaining == 0:
        print("🎉 Tất cả bài viết đã được xử lý xong!")
        return

    pbar = tqdm(total=total_remaining, desc="🚀 Processing", unit=" bài")
    
    while True:
        # --- FIX 2: Sửa câu Query lấy dữ liệu ---
        # Tránh lấy lại những bài đã bị đánh dấu là SKIPPED/Lỗi
        cur.execute("""
            SELECT id, title, content 
            FROM articles 
            WHERE embedding IS NULL AND extracted_facts IS NULL
            LIMIT %s
        """, (BATCH_SIZE,))
        
        rows = cur.fetchall()
        if not rows:
            break
            
        success_data = [] # List chứa bài thành công
        skipped_data = [] # List chứa bài lỗi (để đánh dấu bỏ qua)
        
        for row in rows:
            art_id, title, content = row
            
            try:
                facts, vector = processor.process_article(title, content)
                
                if vector is not None:
                    # Thành công -> Update cả Vector và Facts
                    success_data.append((vector, facts, art_id))
                else:
                    # --- FIX 3: Xử lý bài lỗi ---
                    # Bài quá ngắn/lỗi -> Update facts là 'SKIPPED' để lần sau không lấy lại nữa
                    skipped_data.append((['SKIPPED_TOO_SHORT'], art_id))

            except Exception as e:
                print(f"\n❌ Exception ID {art_id}: {e}")
                # Nếu crash code python thì cũng đánh dấu skip luôn
                skipped_data.append((['ERROR_EXCEPTION'], art_id))
                continue
        
        # Update Batch thành công
        if success_data:
            query_success = """
                UPDATE articles 
                SET embedding = %s, extracted_facts = %s 
                WHERE id = %s;
            """
            cur.executemany(query_success, success_data)

        # Update Batch lỗi (Quan trọng để phá vòng lặp)
        if skipped_data:
            query_skip = """
                UPDATE articles 
                SET extracted_facts = %s 
                WHERE id = %s;
            """
            cur.executemany(query_skip, skipped_data)

        conn.commit()
        pbar.update(len(rows))

    pbar.close()
    cur.close()
    conn.close()
    print("\n✅ HOÀN TẤT! Đã xử lý sạch sẽ cả bài lỗi.")

if __name__ == "__main__":
    run_batch_processing()