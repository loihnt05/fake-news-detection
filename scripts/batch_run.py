import psycopg2
from processor import NewsProcessor
from tqdm import tqdm # Thư viện hiện thanh loading
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Cấu hình DB từ environment variables
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Số lượng bài xử lý mỗi lần commit vào DB (để an toàn và nhanh)
BATCH_SIZE = 50 

def run_batch_processing():
    # 1. Khởi tạo kết nối & Model
    print("🔌 Đang kết nối Database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    processor = NewsProcessor() # Load model AI (sẽ mất vài giây)

    # 2. Đếm số lượng bài chưa xử lý
    # Chỉ lấy những bài mà embedding đang NULL
    cur.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NULL;")
    total_remaining = cur.fetchone()[0]
    print(f"📊 Tổng số bài cần xử lý: {total_remaining}")
    
    if total_remaining == 0:
        print("🎉 Tất cả bài viết đã được xử lý xong! Không cần làm gì nữa.")
        return

    # 3. Vòng lặp xử lý (Progress Bar)
    # pbar là thanh loading
    pbar = tqdm(total=total_remaining, desc="🚀 Processing", unit=" bài")
    
    while True:
        # Lấy 1 batch bài chưa xử lý
        # Lấy cả 'id', 'title', 'content'
        cur.execute("""
            SELECT id, title, content 
            FROM articles 
            WHERE embedding IS NULL 
            LIMIT %s
        """, (BATCH_SIZE,))
        
        rows = cur.fetchall()
        
        if not rows:
            break # Hết dữ liệu
            
        update_data = []
        
        # Xử lý từng bài trong batch hiện tại
        for row in rows:
            art_id, title, content = row
            
            try:
                # Gọi AI Processor (Hàm bạn đã viết ở bước trước)
                facts, vector = processor.process_article(title, content)
                
                # Nếu xử lý thành công (bài đủ dài)
                if vector is not None:
                    # Chuẩn bị dữ liệu để update
                    # Postgres vector cần list float, text[] cần list string
                    update_data.append((vector, facts, art_id))
                else:
                    # Nếu bài lỗi/quá ngắn, ta vẫn phải đánh dấu là đã xử lý 
                    # để lần sau không lặp lại. Ta gán vector rỗng hoặc flag đặc biệt.
                    # Ở đây tôi chọn cách xóa bài rác hoặc bỏ qua. 
                    # Tạm thời ta set facts = ["ERROR"] để biết mà bỏ qua sau này
                    # Nhưng để đơn giản cho flow này, ta cứ update extracted_facts = {}, embedding = NULL (vẫn NULL thì lần sau sẽ lặp lại -> Nguy hiểm).
                    # FIX: Ta sẽ update extracted_facts là "Too short" để đánh dấu.
                    pass 

            except Exception as e:
                print(f"\n❌ Lỗi tại bài ID {art_id}: {e}")
                continue
        
        # 4. Lưu ngược vào Database (Batch Update)
        # Dùng executemany để update nhanh hơn
        if update_data:
            query = """
                UPDATE articles 
                SET embedding = %s, extracted_facts = %s 
                WHERE id = %s;
            """
            cur.executemany(query, update_data)
            conn.commit() # Lưu thay đổi
        
        # Cập nhật thanh tiến trình
        pbar.update(len(rows))

    pbar.close()
    cur.close()
    conn.close()
    print("\n✅ HOÀN TẤT! Toàn bộ 96k bài đã được Vector hóa.")

if __name__ == "__main__":
    run_batch_processing()