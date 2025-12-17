import sys
import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.processor import NewsProcessor

# Load environment variables
load_dotenv()

# Config DB from environment
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "vnexpress"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin123"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def test_one_article():
    # 1. Kết nối DB lấy 1 bài ngẫu nhiên chưa được xử lý
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Lấy 1 bài mà cột embedding đang NULL
    cur.execute("SELECT id, title, content FROM articles WHERE embedding IS NULL LIMIT 1;")
    row = cur.fetchone()
    
    if not row:
        print("⚠️ Không tìm thấy bài nào chưa xử lý (hoặc DB rỗng). Hãy chạy import_data.py trước!")
        return

    article_id, title, content = row
    print(f"📝 Đang test bài ID: {article_id}")
    print(f"📌 Title: {title}")

    # 2. Gọi AI Processor xử lý
    processor = NewsProcessor()
    facts, vector = processor.process_article(title, content)

    if vector:
        print(f"\n✅ Đã tạo Vector 768 chiều (Sample: {vector[:3]}...)")
        print("\n✅ Đã trích xuất các ý chính (Facts):")
        for f in facts:
            print(f"  - {f}")
            
        # 3. (Optional) Thử Update lại vào DB xem có lỗi không
        print("\n⏳ Đang thử lưu vào DB...")
        cur.execute("""
            UPDATE articles 
            SET embedding = %s, extracted_facts = %s 
            WHERE id = %s
        """, (vector, facts, article_id))
        conn.commit()
        print("🎉 Lưu thành công! Hệ thống đã sẵn sàng chạy Batch.")
    else:
        print("❌ Lỗi xử lý bài báo (Nội dung quá ngắn hoặc rỗng).")

    cur.close()
    conn.close()

if __name__ == "__main__":
    test_one_article()