from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()
# Cấu hình kết nối từ environment variables (như trong file docker-compose)
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": "localhost",
    "port": "5432"
}

try:
    # 1. Kết nối
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    
    print("✅ Kết nối Database thành công!")

    # 2. Bật Extension Vector (Bắt buộc cho AI)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("✅ Đã bật Extension pgvector.")

    # 3. Tạo bảng articles với cấu trúc chuẩn
    # Lưu ý: embedding vector(768) khớp với model PhoBERT/Bi-Encoder
    create_table_query = """
    CREATE TABLE IF NOT EXISTS "articles"(
        "id" TEXT, 
        "url" TEXT, 
        "title" TEXT, 
        "content" TEXT, 
        "scraped_at" TEXT, 
        "published_date" TEXT, 
        "label" TEXT,
        "category" TEXT,
        extracted_facts TEXT[],
        embedding vector(768)
    );
    """
    cur.execute(create_table_query)
    
    # 3.5. Thêm các cột còn thiếu nếu bảng đã tồn tại
    cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS extracted_facts TEXT[];")
    
    # 4. Tạo Index tìm kiếm nhanh (HNSW)
    # Giúp tìm trong 96k bài cực nhanh
    cur.execute("""
        CREATE INDEX IF NOT EXISTS articles_embedding_idx 
        ON articles USING hnsw (embedding vector_cosine_ops);
    """)
    
    print("✅ Đã tạo bảng 'articles' và Index thành công!")

    cur.close()
    conn.close()

except Exception as e:
    print("❌ Có lỗi xảy ra:", e)