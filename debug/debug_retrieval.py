import psycopg2
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Cấu hình DB
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def debug_retrieval():
    print("🚀 Đang khởi động Debug Retrieval...")
    
    # ⚠️ QUAN TRỌNG: Model này PHẢI GIỐNG HỆT model bạn dùng trong file import_to_db.py
    model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
    print(f"   Model đang dùng: {model_name}")
    retriever = SentenceTransformer(model_name)
    
    # Các câu query bạn đang bị lỗi
    queries = [
        "Thổ Nhĩ Kỳ điều 500 máy bay sơ tán công dân", # Case số liệu
        "V-League 2024-2025 dự kiến khai mạc vào tháng 12 năm nay", # Case ngày tháng
        "Người ngoài hành tinh đổ bộ xuống Hồ Gươm" # Case tin bịa hoàn toàn
    ]
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    for query in queries:
        print("\n" + "="*80)
        print(f"🔎 TRUY VẤN (QUERY): \"{query}\"")
        print("="*80)
        
        # Mã hóa
        print("⏳ Đang mã hóa câu truy vấn...")
        emb = retriever.encode(query)
        
        # Tìm kiếm thô (Không WHERE distance, lấy thẳng top 10)
        cur.execute("""
            SELECT content, (embedding <=> %s::vector) as distance
            FROM sentence_store
            ORDER BY distance ASC
            LIMIT 10; 
        """, (emb.tolist(),))
        
        results = cur.fetchall()
        
        print(f"{'DISTANCE':<10} | {'NỘI DUNG TÌM ĐƯỢC TRONG DB':<80}")
        print("-" * 95)
        
        for content, dist in results:
            # Đánh giá sơ bộ
            grade = "🟢 TỐT" if dist < 0.4 else ("🟡 KHÁ" if dist < 0.6 else "🔴 KÉM")
            print(f"{dist:.4f}     | {grade} {content[:90]}...")

    cur.close()
    conn.close()

if __name__ == "__main__":
    debug_retrieval()