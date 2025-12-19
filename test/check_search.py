import psycopg2
from sentence_transformers import SentenceTransformer
import os
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

def semantic_search(query_text):
    print(f"🔎 Đang tìm kiếm cho câu: '{query_text}'")
    
    # 1. Load model để embed câu query
    model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    query_vector = model.encode(query_text).tolist()
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # 2. Tìm kiếm Vector (Cosine Similarity)
    # Lấy ra bài giống nhất (LIMIT 1)
    sql = """
    SELECT title, extracted_facts, 1 - (embedding <=> %s::vector) AS similarity
    FROM articles
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT 1;
    """
    
    cur.execute(sql, (query_vector, query_vector))
    result = cur.fetchone()
    
    if result:
        title, facts, score = result
        print("\n✅ KẾT QUẢ TÌM THẤY:")
        print(f"   - Tiêu đề gốc: {title}")
        print(f"   - Độ giống (Score): {score:.4f}")
        print("   - Các ý chính (Facts) trong DB:")
        for f in facts:
            print(f"     + {f}")
    else:
        print("❌ Không tìm thấy bài nào tương đồng.")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    # --- THỬ NGHIỆM ---
    # Bạn hãy nhập một câu KHÔNG GIỐNG HỆT tiêu đề, mà chỉ CÙNG Ý NGHĨA
    # Ví dụ: Bài gốc là "Giá xăng tăng mạnh", bạn tìm "Xăng dầu hôm nay đắt thế"
    test_query = "Trọng tài lần đầu trực tiếp thông báo quyết định của VAR qua loa trong trận Liverpool thua Tottenham 0-1 ở lượt đi bán kết Cup Liên đoàn hôm 8/1" 
    semantic_search(test_query)