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

def debug_vector_search(claim_text):
    print("\n" + "="*80)
    print(f"🔎 TRUY VẤN (QUERY): \"{claim_text}\"")
    print("="*80)
    
    # 1. Load Model Vector (Bi-Encoder)
    # Lưu ý: Model này phải KHỚP với model bạn dùng lúc nạp DB (bkai-foundation-models/vietnamese-bi-encoder)
    print("⏳ Đang mã hóa câu truy vấn...")
    model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    query_vector = model.encode(claim_text)
    
    # 2. Query DB
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Lấy top 10, hiển thị cả khoảng cách (Distance)
    # Toán tử <=> trong pgvector là Cosine Distance
    # Distance càng gần 0 càng giống, càng gần 1 càng khác
    sql = """
        SELECT content, (embedding <=> %s::vector) as distance
        FROM sentence_store
        ORDER BY distance ASC
        LIMIT 10;
    """
    
    cur.execute(sql, (query_vector.tolist(),))
    results = cur.fetchall()
    
    print(f"{'DISTANCE':<10} | {'NỘI DUNG TÌM ĐƯỢC TRONG DB'}")
    print("-" * 80)
    
    for content, dist in results:
        # Tô màu dựa trên độ tốt của kết quả
        mark = ""
        if dist < 0.30: mark = "🟢 TỐT"     # Rất khớp
        elif dist < 0.50: mark = "🟡 KHÁ"   # Khớp chủ đề/Paraphrase
        else: mark = "🔴 KÉM"             # Không liên quan lắm
        
        # Cắt ngắn nội dung hiển thị
        display_content = (content[:90] + '...') if len(content) > 90 else content
        print(f"{dist:.4f}     | {mark} {display_content}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    # --- TEST CASE 1: Paraphrase (Từ đồng nghĩa) ---
    # DB có: "Thổ Nhĩ Kỳ điều 5 phi cơ..."
    # Query: dùng từ "máy bay", số lượng sai "500"
    debug_vector_search("Thổ Nhĩ Kỳ điều 500 máy bay sơ tán công dân")

    # --- TEST CASE 2: Sai lệch số liệu & Ngày tháng ---
    # DB có: "V-League 2024-2025 sẽ khai mạc từ ngày 23/8..."
    # Query: Khai mạc tháng 12
    debug_vector_search("V-League 2024-2025 dự kiến khai mạc vào tháng 12 năm nay")
    
    # --- TEST CASE 3: Rất khó (Nội dung fake hoàn toàn) ---
    # Query: Một tin bịa đặt không có trong DB
    debug_vector_search("Người ngoài hành tinh đổ bộ xuống Hồ Gươm")