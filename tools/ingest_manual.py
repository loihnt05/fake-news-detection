import psycopg2
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- CẤU HÌNH ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": "localhost",
    "port": "5432"
}

MODEL_NAME = 'bkai-foundation-models/vietnamese-bi-encoder'

def ingest_cancer_cure_claims():
    print("⏳ Đang tải model embedding...")
    model = SentenceTransformer(MODEL_NAME)
    
    # DANH SÁCH CÁC CLAIM "ĐỘC HẠI" TRÍCH XUẤT TỪ BÀI BÁO TRÊN
    # Ta tổng quát hóa để bắt được cả những bài tương tự khác
    fake_claims = [
        # Claim 1: Trực tiếp từ bài báo (Ung thư trực tràng)
        "Bệnh nhân ung thư trực tràng giai đoạn cuối hồi phục hoàn toàn nhờ tu luyện Pháp Luân Công mà không cần điều trị thêm.",
        
        # Claim 2: Tổng quát cho các loại ung thư
        "Luyện tập Pháp Luân Công có thể chữa khỏi ung thư di căn và kéo dài tuổi thọ vượt qua dự đoán của bác sĩ.",
        
        # Claim 3: Đánh vào tâm lý 'Bệnh viện trả về'
        "Khi bệnh viện trả về và y học bó tay, tu luyện tâm tính giúp tế bào ung thư tự tiêu biến.",
        
        # Claim 4: Phủ nhận phác đồ điều trị
        "Sức khỏe cải thiện tốt hơn so với trước khi mắc ung thư chỉ nhờ phương pháp tu luyện mà không cần thuốc.",
        "Niệm 9 chữ chân ngôn có thể chữa khỏi ung thư và COVID-19 mà không cần thuốc.",
        "Người bị mù hoàn toàn 100% hồi phục thị lực nhờ đọc sách Pháp Luân Công.",
        "Luyện Pháp Luân Công giúp khỏi ung thư máu và các bệnh nan y không cần dùng thuốc.",
        "Bác sĩ bó tay nhưng niệm thần chú và tu luyện thì khỏi bệnh thần kỳ."
        
    ]

    print(f"🚀 Đang xử lý {len(fake_claims)} luận điểm tin giả mới...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        embeddings = model.encode(fake_claims)
        
        for i, claim in enumerate(fake_claims):
            vector = embeddings[i].tolist()
            
            # Kiểm tra xem claim đã tồn tại chưa
            cur.execute("SELECT id FROM claims WHERE content = %s", (claim,))
            existing = cur.fetchone()
            
            if existing:
                # Update nếu đã tồn tại
                cur.execute("""
                    UPDATE claims 
                    SET embedding = %s, system_label = 'FAKE', verified = TRUE, 
                        source_type = 'dkn_manual', trust_score = 0.0
                    WHERE content = %s
                """, (vector, claim))
                print(f"   🔄 Đã cập nhật: {claim[:60]}...")
            else:
                # Insert mới nếu chưa tồn tại
                cur.execute("""
                    INSERT INTO claims (content, embedding, system_label, verified, source_type, trust_score)
                    VALUES (%s, %s, 'FAKE', TRUE, 'dkn_manual', 0.0)
                """, (claim, vector))
                print(f"   ✅ Đã nạp Blacklist: {claim[:60]}...")

        conn.commit()
        cur.close()
        conn.close()
        print("\n🎉 XONG! Đã cập nhật tường lửa ngữ nghĩa.")
        
    except Exception as e:
        print(f"❌ LỖI: {e}")

if __name__ == "__main__":
    ingest_cancer_cure_claims()