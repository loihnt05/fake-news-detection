import psycopg2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- 1. Load biến môi trường từ file .env ---
load_dotenv() 

# --- 2. Cấu hình DB lấy từ .env ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    # Nếu chạy script từ máy host (ngoài docker) thì là localhost
    # Nếu chạy script từ trong 1 container khác cùng network thì đổi thành tên service (vd: db)
    "host": os.getenv("DB_HOST", "localhost"), 
    "port": os.getenv("DB_PORT", "5432")
}

# Kiểm tra xem đã đọc được config chưa (để debug)
if not DB_CONFIG["dbname"]:
    print("LỖI: Không đọc được file .env. Hãy chắc chắn file .env nằm cùng thư mục với script này.")
    exit(1)

# --- 3. Load model AI ---
print("Đang tải model AI...")
# Lưu ý: Model này sẽ tải về cache máy tính lần đầu tiên
model = SentenceTransformer('keepitreal/vietnamese-sbert') 

def update_embeddings():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Lỗi kết nối Database: {e}")
        return

    # Lấy những dòng chưa có vector
    print("Đang lấy dữ liệu trống vector...")
    cursor.execute("SELECT id, title, content FROM articles WHERE embedding IS NULL")
    rows = cursor.fetchall()
    
    total = len(rows)
    print(f"Tìm thấy {total} dòng cần xử lý.")
    
    if total == 0:
        cursor.close()
        conn.close()
        return

    batch_size = 128 # Có thể tăng lên 128 hoặc 256 nếu VRAM/RAM khỏe
    
    for i in tqdm(range(0, total, batch_size)):
        batch_rows = rows[i:i+batch_size]
        batch_texts = []
        batch_ids = []
        
        # Gộp text để tạo ngữ cảnh đầy đủ
        for row in batch_rows:
            _id, title, content = row
            # Xử lý null để tránh lỗi cộng chuỗi
            t = title if title else ""
            c = content if content else ""
            
            # Cắt ngắn bớt nếu quá dài (SBERT thường chỉ xử lý tốt < 256-512 tokens, 
            # nhưng cắt 2000 ký tự là ngưỡng an toàn cho tốc độ)
            full_text = f"{t}. {c}"[:2000] 
            batch_texts.append(full_text)
            batch_ids.append(_id)
            
        # Tạo vector (Batch processing)
        if batch_texts:
            embeddings = model.encode(batch_texts)
            
            # Update ngược vào DB
            for _id, emb in zip(batch_ids, embeddings):
                cursor.execute(
                    "UPDATE articles SET embedding = %s WHERE id = %s",
                    (str(emb.tolist()), _id)
                )
            
            conn.commit() # Commit sau mỗi batch

    cursor.close()
    conn.close()
    print("Hoàn tất cập nhật Vector!")

if __name__ == "__main__":
    update_embeddings()