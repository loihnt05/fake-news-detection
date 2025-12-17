import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# 1. Cấu hình kết nối (Khớp với docker-compose và .env)
DB_CONFIG = {
    "dbname": "vnexpress_scraper",
    "user": "vnexpress",
    "password": "admin123",
    "host": "localhost",
    "port": "5432"
}

def import_raw_data(file_path):
    print(f"⏳ Đang đọc file: {file_path}...")
    
    # 2. Đọc file dữ liệu
    df = pd.read_csv(file_path)

    print(f"✅ Đã đọc {len(df)} dòng dữ liệu.")
    
    # CSV có cột: id, url, title, description, content, scraped_at, published_date, label, category
    # Database cần: id, url, title, content, scraped_at, published_date, label, category
    # Bỏ cột 'description' vì database không có
    
    # Chỉ lấy các cột khớp với database
    required_cols = ['id', 'url', 'title', 'content', 'scraped_at', 'published_date', 'label', 'category']
    cols_to_import = [c for c in required_cols if c in df.columns]
    df = df[cols_to_import]
    
    print(f"📊 Các cột sẽ import: {cols_to_import}")

    print("⏳ Đang đẩy dữ liệu vào Database (có thể mất vài phút)...")
    
    # 3. Kết nối trực tiếp với psycopg2 (ổn định hơn)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Chuyển DataFrame thành list of tuples
    data = [tuple(row) for row in df.values]
    
    # Insert hàng loạt với execute_values (nhanh nhất)
    insert_query = f"""
        INSERT INTO articles ({', '.join(cols_to_import)}) 
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    
    # Insert từng batch 1000 dòng
    batch_size = 1000
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        execute_values(cur, insert_query, batch)
        conn.commit()
        current_batch = (i // batch_size) + 1
        print(f"  ✓ Đã import batch {current_batch}/{total_batches} ({len(batch)} dòng)")
    
    cur.close()
    conn.close()
    
    print("🎉 Thành công! Đã import toàn bộ dữ liệu vào bảng 'articles'.")

if __name__ == "__main__":
    # --- ĐÃ CẬP NHẬT FILE PATH ---
    MY_FILE = "/home/loiancut/workspace/fake-news-detection/dataset/articles_clean.csv" 
    
    import_raw_data(MY_FILE)