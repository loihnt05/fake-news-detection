import psycopg2
from sentence_transformers import SentenceTransformer
from underthesea import sent_tokenize
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def setup_database():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print("🛠️ Đang tạo bảng lưu trữ Sentence (Claim)...")
    # Bảng này chỉ chứa các câu từ bài báo REAL (Label = 1)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentence_store (
            id SERIAL PRIMARY KEY,
            article_id TEXT,
            content TEXT,
            embedding vector(768)
        );
        CREATE INDEX IF NOT EXISTS sent_embed_idx ON sentence_store USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()
    
    # Load Model Embedding (Bi-Encoder)
    print("⏳ Loading Embedding Model...")
    embed_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    
    # Lấy dữ liệu bài REAL
    print("📥 Đang đọc các bài báo REAL từ DB...")
    # CHỈ LẤY LABEL = 1 (Sự thật)
    cur.execute("SELECT id, content FROM articles WHERE label = 1 AND content IS NOT NULL")
    articles = cur.fetchall()
    
    print(f"⚙️ Bắt đầu xử lý {len(articles)} bài báo...")
    
    # Batch processing để tăng tốc
    batch_data = []
    batch_size = 100 
    
    for art_id, content in tqdm(articles):
        # 1. Sentence Segmentation
        sentences = sent_tokenize(content)
        
        for sent in sentences:
            # 2. Filter (Sơ khai của Claim Extraction)
            # Chỉ lấy câu đủ dài, bỏ qua câu ngắn vô nghĩa
            if len(sent.split()) < 5: continue
            
            batch_data.append((art_id, sent))
            
            if len(batch_data) >= batch_size:
                # 3. Claim Embedding (Batch)
                texts = [b[1] for b in batch_data]
                embeddings = embed_model.encode(texts)
                
                # Insert vào DB
                args = []
                for (aid, txt), emb in zip(batch_data, embeddings):
                    cur.execute(
                        "INSERT INTO sentence_store (article_id, content, embedding) VALUES (%s, %s, %s)",
                        (aid, txt, emb.tolist())
                    )
                batch_data = [] # Reset batch
                conn.commit()

    # Xử lý nốt phần dư
    if batch_data:
        texts = [b[1] for b in batch_data]
        embeddings = embed_model.encode(texts)
        for (aid, txt), emb in zip(batch_data, embeddings):
            cur.execute("INSERT INTO sentence_store (article_id, content, embedding) VALUES (%s, %s, %s)", (aid, txt, emb.tolist()))
        conn.commit()

    cur.close()
    conn.close()
    print("✅ Đã xây dựng xong kho dữ liệu Sentence Vector!")

if __name__ == "__main__":
    setup_database()