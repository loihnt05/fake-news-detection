import psycopg2
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import sent_tokenize
import os
from dotenv import load_dotenv
from tqdm import tqdm
import torch

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def migrate_data_smart():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running on device: {device}")

    # 1. Load Model Lọc Claim (Dùng thư viện Transformers gốc)
    print("⏳ Đang tải Claim Detector Model (HuggingFace Native)...")
    model_path = "./claim_detector_model"
    
    # Load Tokenizer & Model từ folder đã train
    claim_tokenizer = AutoTokenizer.from_pretrained(model_path)
    claim_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    claim_model.to(device)
    claim_model.eval() # Chuyển sang chế độ dự đoán (không train)
    
    # 2. Load Model Vector
    print("⏳ Đang tải Embedding Model...")
    embed_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Lấy bài viết REAL
    print("🔌 Đang truy vấn bài REAL...")
    cur.execute("SELECT id, content FROM articles WHERE label = '1' AND content IS NOT NULL")
    articles = cur.fetchall()
    
    BATCH_SIZE = 32
    batch_sentences = []
    batch_meta = []
    
    print(f"⚙️ Bắt đầu xử lý {len(articles)} bài báo (CHẾ ĐỘ AI FILTER)...")
    
    # Hàm dự đoán nhanh (Batch Inference)
    def predict_batch(texts):
        # Tokenize batch
        inputs = claim_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = claim_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
        return preds.cpu().numpy()

    for art_id, content in tqdm(articles):
        # Tách câu
        sentences = sent_tokenize(content)
        if not sentences: continue

        # --- AI FILTERING ---
        # Dự đoán claim hay non-claim
        labels = predict_batch(sentences)
        
        # Chỉ giữ lại câu mà Model bảo là Claim (Label = 1)
        valid_sentences = []
        for sent, label in zip(sentences, labels):
            if label == 1:
                valid_sentences.append(sent)
        
        if not valid_sentences: continue
        
        # Gom batch để Vector hóa
        for sent in valid_sentences:
            batch_sentences.append(sent)
            batch_meta.append(art_id)
            
            if len(batch_sentences) >= BATCH_SIZE:
                # Vector hóa
                embeddings = embed_model.encode(batch_sentences, show_progress_bar=False)
                
                # Insert DB
                args = [(mid, txt, emb.tolist()) for mid, txt, emb in zip(batch_meta, batch_sentences, embeddings)]
                cur.executemany(
                    "INSERT INTO sentence_store (article_id, content, embedding) VALUES (%s, %s, %s)",
                    args
                )
                conn.commit()
                batch_sentences = []
                batch_meta = []

    # Xử lý phần dư cuối cùng
    if batch_sentences:
        embeddings = embed_model.encode(batch_sentences, show_progress_bar=False)
        args = [(mid, txt, emb.tolist()) for mid, txt, emb in zip(batch_meta, batch_sentences, embeddings)]
        cur.executemany(
            "INSERT INTO sentence_store (article_id, content, embedding) VALUES (%s, %s, %s)",
            args
        )
        conn.commit()

    print("✅ Xong! Database giờ chỉ toàn 'Chất' (Claim xịn).")
    cur.close()
    conn.close()

if __name__ == "__main__":
    migrate_data_smart()