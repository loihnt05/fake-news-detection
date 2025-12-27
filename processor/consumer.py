import json
import os
import sys
import time
import torch
import psycopg2
import numpy as np
from kafka import KafkaConsumer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import sent_tokenize
from dotenv import load_dotenv

load_dotenv()

# --- CẤU HÌNH ---
KAFKA_TOPIC = "raw_articles"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 # Tối ưu tốc độ xử lý hàng loạt

# Cấu hình DB
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

MODEL_EXTRACTOR_PATH = "model/phobert_claim_extractor"

class AIProcessor:
    def __init__(self):
        print(f"🚀 [Consumer] KHỞI ĐỘNG AI PROCESSOR TRÊN {DEVICE.upper()}...")
        
        # 1. Load Model Lọc Câu (Claim Extractor)
        print("   ├─ [1/3] Loading Claim Extractor (PhoBERT)...")
        try:
            self.ext_tokenizer = AutoTokenizer.from_pretrained(MODEL_EXTRACTOR_PATH)
            self.ext_model = AutoModelForSequenceClassification.from_pretrained(MODEL_EXTRACTOR_PATH).to(DEVICE)
            self.ext_model.eval()
        except Exception as e:
            # RAISE ERROR để Docker biết mà restart, không exit() âm thầm
            raise RuntimeError(f"❌ Lỗi load model extractor: {e}. Hãy kiểm tra path '{MODEL_EXTRACTOR_PATH}'")

        # 2. Load Model Embedding (Bi-Encoder)
        print("   ├─ [2/3] Loading Embedding Model (Bi-Encoder)...")
        try:
            self.embedder = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=DEVICE)
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi load model embedding: {e}")

        # 3. Kết nối DB
        print("   ├─ [3/3] Connecting to PostgreSQL...")
        self.connect_db()
        print("✅ HỆ THỐNG SẴN SÀNG XỬ LÝ!")

    def connect_db(self):
        """Hàm kết nối DB có khả năng reconnect"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.conn.autocommit = True # Tự động commit để tránh lock lâu
        except Exception as e:
            raise ConnectionError(f"❌ Không thể kết nối DB: {e}")

    def extract_claims(self, text):
        """Tách câu và dùng AI lọc ra những câu đáng check"""
        if not text: return []
        
        # Bước 1: Tách câu (Heuristic)
        sentences = sent_tokenize(text)
        # Filter sơ bộ: Câu > 5 từ
        candidates = [s.strip() for s in sentences if len(s.split()) > 5]
        
        if not candidates: return []

        # Bước 2: Chạy qua Model Extractor (AI Classifier)
        # Tokenize batch
        inputs = self.ext_tokenizer(
            candidates, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.ext_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
        
        # Chỉ lấy câu có nhãn 1 (CLAIM)
        return [candidates[i] for i, pred in enumerate(preds) if pred == 1]

    def process_message(self, article):
        """Xử lý 1 bài báo (Pipeline: Article -> Claims -> Embeddings -> DB)"""
        url = article.get('url')
        title = article.get('title')
        content = article.get('content')
        published_at = article.get('published_date')
        category = article.get('category', 'General')

        # Sử dụng Context Manager cho Cursor (Best Practice)
        with self.conn.cursor() as cur:
            # 1. Lưu Article (Ingestion)
            try:
                # Kiểm tra xem bài đã tồn tại chưa
                cur.execute("SELECT id FROM articles WHERE url = %s", (url,))
                existing = cur.fetchone()
                
                if existing:
                    article_id = existing[0]
                    # Update nội dung nếu crawl lại
                    cur.execute("""
                        UPDATE articles 
                        SET content = %s, scraped_at = NOW(), published_date = %s, category = %s
                        WHERE id = %s
                    """, (content, published_at, category, article_id))
                else:
                    # Insert bài mới
                    cur.execute("""
                        INSERT INTO articles (url, title, content, published_date, category, scraped_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        RETURNING id;
                    """, (url, title, content, published_at, category))
                    article_id = cur.fetchone()[0]

            except Exception as e:
                print(f"   ❌ Lỗi DB Article: {e}")
                return # Bỏ qua bài này nếu lỗi DB

            # 2. Feature Extraction (Claim Extraction)
            claims = self.extract_claims(content)
            if not claims:
                print(f"   ℹ️ Không tìm thấy claim: {title[:40]}...")
                return

            # 3. Vectorization (Embedding) - Batch Processing
            print(f"   🔍 Vector hóa {len(claims)} claims...")
            embeddings = self.embedder.encode(
                claims, 
                batch_size=BATCH_SIZE, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # 4. Storage (Lưu Claims với nhãn UNDEFINED)
            count = 0
            for text, emb in zip(claims, embeddings):
                try:
                    # Lưu vector dạng list (pgvector tự hiểu)
                    cur.execute("""
                        INSERT INTO claims (article_id, content, embedding, system_label, verified, source_type)
                        VALUES (%s, %s, %s, 'REAL', TRUE, 'article') 
                    """, (article_id, text, emb.tolist()))
                    count += 1
                except Exception as e:
                    print(f"      ❌ Lỗi lưu claim con: {e}")

            if count > 0:
                print(f"   ✅ [Processed] {title[:40]}... -> {count} Claims lưu DB.")

    def start_consuming(self):
        print(f"\n📡 [Consumer] ĐANG LẮNG NGHE TOPIC '{KAFKA_TOPIC}'...")
        
        while True:
            try:
                consumer = KafkaConsumer(
                    KAFKA_TOPIC,
                    bootstrap_servers=[KAFKA_SERVER],
                    auto_offset_reset='earliest', # Đọc từ đầu nếu là group mới
                    enable_auto_commit=True,
                    group_id='ai-processor-group-v2', # ✨ Đổi version để đọc lại từ đầu
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    consumer_timeout_ms=10000,  # Timeout 10s để có thể show heartbeat
                    # Tối ưu fetch
                    fetch_min_bytes=1,  # Giảm xuống để nhận ngay khi có message
                    fetch_max_wait_ms=500
                )
                
                print("✅ [Consumer] Kafka Connected!")
                print(f"   Consumer Group: ai-processor-group-v2")
                print(f"   Auto Offset Reset: earliest\n")
                
                message_count = 0
                heartbeat_count = 0
                
                while True:
                    try:
                        for message in consumer:
                            message_count += 1
                            print(f"📥 [{message_count}] Nhận bài từ offset {message.offset}: {message.value.get('title', 'N/A')[:50]}...")
                            self.process_message(message.value)
                    except StopIteration:
                        # Timeout - no messages received
                        heartbeat_count += 1
                        print(f"💓 [Heartbeat #{heartbeat_count}] Đang chờ tin nhắn mới... (Đã xử lý: {message_count} bài)")
                        continue

            except Exception as e:
                print(f"❌ [Consumer] Lỗi kết nối Kafka: {e}")
                print("⏳ Thử lại sau 5s...")
                time.sleep(5)

if __name__ == "__main__":
    # Đảm bảo DB sẵn sàng trước khi chạy
    # (Trong Production sẽ dùng healthcheck container)
    
    # NOTE: Nếu consumer đã chạy trước đó và đã đọc hết messages, 
    # nó sẽ tiếp tục từ offset cũ. Để đọc lại từ đầu:
    # 1. Đổi group_id trong code (VD: 'ai-processor-group-v2')
    # 2. Hoặc reset offset: kafka-consumer-groups --bootstrap-server localhost:9092 --group ai-processor-group-v1 --reset-offsets --to-earliest --execute --topic raw_articles
    
    processor = AIProcessor()
    processor.start_consuming()