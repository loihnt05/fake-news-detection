import psycopg2
import torch
import os
import sys
from tqdm import tqdm
from underthesea import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- CẤU HÌNH ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

# ĐƯỜNG DẪN MODEL MỚI (Theo cấu trúc project hiện tại)
MODEL_CLAIM_PATH = "model/phobert_claim_extractor" 
MODEL_EMBED_PATH = "bkai-foundation-models/vietnamese-bi-encoder"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Xử lý 32 câu một lúc cho nhanh

class KnowledgeBaseRebuilder:
    def __init__(self):
        print(f"🚀 [Rebuilder] KNOWLEDGE BASE ĐANG ĐƯỢC XÂY DỰNG LẠI TRÊN {DEVICE.upper()}...")

        # 1. Load Model Lọc Claim (Người gác cổng)
        print("   ├─ [1/2] Loading Claim Detector...")
        try:
            self.claim_tokenizer = AutoTokenizer.from_pretrained(MODEL_CLAIM_PATH)
            self.claim_model = AutoModelForSequenceClassification.from_pretrained(MODEL_CLAIM_PATH)
            self.claim_model.to(DEVICE)
            self.claim_model.eval()
        except Exception as e:
            print(f"❌ Lỗi load model Claim: {e}")
            print(f"👉 Hãy chắc chắn bạn đã để model tại: {MODEL_CLAIM_PATH}")
            sys.exit(1)

        # 2. Load Model Vector (Người mã hóa)
        print("   ├─ [2/2] Loading Embedding Model...")
        self.embed_model = SentenceTransformer(MODEL_EMBED_PATH, device=DEVICE)
        
        # 3. Kết nối DB
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.conn.autocommit = True

    def get_raw_articles(self):
        """Lấy tất cả bài báo từ bảng articles"""
        with self.conn.cursor() as cur:
            # Lấy ID và Content của bài báo
            # (Không cần WHERE label=1 nữa vì ta mặc định nguồn cào về là tin cậy)
            cur.execute("SELECT id, content FROM articles WHERE label='1' and content IS NOT NULL")
            return cur.fetchall()

    def predict_batch(self, texts):
        """Dự đoán nhanh một lô câu hỏi (Batch Inference)"""
        inputs = self.claim_tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.claim_model(**inputs)
            # Lấy nhãn có xác suất cao nhất (0 hoặc 1)
            preds = torch.argmax(outputs.logits, dim=1)
        return preds.cpu().numpy()

    def run(self):
        # 1. DỌN DẸP DỮ LIỆU CŨ
        print("\n🧹 Đang dọn dẹp bảng 'claims' cũ...")
        with self.conn.cursor() as cur:
            # Xóa user_reports trước vì nó tham chiếu đến claims
            cur.execute("TRUNCATE TABLE user_reports CASCADE;")
            cur.execute("TRUNCATE TABLE claims CASCADE;")
        print("✅ Database đã sạch.")

        # 2. LẤY DỮ LIỆU NGUỒN
        articles = self.get_raw_articles()
        print(f"📦 Tìm thấy {len(articles)} bài báo gốc. Bắt đầu trích xuất...")

        total_claims_saved = 0
        
        # Biến tạm để gom batch vector hóa
        pending_insert = [] # List các tuple (article_id, content)

        # 3. VÒNG LẶP XỬ LÝ
        for art_id, content in tqdm(articles, desc="Processing"):
            # A. Tách câu
            sentences = sent_tokenize(content)
            # Lọc sơ bộ câu quá ngắn (< 5 từ)
            candidates = [s.strip() for s in sentences if len(s.split()) > 5]
            
            if not candidates: continue

            # B. AI Lọc (Batch Processing)
            # Chia nhỏ candidates thành các batch nhỏ hơn nếu quá nhiều câu
            for i in range(0, len(candidates), BATCH_SIZE):
                batch_text = candidates[i : i + BATCH_SIZE]
                
                # Model phán xét: 1=Claim, 0=Non-Claim
                labels = self.predict_batch(batch_text)
                
                # Chỉ lấy câu Label 1
                for text, label in zip(batch_text, labels):
                    if label == 1:
                        pending_insert.append((art_id, text))

            # C. Vector hóa & Lưu (Khi gom đủ lượng lớn hoặc hết bài)
            # Gom khoảng 64 câu rồi xử lý 1 lần cho tối ưu GPU
            if len(pending_insert) >= 64:
                self.flush_to_db(pending_insert)
                total_claims_saved += len(pending_insert)
                pending_insert = [] # Reset

        # Xử lý nốt phần còn dư
        if pending_insert:
            self.flush_to_db(pending_insert)
            total_claims_saved += len(pending_insert)

        print(f"\n🎉 HOÀN TẤT! Đã xây dựng Knowledge Base với {total_claims_saved} claims chất lượng.")
        self.conn.close()

    def flush_to_db(self, items):
        """Vector hóa và Insert vào DB"""
        if not items: return
        
        # Tách list tuple thành 2 list riêng
        art_ids = [x[0] for x in items]
        texts = [x[1] for x in items]
        
        # Vector hóa hàng loạt
        embeddings = self.embed_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
        
        # Insert Bulk
        with self.conn.cursor() as cur:
            # Chuẩn bị dữ liệu cho execute_values hoặc loop
            # Ở đây dùng loop đơn giản vì psycopg2 xử lý khá nhanh
            insert_args = []
            for mid, txt, emb in zip(art_ids, texts, embeddings):
                # QUAN TRỌNG: Gán nhãn REAL
                insert_args.append((mid, txt, emb.tolist(), 'REAL', True))
            
            # Sử dụng executemany để insert nhanh
            query = """
                INSERT INTO claims (article_id, content, embedding, system_label, verified, source_type)
                VALUES (%s, %s, %s, %s, %s, 'article')
            """
            cur.executemany(query, insert_args)

if __name__ == "__main__":
    rebuilder = KnowledgeBaseRebuilder()
    rebuilder.run()