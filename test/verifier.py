import psycopg2
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from underthesea import sent_tokenize
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

class FactCheckerPipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing Pipeline on {self.device}...")
        
        # [Step 3] Model Embedding (Bi-Encoder) - Dùng để tìm kiếm
        print("   ├─ Loading Retriever (Bi-Encoder)...")
        self.retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=self.device)
        
        # [Step 5] Model Verification (NLI) - Model bạn train trên Colab
        print("   ├─ Loading Verifier (NLI Model)...")
        model_path = './my_model' 
        if not os.path.exists(model_path):
            raise Exception("❌ Thiếu model! Hãy tải model train từ Colab về folder './my_model'")
        self.nli_model = CrossEncoder(model_path, device=self.device)
        
        print("✅ Pipeline Ready!")

    def _get_db_connection(self):
        return psycopg2.connect(**DB_CONFIG)

    def run(self, article_text):
        """Hàm chạy toàn bộ quy trình kiểm tra"""
        
        # --- [Step 1 & 2] Segmentation & Extraction ---
        print("\n1️⃣ Tách câu & Trích xuất Claim...")
        raw_sentences = sent_tokenize(article_text)
        # Chỉ lấy câu có độ dài > 5 từ (Coi là Claim)
        claims = [s for s in raw_sentences if len(s.split()) > 5]
        print(f"   -> Tìm thấy {len(claims)} claims quan trọng.")
        
        if not claims:
            return {"status": "ERROR", "reason": "Bài viết quá ngắn hoặc không có thông tin."}

        # --- [Step 3 & 4] Embedding & Retrieval ---
        print("2️⃣ Tìm kiếm bằng chứng (Evidence Retrieval)...")
        claim_vectors = self.retriever.encode(claims)
        
        conn = self._get_db_connection()
        cur = conn.cursor()
        
        verified_claims = []
        
        for i, claim in enumerate(claims):
            # Tìm 3 câu trong kho dữ liệu REAL giống nhất với claim này
            query = """
                SELECT content, (embedding <=> %s::vector) as distance
                FROM sentence_store
                ORDER BY distance ASC
                LIMIT 3;
            """
            cur.execute(query, (claim_vectors[i].tolist(),))
            results = cur.fetchall()
            
            # Lọc bằng chứng: Chỉ lấy nếu distance < 0.4 (tức là có liên quan về mặt ngữ nghĩa)
            valid_evidence = [row[0] for row in results if row[1] < 0.4]
            
            # --- [Step 5] Verification (NLI) ---
            # Nếu không tìm thấy bằng chứng nào trong kho dữ liệu Real -> NEI (Not Enough Info)
            if not valid_evidence:
                verified_claims.append({
                    "claim": claim,
                    "evidence": None,
                    "status": "NEUTRAL", # Không thể kiểm chứng
                    "score": 0.5
                })
                continue
            
            # Ghép cặp Claim với từng Evidence để AI chấm điểm
            pairs = [[ev, claim] for ev in valid_evidence]
            scores = self.nli_model.predict(pairs)
            
            # Lấy bằng chứng có điểm cao nhất (tức là khớp nhất hoặc mâu thuẫn nhất)
            # Vì model train: 1=True, 0=Fake
            # Nếu điểm rất cao (>0.7) -> Evidence ỦNG HỘ Claim -> TRUE
            # Nếu điểm rất thấp (<0.3) -> Evidence MÂU THUẪN Claim -> FAKE
            
            best_idx = np.argmax(scores) # Vị trí của điểm cao nhất chưa chắc tốt nếu tất cả đều thấp
            # Nhưng với logic của CrossEncoder 1 output:
            # Ta cần xem xét giá trị score cụ thể
            
            # Lấy score cực trị (quan tâm nhất là nó Rất Đúng hoặc Rất Sai)
            max_score = np.max(scores)
            min_score = np.min(scores)
            
            final_status = "NEUTRAL"
            final_score = 0.5
            best_ev = valid_evidence[0] # Mặc định
            
            # Ưu tiên bắt lỗi Fake (nếu có 1 bằng chứng mâu thuẫn mạnh -> FAKE)
            if min_score < 0.2: 
                final_status = "REFUTED" # Fake
                final_score = float(min_score)
                best_ev = valid_evidence[np.argmin(scores)]
            elif max_score > 0.7:
                final_status = "SUPPORTED" # True
                final_score = float(max_score)
                best_ev = valid_evidence[np.argmax(scores)]
            else:
                final_status = "NEUTRAL" # Mơ hồ
                final_score = float(max_score)
            
            verified_claims.append({
                "claim": claim,
                "evidence": best_ev,
                "status": final_status,
                "score": final_score
            })
            
        cur.close()
        conn.close()

        # --- [Step 6 & 7] Aggregation & Classification ---
        print("3️⃣ Tổng hợp & Kết luận...")
        
        # Đếm số lượng
        n_refuted = sum(1 for c in verified_claims if c['status'] == 'REFUTED')
        n_supported = sum(1 for c in verified_claims if c['status'] == 'SUPPORTED')
        total = len(verified_claims)
        
        final_label = "NEUTRAL"
        explanation = "Không đủ dữ liệu để xác thực."
        confidence = 0.0
        
        if n_refuted > 0:
            # Chỉ cần 1 câu nói láo -> Cả bài FAKE (nguyên tắc nghiêm ngặt)
            final_label = "FAKE"
            explanation = f"Phát hiện {n_refuted} thông tin sai lệch so với cơ sở dữ liệu."
            # Lấy độ tin cậy từ các câu bị refute
            confidence = 1 - (sum(c['score'] for c in verified_claims if c['status'] == 'REFUTED') / n_refuted)
            
        elif n_supported > (total * 0.5): # Hơn 50% câu được xác thực đúng
            final_label = "REAL"
            explanation = f"Xác thực được {n_supported}/{total} thông tin khớp với dữ liệu gốc."
            confidence = sum(c['score'] for c in verified_claims if c['status'] == 'SUPPORTED') / n_supported
            
        return {
            "label": final_label,
            "confidence": confidence,
            "explanation": explanation,
            "details": verified_claims
        }

# --- TEST ---
if __name__ == "__main__":
    pipeline = FactCheckerPipeline()
    
    # Test 1: Bài Fake (Sai số liệu)
    fake_text = "Thổ Nhĩ Kỳ điều 500 máy bay sơ tán công dân. Đây là chiến dịch lớn nhất lịch sử."
    
    result = pipeline.run(fake_text)
    
    print("\n" + "="*30)
    print(f"🛑 KẾT QUẢ: {result['label']} ({result['confidence']:.2%})")
    print(f"💡 Lý do: {result['explanation']}")
    print("-" * 30)
    for detail in result['details']:
        if detail['status'] != 'NEUTRAL':
            print(f"[{detail['status']}] Claim: {detail['claim']}")
            print(f"   -> Evid: {detail['evidence']}")
            print(f"   -> Score: {detail['score']:.4f}")