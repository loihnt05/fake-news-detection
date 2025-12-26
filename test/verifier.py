import psycopg2
import torch
import numpy as np
import os
import re
from underthesea import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
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

# Đường dẫn Model V6 (Hard Negative)
MODEL_PATH = "model/phobert_v6_hard_negative" 

class AdvancedFactChecker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [Verifier] KHỞI ĐỘNG DECISION ENGINE TRÊN {self.device.upper()}...")

        # 1. RETRIEVER (Bi-Encoder)
        print("   ├─ [1/2] Loading Retriever...")
        self.retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=self.device)

        # 2. VERIFIER (Cross-Encoder V6)
        print(f"   ├─ [2/2] Loading Verifier V6 từ {MODEL_PATH}...")
        try:
            # Model V6 train bằng CrossEncoder, nên load bằng class CrossEncoder sẽ chuẩn hơn AutoModel
            self.verifier_model = CrossEncoder(MODEL_PATH, device=self.device)
            print("      ✅ Model V6 đã sẵn sàng!")
        except Exception as e:
            print(f"      ❌ LỖI LOAD MODEL: {e}")
            raise RuntimeError("Không tìm thấy model. Hãy đảm bảo folder model đúng vị trí.")

    def clean_text(self, text):
        """Vệ sinh văn bản đầu vào"""
        if not text: return ""
        text = str(text).replace('\n', '. ').replace('\r', '. ').replace('\t', ' ')
        text = re.sub(r'\.\.+', '.', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_claims(self, text):
        """Tách văn bản thành các câu đơn"""
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)
        # Lọc câu quá ngắn
        return [s.strip() for s in sentences if len(s.split()) > 5]

    def verify(self, article_text):
        """
        Luồng kiểm chứng chính:
        1. Tách input thành các claims.
        2. Với mỗi claim, tìm kiếm trong DB (Chỉ tìm REAL claims).
        3. Dùng Model V6 so sánh -> Ra quyết định.
        """
        claims = self.extract_claims(article_text)
        if not claims: 
            return {"status": "NEUTRAL", "confidence": 0.0, "explanation": "Nội dung quá ngắn hoặc không đủ thông tin.", "details": []}

        # Mã hóa Claims input để tìm kiếm
        claim_vectors = self.retriever.encode(claims)
        
        conn = psycopg2.connect(**DB_CONFIG)
        results_list = []
        
        with conn.cursor() as cur:
            for i, claim in enumerate(claims):
                # --- QUAN TRỌNG: SỬA SQL QUERY ---
                # Chỉ lấy những claim có system_label = 'REAL'
                cur.execute("""
                    SELECT content, system_label, (embedding <=> %s::vector) as distance
                    FROM claims
                    WHERE system_label = 'REAL' 
                    ORDER BY distance ASC
                    LIMIT 1; 
                """, (claim_vectors[i].tolist(),))
                
                row = cur.fetchone()
                
                # Mặc định là NEUTRAL nếu không tìm thấy bằng chứng
                status = "NEUTRAL"
                evidence_text = "Không tìm thấy dữ liệu đối chiếu."
                confidence = 0.5
                scores_debug = [0, 0, 0]

                # Nếu tìm thấy ứng viên trong DB và khoảng cách vector đủ gần (< 0.5)
                if row and row[2] < 0.5:
                    evidence_text = row[0]
                    
                    # Dùng Model V6 phán xét (0: Fake, 1: Real, 2: NEI)
                    scores = self.verifier_model.predict([claim, evidence_text])
                    scores_softmax = np.exp(scores) / np.sum(np.exp(scores)) # Softmax thủ công
                    pred_label = np.argmax(scores_softmax)
                    confidence = float(scores_softmax[pred_label])
                    scores_debug = scores_softmax.tolist()

                    if pred_label == 0:   # REFUTES
                        status = "REFUTED"
                    elif pred_label == 1: # SUPPORTS
                        status = "SUPPORTED"
                    else:
                        status = "NEI"

                results_list.append({
                    "claim": claim, 
                    "status": status, 
                    "evidence": evidence_text, 
                    "score": confidence,
                    "probs": scores_debug
                })
        
        conn.close()
        return self.make_final_decision(results_list)

    def make_final_decision(self, details):
        """Logic tổng hợp kết quả (Decision Engine)"""
        refuted_items = [d for d in details if d['status'] == 'REFUTED']
        supported_items = [d for d in details if d['status'] == 'SUPPORTED']
        
        # RULE 1: Có bằng chứng bác bỏ mạnh (> 85%) -> FAKE
        strong_fakes = [d for d in refuted_items if d['score'] > 0.85]
        if strong_fakes:
            top = strong_fakes[0]
            return {
                "status": "FAKE",
                "confidence": top['score'],
                "explanation": f"Thông tin sai lệch: '{top['claim']}' mâu thuẫn với dữ liệu gốc.",
                "details": details
            }

        # RULE 2: Hầu hết là ủng hộ -> REAL
        if len(supported_items) >= len(details) * 0.5 and not refuted_items:
            avg_score = sum(d['score'] for d in supported_items) / len(supported_items)
            return {
                "status": "REAL",
                "confidence": avg_score,
                "explanation": "Nội dung khớp với dữ liệu đã xác thực.",
                "details": details
            }

        # RULE 3: Còn lại -> NEUTRAL (Chưa đủ thông tin)
        return {
            "status": "NEUTRAL",
            "confidence": 0.5,
            "explanation": "Hệ thống chưa có đủ dữ liệu xác thực (REAL) cho thông tin này.",
            "details": details
        }