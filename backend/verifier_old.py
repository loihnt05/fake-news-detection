import os
import torch
import psycopg2
import numpy as np
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

# Đường dẫn Model
MODEL_V6_PATH = "my_model_v6" # Model so sánh câu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FactChecker:
    def __init__(self):
        print(f"🧠 [Verifier] Khởi động Decision Engine trên {DEVICE.upper()}...")
        
        # 1. Load Retriever (Tìm kiếm vector)
        print("   ├─ Loading Retriever...")
        self.retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=DEVICE)
        
        # 2. Load Verifier (Model V6 - NLI)
        print("   ├─ Loading Verifier (Model V6)...")
        try:
            # Model V6 là Cross-Encoder (Input cặp câu)
            self.verifier_model = CrossEncoder(MODEL_V6_PATH, device=DEVICE)
        except Exception as e:
            raise RuntimeError(f"❌ Không tìm thấy Model V6 tại {MODEL_V6_PATH}. Hãy train và tải về trước!")

        self.conn = psycopg2.connect(**DB_CONFIG)
        print("✅ DECISION ENGINE SẴN SÀNG!")

    def check_claim(self, claim_text):
        """
        Input: Một câu khẳng định cần kiểm tra.
        Output: Kết quả (Fake/Real/Neutral) + Bằng chứng.
        """
        # B1: Mã hóa câu hỏi
        query_vec = self.retriever.encode(claim_text).tolist()
        
        # B2: Tìm kiếm trong DB (Chỉ tìm những claim đã được verify là REAL)
        # Lưu ý: Hiện tại DB bạn toàn UNDEFINED, nên giai đoạn đầu sẽ chưa tìm thấy gì đâu.
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, content, system_label, (embedding <=> %s::vector) as distance
                FROM claims
                WHERE system_label = 'REAL'  -- Chỉ so sánh với sự thật
                ORDER BY distance ASC
                LIMIT 3;
            """, (query_vec,))
            rows = cur.fetchall()

        if not rows:
            return {
                "status": "NEUTRAL",
                "confidence": 0.0,
                "evidence": "Chưa có thông tin xác thực trong cơ sở dữ liệu."
            }

        # Lấy ứng viên tốt nhất (Distance < 0.4 là khá giống về ngữ nghĩa)
        best_candidate = rows[0]
        evidence_text = best_candidate[1]
        distance = best_candidate[3]

        if distance > 0.4:
            return {
                "status": "NEUTRAL",
                "confidence": 0.0,
                "evidence": "Không tìm thấy bằng chứng liên quan đủ mạnh."
            }

        # B3: Verification (Model V6 phán xét)
        # Model V6 trả về 3 nhãn: 0: REFUTES (Fake), 1: SUPPORTS (Real), 2: NEI
        scores = self.verifier_model.predict([claim_text, evidence_text])
        pred_label = np.argmax(scores)
        confidence = float(scores[pred_label]) # Convert numpy to float

        # Mapping nhãn V6
        result = {}
        if pred_label == 0: # REFUTES -> FAKE
            result = {
                "status": "FAKE",
                "confidence": confidence,
                "explanation": f"Mâu thuẫn với dữ liệu gốc: '{evidence_text}'"
            }
        elif pred_label == 1: # SUPPORTS -> REAL
            result = {
                "status": "REAL",
                "confidence": confidence,
                "explanation": f"Được xác thực bởi: '{evidence_text}'"
            }
        else:
            result = {
                "status": "NEUTRAL",
                "confidence": confidence,
                "explanation": "Thông tin liên quan nhưng không đủ để khẳng định đúng sai."
            }
            
        return result