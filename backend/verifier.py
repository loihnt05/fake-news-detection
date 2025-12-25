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

# ĐỊNH DANH PHIÊN BẢN MODEL (Cực quan trọng cho Traceability)
CURRENT_MODEL_VERSION = "v6_hard_negative_2025_01"
MODEL_PATH = "my_model_v6" 

class AdvancedFactChecker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [Verifier {CURRENT_MODEL_VERSION}] KHỞI ĐỘNG...")

        print("   ├─ Loading Retriever...")
        self.retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=self.device)

        print(f"   ├─ Loading Verifier...")
        self.verifier_model = CrossEncoder(MODEL_PATH, device=self.device)
    
    def clean_text(self, text):
        if not text: return ""
        text = str(text).replace('\n', '. ').replace('\r', '. ').replace('\t', ' ')
        text = re.sub(r'\.\.+', '.', text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_claims(self, text):
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)
        return [s.strip() for s in sentences if len(s.split()) > 5]

    def verify(self, article_text):
        claims = self.extract_claims(article_text)
        if not claims: 
            return {
                "status": "NEUTRAL", 
                "confidence": 0.0, 
                "explanation": "Không đủ thông tin.", 
                "model_version": CURRENT_MODEL_VERSION,
                "details": []
            }

        claim_vectors = self.retriever.encode(claims)
        conn = psycopg2.connect(**DB_CONFIG)
        results_list = []
        
        with conn.cursor() as cur:
            for i, claim in enumerate(claims):
                # Lấy thêm ID để phục vụ Feedback Loop
                cur.execute("""
                    SELECT id, content, system_label, (embedding <=> %s::vector) as distance
                    FROM claims
                    WHERE system_label = 'REAL' 
                    ORDER BY distance ASC
                    LIMIT 1; 
                """, (claim_vectors[i].tolist(),))
                
                row = cur.fetchone()
                
                status = "NEUTRAL"
                evidence_text = "Không tìm thấy dữ liệu đối chiếu."
                confidence = 0.5
                claim_id_db = None # ID trong DB (nếu tìm thấy)

                if row and row[3] < 0.5:
                    claim_id_db = row[0]
                    evidence_text = row[1]
                    
                    scores = self.verifier_model.predict([claim, evidence_text])
                    scores_softmax = np.exp(scores) / np.sum(np.exp(scores))
                    pred_label = np.argmax(scores_softmax)
                    confidence = float(scores_softmax[pred_label])

                    status = ["REFUTED", "SUPPORTED", "NEI"][pred_label]

                results_list.append({
                    "claim_id": claim_id_db, # Trả về ID để Extension biết report vào đâu
                    "claim": claim, 
                    "status": status, 
                    "evidence": evidence_text, 
                    "score": confidence
                })
        
        conn.close()
        return self.make_final_decision(results_list)

    def make_final_decision(self, details):
        # Logic aggregation (giữ nguyên hoặc nâng cấp)
        refuted = [d for d in details if d['status'] == 'REFUTED']
        supported = [d for d in details if d['status'] == 'SUPPORTED']
        
        final_status = "NEUTRAL"
        explanation = "Chưa đủ dữ liệu."
        confidence = 0.5

        if refuted:
            top = max(refuted, key=lambda x: x['score'])
            if top['score'] > 0.85:
                final_status = "FAKE"
                explanation = f"Mâu thuẫn: '{top['claim']}'"
                confidence = top['score']
        elif len(supported) >= len(details) * 0.5:
            final_status = "REAL"
            explanation = "Khớp với dữ liệu xác thực."
            confidence = sum(d['score'] for d in supported) / len(supported)

        return {
            "status": final_status,
            "confidence": confidence,
            "explanation": explanation,
            "model_version": CURRENT_MODEL_VERSION, # Đóng dấu phiên bản
            "details": details
        }