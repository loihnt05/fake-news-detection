import psycopg2
import torch
import numpy as np
import os
import re
from underthesea import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Import Instant Filter for immediate blocking
from backend.instant_filter import InstantFakeNewsFilter

load_dotenv()

# --- CẤU HÌNH ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": "5432"
}

# Load Model
MODEL_PATH = "my_model_v7"  # Dùng bản mới nhất
CURRENT_MODEL_VERSION = "v7_robust_dual_branch"

class AdvancedFactChecker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [Verifier {CURRENT_MODEL_VERSION}] KHỞI ĐỘNG...")
        print(f"   🔥 Mode: Triple-Layer (Instant Filter + Memory Check + Evidence Check)")

        # LAYER 0: Instant Filter (No AI needed - Pattern matching)
        print("   ├─ Loading Instant Filter (Pattern-Based Blocking)...")
        self.instant_filter = InstantFakeNewsFilter()

        print("   ├─ Loading Retriever (Bi-Encoder)...")
        self.retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=self.device)

        print(f"   ├─ Loading Verifier (Cross-Encoder)...")
        try:
            self.verifier_model = CrossEncoder(MODEL_PATH, device=self.device)
        except:
            print("   ⚠️ Không tìm thấy model V7, load base model.")
            self.verifier_model = CrossEncoder("vinai/phobert-base", device=self.device)
    
    def clean_text(self, text):
        if not text: return ""
        text = str(text).replace('\n', '. ').replace('\r', '. ').replace('\t', ' ')
        text = re.sub(r'\.\.+', '.', text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_claims(self, text):
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)
        # Lọc câu quá ngắn
        return [s.strip() for s in sentences if len(s.split()) > 5]

    def verify(self, article_text, url=None):
        # ============================================================
        # 🚨 LAYER 0: INSTANT BLOCKING (No AI needed - milliseconds)
        # Check for known fake news patterns FIRST
        # ============================================================
        instant_result = self.instant_filter.check(article_text, url)
        
        if instant_result["should_block"]:
            print(f"   🚫 INSTANT BLOCK - Severity: {instant_result['severity']}")
            print(f"   📊 Suspicion Score: {instant_result['suspicion_score']:.2%}")
            
            return {
                "status": "FAKE",
                "confidence": min(0.95, 0.70 + instant_result['suspicion_score'] * 0.25),
                "explanation": self.instant_filter.get_warning_message(instant_result),
                "model_version": f"{CURRENT_MODEL_VERSION}_INSTANT_FILTER",
                "instant_block": True,
                "instant_reasons": instant_result["reasons"],
                "matched_patterns": [p["type"] for p in instant_result["matched_patterns"]],
                "details": [{
                    "claim_id": None,
                    "claim": "⚠️ NỘI DUNG BỊ CHẶN TỰ ĐỘNG",
                    "status": "REFUTED",
                    "score": instant_result['suspicion_score'],
                    "evidence": instant_result['reasons'][0] if instant_result['reasons'] else "Phát hiện nội dung nguy hiểm"
                }]
            }
        
        # If passed instant filter, continue with normal AI verification
        claims = self.extract_claims(article_text)
        if not claims: 
            return {
                "status": "NEUTRAL", 
                "confidence": 0.0, 
                "explanation": "Không đủ thông tin.", 
                "model_version": CURRENT_MODEL_VERSION,
                "details": []
            }

        # Vector hóa tất cả claims 1 lần (Batch processing)
        claim_vectors = self.retriever.encode(claims, convert_to_numpy=True)
        
        results_list = []
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cur:
            for i, claim in enumerate(claims):
                vector = claim_vectors[i].tolist()
                
                # ---------------------------------------------------------
                # 🛑 NHÁNH 1: MEMORY CHECK (So khớp với KNOWN FAKES)
                # "Có giống tin giả nào đã biết không?"
                # ---------------------------------------------------------
                cur.execute("""
                    SELECT id, content, (embedding <=> %s::vector) as distance
                    FROM claims
                    WHERE system_label = 'FAKE' 
                    ORDER BY distance ASC
                    LIMIT 1;
                """, (vector,))
                
                fake_row = cur.fetchone()
                
                # Ngưỡng chặn tin giả (Distance càng nhỏ càng giống)
                # < 0.15 nghĩa là giống khoảng > 85% về ngữ nghĩa
                if fake_row and fake_row[2] < 0.15:
                    print(f"   🚨 HIT BLACKLIST: {claim[:30]}... (Dist: {fake_row[2]:.3f})")
                    results_list.append({
                        "claim_id": fake_row[0],
                        "claim": claim,
                        "status": "REFUTED", # FAKE
                        "score": 0.99,       # Rất tự tin
                        "evidence": f"[CẢNH BÁO SỚM] Trùng khớp với tin giả đã xác minh: '{fake_row[1]}'"
                    })
                    continue # Bỏ qua bước check tiếp theo -> Tối ưu tốc độ

                # ---------------------------------------------------------
                # 🟢 NHÁNH 2: EVIDENCE CHECK (So khớp với REAL KNOWLEDGE)
                # "Có bằng chứng nào ủng hộ/bác bỏ không?"
                # ---------------------------------------------------------
                cur.execute("""
                    SELECT id, content, system_label, (embedding <=> %s::vector) as distance
                    FROM claims
                    WHERE system_label = 'REAL' 
                    ORDER BY distance ASC
                    LIMIT 1; 
                """, (vector,))
                
                row = cur.fetchone()
                
                status = "NEUTRAL"
                evidence_text = "Không tìm thấy dữ liệu đối chiếu."
                confidence = 0.5
                claim_id_db = None

                if row and row[3] < 0.45: # Chỉ xét nếu tìm thấy cái gì đó liên quan
                    claim_id_db = row[0]
                    evidence_text = row[1]
                    
                    # AI Phán Xét (Cross-Encoder)
                    scores = self.verifier_model.predict([claim, evidence_text])
                    scores_softmax = np.exp(scores) / np.sum(np.exp(scores))
                    pred_label = np.argmax(scores_softmax) # 0: FAKE, 1: REAL, 2: NEI
                    confidence = float(scores_softmax[pred_label])

                    status = ["REFUTED", "SUPPORTED", "NEI"][pred_label]
                
                results_list.append({
                    "claim_id": claim_id_db,
                    "claim": claim, 
                    "status": status, 
                    "evidence": evidence_text, 
                    "score": confidence
                })
        
        conn.close()
        return self.make_final_decision(results_list)

    def make_final_decision(self, details):
        # Ưu tiên cảnh báo FAKE nếu có bất kỳ claim nào bị REFUTED
        refuted = [d for d in details if d['status'] == 'REFUTED']
        supported = [d for d in details if d['status'] == 'SUPPORTED']
        
        final_status = "NEUTRAL"
        explanation = "Chưa đủ dữ liệu xác thực."
        confidence = 0.5

        if refuted:
            # Lấy cái sai nặng nhất
            top = max(refuted, key=lambda x: x['score'])
            final_status = "FAKE"
            confidence = top['score']
            
            # Kiểm tra xem do AI phát hiện hay do Blacklist
            if "CẢNH BÁO SỚM" in top['evidence']:
                explanation = "Phát hiện nội dung trùng khớp với tin giả đã biết."
            else:
                explanation = f"Mâu thuẫn với dữ liệu gốc: '{top['evidence']}'"
                
        elif len(supported) >= len(details) * 0.5:
            final_status = "REAL"
            explanation = "Nội dung khớp với dữ liệu đã xác thực."
            confidence = sum(d['score'] for d in supported) / len(supported)

        return {
            "status": final_status,
            "confidence": confidence,
            "explanation": explanation,
            "model_version": CURRENT_MODEL_VERSION,
            "details": details
        }