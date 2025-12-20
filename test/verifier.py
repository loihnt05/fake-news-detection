import psycopg2
import torch
import numpy as np
import pandas as pd
import os
import joblib
import re
from underthesea import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

class AdvancedFactChecker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 KHỞI ĐỘNG HỆ THỐNG TRÊN {self.device.upper()}...")

        # 1. LOAD CLAIM DETECTOR (PhoBERT)
        print("   ├─ [1/4] Loading Claim Detector...")
        claim_path = "./claim_detector_model"
        if os.path.exists(claim_path):
            self.claim_tokenizer = AutoTokenizer.from_pretrained(claim_path)
            self.claim_model = AutoModelForSequenceClassification.from_pretrained(claim_path).to(self.device)
        else:
            print("   ⚠️ Không thấy Claim Model, sẽ dùng luật Heuristic.")
            self.claim_model = None

        # 2. LOAD RETRIEVER (Bi-Encoder)
        print("   ├─ [2/4] Loading Retriever...")
        self.retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=self.device)

        # 3. LOAD VERIFIER (Cross-Encoder Fine-tuned)
        print("   ├─ [3/4] Loading NLI Verifier...")
        possible_paths = ["model/my_model_v2/final_model_saved", "my_model_v2/final_model_saved", "my_model"]
        nli_path = next((p for p in possible_paths if os.path.exists(p)), None)
        
        if nli_path:
            print(f"      -> Dùng model: {nli_path}")
            self.verifier = CrossEncoder(nli_path, device=self.device, model_kwargs={"ignore_mismatched_sizes": True})
        else:
            print("      ⚠️ Dùng model gốc (kém chính xác hơn).")
            self.verifier = CrossEncoder("cross-encoder/nli-distilroberta-base", num_labels=1, device=self.device)

        # 4. LOAD FINAL CLASSIFIER (XGBoost/Sklearn)
        print("   ├─ [4/4] Loading Final Classifier...")
        clf_path = 'final_classifier.pkl'
        self.final_clf = joblib.load(clf_path) if os.path.exists(clf_path) else None
            
        print("✅ HỆ THỐNG SẴN SÀNG!\n")

    def super_logic_check(self, claim, evidence):
        """
        Bộ lọc Logic Cứng (Hard Rules) - Phiên bản Fix lỗi 90.0 vs 9.0
        Thứ tự ưu tiên: SỐ LIỆU > NGÀY THÁNG > TEXT OVERLAP
        """
        c_lower = claim.lower().strip()
        e_lower = evidence.lower().strip()
        
        # --- 1. LOGIC SỐ LIỆU (NUMBER CHECK) - QUAN TRỌNG NHẤT ---
        # Regex bắt số thực (9.0, 90.0, 1,500) và số nguyên
        # Pattern: Số + (dấu chấm/phẩy + số) tuỳ chọn
        num_pattern = r'\d+(?:[.,]\d+)?'
        
        c_nums = re.findall(num_pattern, c_lower)
        e_nums = re.findall(num_pattern, e_lower)
        
        # Hàm chuẩn hóa số (9,0 -> 9.0)
        def parse_num(s):
            try: return float(s.replace(',', '.'))
            except: return None

        # Danh sách số trong Evidence (đổi sang float để so sánh giá trị)
        e_vals = [parse_num(x) for x in e_nums if parse_num(x) is not None]
        
        missing_nums = []
        for c_str in c_nums:
            c_val = parse_num(c_str)
            if c_val is None: continue
            
            # Bỏ qua các số ngày tháng (để logic ngày tháng xử lý sau)
            # VD: tránh bắt lỗi số 4 trong "ngày 4/1" nếu logic ngày tháng làm tốt
            # Nhưng ở đây ta cứ check chặt.
            
            # Logic: Số trong Claim phải TỒN TẠI trong Evidence (sai số cực nhỏ)
            found = False
            for e_val in e_vals:
                if abs(c_val - e_val) < 0.001: # Chấp nhận sai số float
                    found = True
                    break
            
            if not found:
                missing_nums.append(c_str)
        
        if missing_nums:
            # Nếu sai số -> REFUTED ngay lập tức
            return "REFUTED", f"Sai số liệu: Claim có {missing_nums} nhưng Evidence không có (tìm thấy {e_nums})."

        # --- 2. LOGIC NGÀY THÁNG (DATE CHECK) ---
        month_match = re.search(r'tháng (\d{1,2})', c_lower)
        if month_match:
            m_claim = int(month_match.group(1))
            patterns = [
                f"tháng {m_claim}", f"tháng {m_claim:02d}",
                f"/{m_claim}/", f"/{m_claim:02d}/",
                f"-{m_claim}-", f"-{m_claim:02d}-",
                f"/{m_claim} ", f"/{m_claim:02d} ",
                f"/{m_claim}.", f"/{m_claim:02d}.",
                f"/{m_claim}", f"/{m_claim:02d}"
            ]
            has_month = any(p in e_lower for p in patterns)
            if not has_month:
                regex_date = fr"[\/\-]0?{m_claim}[\/\-]"
                if not re.search(regex_date, e_lower):
                    return "REFUTED", f"Sai tháng: Claim tháng {m_claim} nhưng Evidence không có."

        # --- 3. LOGIC TRÙNG KHỚP VĂN BẢN (TEXT OVERLAP) ---
        # Chỉ chạy khi Số liệu và Ngày tháng đã OK
        c_clean = c_lower.replace('\n', ' ')
        e_clean = e_lower.replace('\n', ' ')
        
        if e_clean in c_clean or c_clean in e_clean:
            return "SUPPORTED", 1.0

        c_tokens = set(c_clean.split())
        e_tokens = set(e_clean.split())
        if not c_tokens or not e_tokens: return "PASS", "No tokens"

        overlap_ratio = len(c_tokens.intersection(e_tokens)) / min(len(c_tokens), len(e_tokens))
        
        if overlap_ratio > 0.85:
             return "SUPPORTED", 0.95

        return "PASS", "Logic OK"

    def extract_claims(self, text):
        sentences = sent_tokenize(text)
        if not sentences: return []
        
        candidates = [s for s in sentences if len(s.split()) > 5]
        final_claims = []
        
        if self.claim_model:
            inputs = self.claim_tokenizer(candidates, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.claim_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            
            for i, sent in enumerate(candidates):
                has_digit = bool(re.search(r'\d+', sent))
                # Lấy nếu AI tự tin hoặc có số liệu (tránh bỏ sót ngày tháng)
                if scores[i] > 0.4 or has_digit: 
                    final_claims.append(sent)
        else:
            final_claims = [s for s in candidates if any(c.isdigit() for c in s)]
            
        return final_claims

    def verify(self, article_text):
        print("="*60)
        print("📝 BẮT ĐẦU KIỂM TRA BÀI VIẾT...")
        claims = self.extract_claims(article_text)
        print(f"🔍 Tìm thấy {len(claims)} câu cần kiểm chứng (Claims).")
        
        if not claims: 
            return {"status": "NEUTRAL", "explanation": "Không tìm thấy thông tin định lượng để kiểm chứng.", "details": []}

        # --- GIAI ĐOẠN 1: RETRIEVAL (TÌM KIẾM) ---
        print("📡 Đang truy xuất bằng chứng từ Kho tri thức...")
        claim_vectors = self.retriever.encode(claims)
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        verified_details = []
        
        for i, claim in enumerate(claims):
            # Tìm top 5 câu gần nhất
            cur.execute("""
                SELECT content, (embedding <=> %s::vector) as distance
                FROM sentence_store
                ORDER BY distance ASC
                LIMIT 5; 
            """, (claim_vectors[i].tolist(),))
            results = cur.fetchall()
            
            # Ngưỡng 0.6 để bắt Paraphrase
            valid_evidence = [r for r in results if r[1] < 0.60]
            
            if not valid_evidence:
                verified_details.append({"claim": claim, "status": "NEI", "score": 0.5, "evidence": "Không tìm thấy dữ liệu đối chiếu."})
                continue
            
            # Lấy câu bằng chứng tốt nhất (Distance nhỏ nhất)
            best_evid_text = valid_evidence[0][0]
            best_dist = valid_evidence[0][1]
            
            # --- GIAI ĐOẠN 2: VERIFICATION (LOGIC + AI) ---
            
            # A. Kiểm tra Logic Cứng
            # Hàm logic bây giờ trả về (Status, Message/Score)
            logic_result = self.super_logic_check(claim, best_evid_text)
            logic_status, logic_info = self.super_logic_check(claim, best_evid_text)
            
            if logic_status == "REFUTED":
                status = "REFUTED"
                final_score = 0.0  # Điểm 0 tròn trĩnh
                print(f"   🛑 LOGIC CATCH: {logic_info}")
            
            elif logic_status == "SUPPORTED":
                status = "SUPPORTED"
                final_score = float(logic_info)
            else:
                # Logic PASS -> Dùng AI chấm
                pairs = [[best_evid_text, claim]]
                nli_score = float(self.verifier.predict(pairs)[0])
                final_score = nli_score
                
                if final_score > 0.65: status = "SUPPORTED"
                elif final_score < 0.35: status = "REFUTED"
                else: status = "NEUTRAL"
                
                # Boost điểm nếu Logic PASS và NLI > 0.55
                if logic_status == "PASS" and final_score > 0.55:
                    status = "SUPPORTED"
                    final_score = 0.85

            verified_details.append({
                "claim": claim, "status": status, "evidence": best_evid_text, "score": final_score
            })

        cur.close()
        conn.close()

        # --- TỔNG HỢP KẾT QUẢ ---
        scores = [x['score'] for x in verified_details if x['status'] != 'NEI']
        
        if not scores: 
            final_status = "NEUTRAL"
            confidence = 0.5
            explanation = "Chưa đủ dữ liệu trong kho tri thức."
        # Quy tắc: 1 câu sai -> Cả bài sai (Tin giả thường trộn 9 thật 1 giả)
        elif any(x['status'] == 'REFUTED' for x in verified_details):
            final_status = "FAKE"
            confidence = 1.0 # Rất tự tin là Fake
            explanation = "Hệ thống phát hiện mâu thuẫn về số liệu hoặc thời gian với dữ liệu gốc."
        elif np.mean(scores) > 0.7:
            final_status = "REAL"
            confidence = np.mean(scores)
            explanation = "Nội dung khớp với dữ liệu đã được xác thực."
        else:
            final_status = "NEUTRAL"
            confidence = 0.5
            explanation = "Thông tin chưa rõ ràng hoặc gây tranh cãi."

        print("-" * 60)
        print(f"🤖 KẾT LUẬN CUỐI CÙNG: {final_status} (Độ tin cậy: {confidence:.2%})")
        print(f"📝 Giải thích: {explanation}")
        print("=" * 60)
        
        return {"status": final_status, "confidence": confidence, "explanation": explanation, "details": verified_details}

if __name__ == "__main__":
    checker = AdvancedFactChecker()
    
    # --- CHẠY THỬ ---
    print("\n>>> TEST CASE 1: Báo Giả (Nisha Patel - Sai ngày tháng)")
    fake_news = """
    Fadi bị bắt vì tội giết vợ vào ngày 32/2/2007. 
    Ngày 56/5/2008, Fadi bị kết tội.
    """
    checker.verify(fake_news)

    print("\n>>> TEST CASE 2: Báo Thật (V-League)")
    real_news = "V-League 2024-2025 dự kiến khai mạc vào tháng 8."
    checker.verify(real_news)
    
    print("\n>>> TEST CASE 3: Báo Giả (V-League sai tháng)")
    fake_vleague = "V-League 2024-2025 dự kiến khai mạc vào tháng 12 năm nay."
    checker.verify(fake_vleague)