import psycopg2
import torch
import numpy as np
import os
import re
from underthesea import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# --- CẤU HÌNH ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Đường dẫn Model V6 (Hard Negative Specialist)
MODEL_PATH = "my_model_v6"

class AdvancedFactChecker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 KHỞI ĐỘNG HỆ THỐNG FACT-CHECKING (V6) TRÊN {self.device.upper()}...")

        # 1. RETRIEVER (Bi-Encoder)
        print("   ├─ [1/2] Loading Retriever (Search Engine)...")
        self.retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=self.device)

        # 2. VERIFIER (Cross-Encoder V6)
        print(f"   ├─ [2/2] Loading Verifier V6 từ {MODEL_PATH}...")
        try:
            # Dùng Native Transformers để tránh lỗi Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            self.model.to(self.device)
            self.model.eval() # Chế độ đánh giá
            print("      ✅ Model V6 đã sẵn sàng (Native Mode)!")
        except Exception as e:
            print(f"      ❌ LỖI LOAD MODEL: {e}")
            print("      💡 Gợi ý: Kiểm tra folder model xem có đủ file vocab.txt/config.json không?")
            exit()

    def clean_text(self, text):
        """
        Vệ sinh văn bản đầu vào.
        QUAN TRỌNG: Thay xuống dòng (\n) bằng dấu chấm (.) để tránh Title dính vào Body.
        """
        if not text: return ""
        
        # 1. Ép kiểu string
        text = str(text)
        
        # 2. Thay xuống dòng bằng dấu chấm + cách. 
        # (VD: "Tiêu đề\nNội dung" -> "Tiêu đề. Nội dung")
        text = text.replace('\n', '. ').replace('\r', '. ').replace('\t', ' ')
        
        # 3. Xóa các ký tự chấm thừa (VD: .. -> .)
        text = re.sub(r'\.\.+', '.', text)
        
        # 4. Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_claims(self, text):
        """Tách văn bản thành các câu đơn (Claims)"""
        # Bước 1: Clean text kỹ càng
        cleaned_text = self.clean_text(text)
        
        # Bước 2: Tách câu bằng Underthesea (Tách câu tiếng Việt chuẩn nhất)
        sentences = sent_tokenize(cleaned_text)
        
        # Bước 3: Lọc câu rác
        valid_claims = []
        for s in sentences:
            s = s.strip()
            # Bỏ qua câu quá ngắn (dưới 5 từ) hoặc rác điều hướng
            if len(s.split()) < 5: continue
            
            valid_claims.append(s)
            
        return valid_claims

    def predict_pair(self, claim, evidence):
        """
        Dự đoán quan hệ giữa Claim và Evidence.
        Output: List xác suất [Fake, Real, NEI]
        """
        # Tokenize (Tự động thêm <s> và </s> đúng chuẩn PhoBERT)
        inputs = self.tokenizer(
            claim, 
            evidence, 
            return_tensors='pt', 
            truncation=True, 
            max_length=256
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = softmax(outputs.logits, dim=1)[0].cpu().numpy()
        
        # Mapping nhãn model V6: 0: REFUTED, 1: SUPPORTED, 2: NEI
        return probs

    def verify(self, article_text):
        print("\n" + "="*70)
        print("📝 BẮT ĐẦU QUY TRÌNH KIỂM CHỨNG...")
        
        claims = self.extract_claims(article_text)
        print(f"🔍 Tìm thấy {len(claims)} câu cần kiểm tra (Claims).")
        
        if not claims: 
            return {"status": "NEUTRAL", "explanation": "Nội dung không đủ thông tin.", "details": []}

        # Mã hóa Claims để tìm kiếm
        claim_vectors = self.retriever.encode(claims)
        
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        results_list = []
        
        for i, raw_claim in enumerate(claims):
            claim = raw_claim # Đã clean ở bước extract
            
            # --- BƯỚC 1: RETRIEVAL (TÌM BẰNG CHỨNG) ---
            cur.execute("""
                SELECT content, (embedding <=> %s::vector) as distance
                FROM sentence_store
                ORDER BY distance ASC
                LIMIT 3; 
            """, (claim_vectors[i].tolist(),))
            rows = cur.fetchall()
            
            # Lọc ngưỡng: Distance < 0.65 (Nới lỏng chút vì V6 đủ khôn để lọc NEI)
            candidates = [self.clean_text(r[0]) for r in rows if r[1] < 0.65]
            
            if not candidates:
                print(f"   ⚪ Claim: {claim[:40]}... | SKIP (Không tìm thấy data gốc)")
                continue

            best_evidence = candidates[0]

            # --- BƯỚC 2: VERIFICATION (MODEL V6 QUYẾT ĐỊNH) ---
            # Không dùng Rule If/Else nữa, tin tưởng Model hoàn toàn.
            probs = self.predict_pair(claim, best_evidence)
            
            fake_score = probs[0]
            real_score = probs[1]
            nei_score  = probs[2]
            
            idx = np.argmax(probs)
            
            if idx == 0:   
                status = "REFUTED"
                icon = "🛑"
                confidence = fake_score
            elif idx == 1: 
                status = "SUPPORTED"
                icon = "✅"
                confidence = real_score
            else:                
                status = "NEI"
                icon = "⚪"
                confidence = nei_score
            
            print(f"   {icon} Claim: {claim[:40]}... | {status} ({confidence:.1%})")
            if status == "REFUTED":
                print(f"      ➥ Gốc: {best_evidence[:60]}...")

            results_list.append({
                "claim": claim, 
                "status": status, 
                "evidence": best_evidence, 
                "score": float(confidence),
                "probs": probs.tolist() # Lưu full để debug
            })
            
        cur.close()
        conn.close()

        # --- BƯỚC 3: KẾT LUẬN (AGGREGATION) ---
        return self.make_final_decision(results_list)

    def make_final_decision(self, details):
        if not details:
            return {"status": "NEUTRAL", "confidence": 0, "explanation": "Không tìm thấy dữ liệu đối chiếu."}

        refuted_items = [d for d in details if d['status'] == 'REFUTED']
        supported_items = [d for d in details if d['status'] == 'SUPPORTED']
        
        # --- RULE 1: PHÁT HIỆN TIN GIẢ (FAKE) ---
        # Chỉ cần 1 câu bị REFUTED với độ tin cậy > 85%
        # Model V6 đã học Hard Negative nên điểm REFUTED > 0.85 là rất đáng tin.
        strong_fakes = [d for d in refuted_items if d['score'] > 0.85]
        
        if strong_fakes:
            top = strong_fakes[0]
            return {
                "status": "FAKE",
                "confidence": top['score'],
                "explanation": f"Phát hiện sai lệch nghiêm trọng: '{top['claim']}' mâu thuẫn với dữ liệu gốc.",
                "details": details
            }

        # --- RULE 1.5: NGHI VẤN (SUSPICIOUS) ---
        # Trường hợp Model thấy điểm FAKE cao (>0.5) nhưng chưa thắng tuyệt đối (ví dụ NEI cao hơn xíu)
        # Hoặc điểm FAKE áp đảo điểm REAL (gấp 3 lần)
        for d in details:
            p_fake = d['probs'][0]
            p_real = d['probs'][1]
            if p_fake > 0.5 and p_fake > (p_real * 3):
                 return {
                    "status": "FAKE",
                    "confidence": p_fake,
                    "explanation": f"Nghi vấn sai lệch số liệu/thời gian: '{d['claim']}'.",
                    "details": details
                }

        # --- RULE 2: XÁC NHẬN TIN THẬT (REAL) ---
        # Hơn 50% câu là SUPPORTED và KHÔNG có câu nào REFUTED
        if len(supported_items) >= len(details) * 0.5 and not refuted_items:
            avg_score = sum(d['score'] for d in supported_items) / len(supported_items)
            return {
                "status": "REAL",
                "confidence": avg_score,
                "explanation": "Nội dung bài viết khớp với dữ liệu đã xác thực.",
                "details": details
            }

        # --- RULE 3: TRUNG LẬP ---
        return {
            "status": "NEUTRAL",
            "confidence": 0.5,
            "explanation": "Chưa đủ bằng chứng để kết luận (Thông tin hỗn hợp hoặc Model không chắc chắn).",
            "details": details
        }

if __name__ == "__main__":
    checker = AdvancedFactChecker()
    
    # Test case: Bẫy Số Liệu (Hard Negative)
    # Giả sử trong DB có: "Việt Nam có 7 tỷ dân" (Ví dụ vui)
    # Input sai: "Việt Nam có 70 tỷ dân"
    
    text = "Việt Nam hiện nay có dân số khoảng 70 tỷ người."
    print(f"\n>>> Input: {text}")
    
    # Lưu ý: Cần có data trong DB mới chạy được nhé
    # result = checker.verify(text)
    # print(f"👉 KẾT QUẢ: {result['status']}")