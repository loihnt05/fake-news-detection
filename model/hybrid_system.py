import torch
import torch.nn.functional as F
import faiss
import pickle
import numpy as np
import re
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# ================= CẤU HÌNH HỆ THỐNG =================
# Đường dẫn file (Sửa lại cho đúng thư mục của bạn)
BASE_DIR = Path(__file__).resolve().parent.parent  # Project root directory
FAISS_INDEX_PATH = str(BASE_DIR / 'dataset' / 'articles.index')
FAISS_META_PATH = str(BASE_DIR / 'dataset' / 'articles_metadata.pkl')
CLASSIFIER_PATH = str(BASE_DIR / 'model' / 'phobert_classifier.pth')  # Model bạn vừa train xong

# Ngưỡng quyết định (Cần tinh chỉnh khi test thực tế)
THRESHOLD_SIMILARITY = 20   # Nếu khoảng cách < 5.0 => Coi là tìm thấy trong DB
THRESHOLD_CONFIDENCE = 0.90  # Nếu xác suất > 90% => Mới tin model phân loại

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ System running on: {device}")

# ================= CLASS XỬ LÝ CHÍNH =================
class FakeNewsDetector:
    def __init__(self):
        print("⏳ Đang khởi động hệ thống Hybrid...")
        
        # 1. Load PhoBERT Classifier (Model 2)
        print("   - Loading Classifier Model...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.classifier = self._load_classifier_model()
        self.classifier.to(device)
        self.classifier.eval()

        # 2. Load FAISS & SBERT (Model 1)
        print("   - Loading FAISS Database...")
        self.vector_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, 'rb') as f:
            self.metadata = pickle.load(f)
            
        print("✅ Hệ thống đã sẵn sàng sàng lọc tin giả!")

    def _load_classifier_model(self):
        # Định nghĩa lại kiến trúc model để load weights
        import torch.nn as nn
        class PhoBertClassifier(nn.Module):
            def __init__(self):
                super(PhoBertClassifier, self).__init__()
                self.bert = AutoModel.from_pretrained("vinai/phobert-base")
                self.classifier = nn.Sequential(
                    nn.Linear(768, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 2)
                )
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                return self.classifier(cls_output)
        
        model = PhoBertClassifier()
        # Load trọng số đã train
        model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
        return model

    def normalize_text(self, text):
        # Hàm làm sạch giống hệt lúc train
        if not isinstance(text, str): return ""
        text = re.sub(r'\s+([.,;?!:])', r'\1', text) # Xóa space thừa
        text = re.sub(r'(\d)\s+/\s+(\d)', r'\1/\2', text) # Fix ngày tháng
        text = re.sub(r'^[A-ZĐÀ-Ỹ ]+\s*-\s*', '', text) # Xóa nguồn tin lộ
        return text.strip()

    def check(self, raw_text):
        t0 = time.time()
        clean_text = self.normalize_text(raw_text)
        
        # === BƯỚC 1: TRA CỨU DATABASE (FAISS) ===
        query_vec = self.vector_model.encode([clean_text])
        D, I = self.index.search(query_vec, k=1) 
        
        distance = D[0][0]
        db_idx = I[0][0]

        # In ra để bạn tinh chỉnh (Sau này xóa đi)
        print(f"   [Debug] Distance: {distance:.2f}") 
        
        # === BƯỚC 2: PHÂN TÍCH VĂN PHONG (CLASSIFIER) ===
        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.classifier(inputs['input_ids'], inputs['attention_mask'])
            probs = F.softmax(logits, dim=1)
            
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

        # === BƯỚC 3: RA QUYẾT ĐỊNH (LOGIC HYBRID MỚI) ===
        
        # Case A: Tìm thấy bài giống hệt trong DB (Khoảng cách rất gần)
        if distance < THRESHOLD_SIMILARITY and db_idx != -1: # Ví dụ < 5.0
            label_code = self.metadata['labels'][db_idx]
            label = "REAL" if label_code == 1 else "FAKE"
            return {
                "result": label,
                "reason": "MATCH_DB",
                "message": f"Khớp dữ liệu gốc (Độ lệch: {distance:.2f})",
                "confidence": 1.0,
                "time": time.time() - t0
            }

        # Case B: Nội dung quá xa lạ (Distance quá lớn) -> UNDEFINED NGAY LẬP TỨC
        # Đây chính là cái "lưới" để bắt câu Người ngoài hành tinh
        THRESHOLD_UNKNOWN = 55 # Bạn hãy chỉnh số này dựa trên kết quả debug
        
        if distance > THRESHOLD_UNKNOWN:
            return {
                "result": "UNDEFINED",
                "reason": "UNKNOWN_TOPIC", # Lý do: Chủ đề lạ
                "message": f"Nội dung quá mới hoặc lạ lẫm (Distance: {distance:.2f}). AI chưa đủ dữ liệu kiểm chứng.",
                "confidence": 0.0,
                "time": time.time() - t0
            }

        # Case C: Nội dung có liên quan (5 < Distance < 25) -> Tin vào Classifier
        if real_prob > THRESHOLD_CONFIDENCE:
            return {
                "result": "REAL",
                "reason": "AI_PREDICT",
                "confidence": real_prob,
                "message": f"Văn phong tin cậy ({real_prob:.1%})",
                "time": time.time() - t0
            }
        elif fake_prob > THRESHOLD_CONFIDENCE:
            return {
                "result": "FAKE",
                "reason": "AI_PREDICT",
                "confidence": fake_prob,
                "message": f"Văn phong lừa đảo ({fake_prob:.1%})",
                "time": time.time() - t0
            }
        else:
            return {
                "result": "UNDEFINED",
                "reason": "UNCERTAIN",
                "message": "AI lưỡng lự.",
                "confidence": max(real_prob, fake_prob),
                "time": time.time() - t0
            }

# ================= CHẠY THỬ =================
if __name__ == "__main__":
    detector = FakeNewsDetector()
    
    # 3 Trường hợp test kinh điển
    test_cases = [
        # Case 1: Tin thật (Copy từ DB hoặc sửa nhẹ)
        "Bộ Y tế yêu cầu các địa phương đẩy mạnh tiêm chủng vắc xin COVID-19 mũi nhắc lại.",
        
        # Case 2: Tin giả văn phong lừa đảo (Model Classifier sẽ bắt)
        "SỐC!!!! Chia sẻ ngay để nhận tiền từ thiện. Bấm vào link bên dưới nếu không sẽ bị khóa tài khoản vĩnh viễn!!!",
        
        # Case 3: Tin bịa đặt nhưng văn phong nghiêm túc (Undefined)
        "Người ngoài hành tinh vừa hạ cánh xuống Hồ Gươm và đi ăn kem Tràng Tiền chiều nay."
    ]

    print("\n" + "="*50)
    for text in test_cases:
        print(f"\n📰 Input: {text}")
        res = detector.check(text)
        
        # In kết quả đẹp
        color = "🟢" if res['result'] == 'REAL' else "🔴" if res['result'] == 'FAKE' else "🟡"
        print(f"{color} KẾT LUẬN: {res['result']}")
        print(f"   Logic: {res['reason']}")
        print(f"   Chi tiết: {res['message']}")
        if 'evidence' in res:
             print(f"   Bằng chứng: {res['evidence']}")
        print(f"   Thời gian xử lý: {res['time']:.4f}s")
    print("\n" + "="*50)