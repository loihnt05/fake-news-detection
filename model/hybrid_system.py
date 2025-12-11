import torch
import torch.nn.functional as F
import faiss
import pickle
import numpy as np
import re
import time
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# ================= CẤU HÌNH HỆ THỐNG =================
# Đường dẫn file (Sửa lại cho đúng thư mục của bạn)
FAISS_INDEX_PATH = 'articles.index'
FAISS_META_PATH = 'articles_metadata.pkl'
CLASSIFIER_PATH = 'phobert_classifier.pth' # Model bạn vừa train xong

# Ngưỡng quyết định (Cần tinh chỉnh khi test thực tế)
THRESHOLD_SIMILARITY = 5.0   # Nếu khoảng cách < 5.0 => Coi là tìm thấy trong DB
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
        D, I = self.index.search(query_vec, k=1) # Tìm bài giống nhất
        
        distance = D[0][0]
        db_idx = I[0][0]
        
        # Nếu tìm thấy bài rất giống (Distance nhỏ)
        if distance < THRESHOLD_SIMILARITY and db_idx != -1:
            label_code = self.metadata['labels'][db_idx]
            label = "REAL" if label_code == 1 else "FAKE"
            original_text = self.metadata['texts'][db_idx][:100] + "..."
            
            return {
                "result": label,
                "reason": "MATCH_DB",
                "confidence": 1.0, # Tin tưởng tuyệt đối vì khớp DB
                "message": f"Tìm thấy bài viết gốc tương tự trong CSDL (Độ lệch: {distance:.2f})",
                "evidence": original_text,
                "time": time.time() - t0
            }

        # === BƯỚC 2: PHÂN TÍCH VĂN PHONG (CLASSIFIER) ===
        # Nếu không tìm thấy trong DB, dùng Model đoán
        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.classifier(inputs['input_ids'], inputs['attention_mask'])
            probs = F.softmax(logits, dim=1)
            
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()
        
        # Logic 3 trạng thái
        if real_prob > THRESHOLD_CONFIDENCE:
            return {
                "result": "REAL",
                "reason": "AI_PREDICT",
                "confidence": real_prob,
                "message": f"Văn phong chuẩn mực, độ tin cậy cao ({real_prob:.1%})",
                "time": time.time() - t0
            }
        elif fake_prob > THRESHOLD_CONFIDENCE:
            return {
                "result": "FAKE",
                "reason": "AI_PREDICT",
                "confidence": fake_prob,
                "message": f"Phát hiện văn phong nghi vấn tin giả ({fake_prob:.1%})",
                "time": time.time() - t0
            }
        else:
            return {
                "result": "UNDEFINED",
                "reason": "UNCERTAIN",
                "confidence": max(real_prob, fake_prob),
                "message": "Nội dung lạ, chưa được kiểm chứng. Cần cảnh giác!",
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