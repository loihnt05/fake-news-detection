from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import psycopg2
from typing import Optional
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load biến môi trường từ file .env
load_dotenv()

# --- CẤU HÌNH ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

THRESHOLD_DUPLICATE = 0.05  # Độ giống để coi là tin cũ
THRESHOLD_CONFIDENCE = 0.85 # Độ tin cậy để phán quyết

# --- 1. ĐỊNH NGHĨA MODEL (Phải khớp y hệt file train) ---
class NewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(NewsClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.network(x)

# --- 2. KHỞI TẠO APP & LOAD MODEL ---
app = FastAPI()
device = torch.device("cpu")

# Load embedding model (giống như trong embeded.py)
print("⏳ Đang load embedding model...")
embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
print("✅ Embedding model đã sẵn sàng!")

# Load classifier model đã train
print("⏳ Đang load classifier model...")
model = NewsClassifier(input_dim=768) # 768 khớp với vietnamese-sbert
model.load_state_dict(torch.load("model/fakenews_classifier.pth", map_location=device))
model.eval() # Chuyển sang chế độ dự đoán (tắt dropout)
print("✅ Classifier model đã sẵn sàng!")

# --- 3. HÀM TẠO VECTOR ---
def get_embedding_from_text(text: str):
    """
    Hàm này chuyển text thành vector 768 chiều sử dụng vietnamese-sbert
    (Giống logic trong embeded.py)
    """
    # Cắt text về 2000 ký tự để tránh quá dài
    text_truncated = text[:2000]
    # Tạo embedding
    embedding = embedding_model.encode([text_truncated])[0]
    return embedding.astype(np.float32)

# --- 4. LOGIC XỬ LÝ CHÍNH ---
class NewsRequest(BaseModel):
    content: str
    url: Optional[str] = None

@app.get("/")
def health_check():
    """Endpoint kiểm tra sức khỏe của API"""
    return {
        "status": "OK",
        "message": "Fake News Detection API is running",
        "models_loaded": True
    }

@app.post("/check-news")
def check_news(request: NewsRequest):
    vector_np = get_embedding_from_text(request.content)
    vector_tensor = torch.from_numpy(vector_np).unsqueeze(0) # Thêm batch dim

    # BƯỚC 1: TRA CỨU DATABASE
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Tìm bài giống nhất trong database bằng cách tính cosine similarity
        # Lưu ý: Database không dùng pgvector extension, dùng text field
        vector_str = str(vector_np.tolist())
        
        # Lấy một số bài gần đây để so sánh (simplified approach)
        sql = """
        SELECT id, label, embedding
        FROM articles
        WHERE embedding IS NOT NULL
        LIMIT 100;
        """
        cur.execute(sql)
        candidates = cur.fetchall()
        conn.close()
        
        # Tính cosine similarity với các candidates
        best_distance = float('inf')
        best_label = None
        
        if candidates:
            import ast
            for article_id, label, emb_str in candidates:
                emb = np.array(ast.literal_eval(emb_str))
                # Cosine distance = 1 - cosine similarity
                cosine_sim = np.dot(vector_np, emb) / (np.linalg.norm(vector_np) * np.linalg.norm(emb))
                distance = 1 - cosine_sim
                if distance < best_distance:
                    best_distance = distance
                    best_label = label
            
            result = (best_label, best_distance) if best_label is not None else None
        else:
            result = None
            
    except Exception as e:
        print(f"⚠️ Database lookup error: {e}")
        result = None

    # Nếu tìm thấy tin giống hệt (Cache Hit)
    if result and result[1] < THRESHOLD_DUPLICATE:
        label_db = "Real" if result[0] == 1 else "Fake"
        return {
            "status": "CACHE_HIT",
            "label": label_db,
            "confidence": 1.0,
            "color": "green" if label_db == "Real" else "red",
            "message": "Đã tìm thấy bài viết gốc trong cơ sở dữ liệu."
        }

    # BƯỚC 2: AI SUY LUẬN (INFERENCE)
    with torch.no_grad():
        logits = model(vector_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0] # [prob_fake, prob_real]
    
    score_fake = float(probs[0])
    score_real = float(probs[1])

    # BƯỚC 3: PHÂN LOẠI THEO NGƯỠNG (Logic 3 nhãn)
    if score_real > THRESHOLD_CONFIDENCE:
        final_label = "Real"
        color = "green"
    elif score_fake > THRESHOLD_CONFIDENCE:
        final_label = "Fake"
        color = "red"
    else:
        final_label = "Undefined"
        color = "yellow"

    return {
        "status": "AI_INFERENCE",
        "label": final_label,
        "confidence": max(score_real, score_fake),
        "scores": {"real": score_real, "fake": score_fake},
        "color": color,
        "message": "Kết quả dự đoán từ AI."
    }

# Chạy server: uvicorn main:app --reload