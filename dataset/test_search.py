import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# ================= CẤU HÌNH =================
INDEX_FILE = 'articles.index'
META_FILE = 'articles_metadata.pkl'
MODEL_NAME = 'keepitreal/vietnamese-sbert'

# ================= LOAD HỆ THỐNG =================
print("⏳ Đang khởi động hệ thống tìm kiếm...")

# 1. Load Model
model = SentenceTransformer(MODEL_NAME)

# 2. Load FAISS Index
index = faiss.read_index(INDEX_FILE)

# 3. Load Metadata (Nhãn & Text gốc)
with open(META_FILE, 'rb') as f:
    metadata = pickle.load(f)
    stored_texts = metadata['texts']
    stored_labels = metadata['labels']

print(f"✅ Hệ thống sẵn sàng! Đang chứa {index.ntotal} bài báo.")
print("-------------------------------------------------")

def search(query, top_k=5):
    t0 = time.time()
    
    # Vector hóa câu query
    query_vec = model.encode([query])
    
    # Tìm kiếm trong FAISS
    # D: Distance (Khoảng cách), I: Index (Vị trí trong DB)
    D, I = index.search(query_vec, top_k)
    
    t1 = time.time()
    print(f"\n🔍 Kết quả tìm kiếm cho: '{query}'")
    print(f"⏱️ Thời gian: {t1-t0:.4f} giây")
    print("-" * 60)
    
    # Duyệt qua các kết quả tìm được
    for i in range(top_k):
        idx = I[0][i]     # Vị trí trong DB
        score = D[0][i]   # Điểm khoảng cách (Càng NHỎ càng GIỐNG)
        
        if idx == -1: continue # Không tìm thấy
        
        label_code = stored_labels[idx]
        label_text = "✅ REAL" if label_code == 1 else "❌ FAKE"
        content = stored_texts[idx][:200] + "..." # Lấy 200 ký tự đầu
        
        print(f"#{i+1} | Distance: {score:.4f} | Nhãn: {label_text}")
        print(f"   📜 Nội dung: {content}")
        print("-" * 60)

# ================= VÒNG LẶP TEST =================
if __name__ == "__main__":
    while True:
        text = input("\n✍️ Nhập nội dung tin tức cần check (hoặc 'exit' để thoát): ")
        if text.lower() in ['exit', 'quit']:
            break
        
        if text.strip() == "":
            continue
            
        search(text)