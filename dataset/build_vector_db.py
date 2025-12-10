import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ================= CẤU HÌNH (SỬA LẠI CHO ĐÚNG FILE CỦA BẠN) =================
INPUT_FILE = 'articles_clean.csv'   # File kết quả của bước clean_data.py
INDEX_FILE = 'articles.index'      # Tên file DB Vector sẽ tạo ra
META_FILE = 'articles_metadata.pkl' # Tên file chứa nhãn
MODEL_NAME = 'keepitreal/vietnamese-sbert' 

# QUAN TRỌNG: Sửa tên cột này giống hệt bước trước bạn đã sửa
COL_TEXT = 'content'  
COL_LABEL = 'label'   # Tên cột nhãn

# ================= CODE XỬ LÝ =================
def build_db():
    print(f"📂 Đang đọc file {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        
        # Kiểm tra xem cột có tồn tại không
        if COL_TEXT not in df.columns:
            print(f"❌ Lỗi: Không tìm thấy cột '{COL_TEXT}' trong file csv.")
            print(f"   Các cột hiện có: {list(df.columns)}")
            return

        documents = df[COL_TEXT].tolist()
        labels = df[COL_LABEL].tolist()
        print(f"✅ Đã load {len(documents)} bài báo.")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    # Load Model
    print("🤖 Đang tải model AI...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Tạo Vector
    print("🚀 Đang biến đổi văn bản thành Vector (Sẽ mất thời gian)...")
    # Batch size giúp không bị tràn RAM
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # Xây dựng FAISS
    print("🗄️ Đang đóng gói vào FAISS Index...")
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Lưu file
    print("💾 Đang lưu xuống ổ cứng...")
    faiss.write_index(index, INDEX_FILE)
    
    # Lưu metadata (Nhãn)
    with open(META_FILE, 'wb') as f:
        pickle.dump({'texts': documents, 'labels': labels}, f)

    print("\n🎉 XONG! Bạn đã có Database AI.")
    print(f"Output: {INDEX_FILE} và {META_FILE}")

if __name__ == "__main__":
    build_db()