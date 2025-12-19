import pandas as pd
import psycopg2
import re
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from underthesea import sent_tokenize
import os
from dotenv import load_dotenv
from tqdm import tqdm
import torch

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def generate_training_data():
    print("🛠️ Đang tạo dữ liệu huấn luyện từ Database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Lấy 5000 bài bất kỳ để sinh dữ liệu train
    cur.execute("SELECT content FROM articles LIMIT 5000")
    articles = cur.fetchall()
    
    claims = []
    non_claims = []
    
    print("⚙️ Đang phân loại dữ liệu mẫu (Heuristic)...")
    for doc in tqdm(articles):
        if not doc[0]: continue
        sentences = sent_tokenize(doc[0])
        
        for s in sentences:
            s_clean = s.strip()
            words = s_clean.split()
            
            # --- LUẬT ĐỂ TẠO DỮ LIỆU MẪU (CHỈ DÙNG ĐỂ TRAIN) ---
            
            # 1. NON-CLAIM (Rác, câu dẫn, câu hỏi)
            if (len(words) < 6 or 
                "?" in s_clean or 
                s_clean.lower().startswith("tuy nhiên") or
                s_clean.lower().startswith("theo đó") or
                not re.search(r'[a-zA-ZđĐ]', s_clean)): # Không có chữ cái
                non_claims.append([s_clean, 0])
                
            # 2. CLAIM (Chứa số liệu HOẶC Thực thể viết hoa + Độ dài đủ)
            elif (re.search(r'\d+', s_clean) or re.search(r'[A-ZĐ][a-zà-ỹ]+', s_clean)):
                if 10 <= len(words) <= 60: # Claim thường không quá ngắn cũng không quá dài (cả đoạn văn)
                    claims.append([s_clean, 1])

    # Cân bằng dữ liệu: Lấy 5000 Claim + 5000 Non-Claim
    min_len = min(len(claims), len(non_claims), 5000)
    
    print(f"📊 Tìm thấy: {len(claims)} claims tiềm năng, {len(non_claims)} non-claims.")
    print(f"⚖️ Đang cân bằng dữ liệu về {min_len} mẫu mỗi loại...")
    
    import random
    random.shuffle(claims)
    random.shuffle(non_claims)
    
    final_data = claims[:min_len] + non_claims[:min_len]
    df = pd.DataFrame(final_data, columns=["text", "labels"])
    
    # Trộn đều
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def train_model():
    # 1. Chuẩn bị dữ liệu
    train_df = generate_training_data()
    
    # 2. Cấu hình Model PhoBERT
    model_args = {
        "num_train_epochs": 2,              # Train nhanh 2 vòng là đủ học pattern
        "train_batch_size": 32,
        "overwrite_output_dir": True,
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False,
        "output_dir": "claim_detector_model",
        "use_multiprocessing": False,
        "fp16": torch.cuda.is_available(),
    }
    
    # 3. Khởi tạo Model
    print("🚀 Đang load PhoBERT base...")
    model = ClassificationModel(
        "roberta", 
        "vinai/phobert-base-v2", 
        num_labels=2, 
        args=model_args, 
        use_cuda=torch.cuda.is_available()
    )
    
    # 4. Train
    print("🔥 BẮT ĐẦU TRAINING CLAIM DETECTOR...")
    train_split, eval_split = train_test_split(train_df, test_size=0.1)
    model.train_model(train_split)
    
    # 5. Đánh giá
    result, _, _ = model.eval_model(eval_split)
    print(f"✅ Kết quả đánh giá: {result}")
    print("💾 Model đã lưu tại: ./claim_detector_model")

if __name__ == "__main__":
    # Cài thư viện nếu thiếu: pip install simpletransformers
    train_model()