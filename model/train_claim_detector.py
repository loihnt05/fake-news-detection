import pandas as pd
import psycopg2
import re
import torch
import os
from sklearn.model_selection import train_test_split
from underthesea import sent_tokenize
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# --- 1. SINH DỮ LIỆU (Heuristic Weak Supervision) ---
def generate_training_data():
    print("🛠️ Đang tạo dữ liệu huấn luyện từ DB...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Lấy 5000 bài để làm mẫu
    cur.execute("SELECT content FROM articles LIMIT 5000")
    articles = cur.fetchall()
    
    claims = []
    non_claims = []
    
    print("⚙️ Đang phân loại dữ liệu mẫu...")
    for doc in tqdm(articles):
        if not doc[0]: continue
        sentences = sent_tokenize(doc[0])
        for s in sentences:
            s_clean = s.strip()
            words = s_clean.split()
            
            # Label 0: Non-claim (Rác, câu hỏi, câu dẫn)
            if (len(words) < 6 or "?" in s_clean or 
                s_clean.lower().startswith("tuy nhiên") or 
                s_clean.lower().startswith("theo đó") or
                not re.search(r'[a-zA-ZđĐ]', s_clean)):
                non_claims.append({"text": s_clean, "label": 0})
            
            # Label 1: Claim (Số liệu, Thực thể)
            elif (re.search(r'\d+', s_clean) or re.search(r'[A-ZĐ][a-zà-ỹ]+', s_clean)):
                if 10 <= len(words) <= 60:
                    claims.append({"text": s_clean, "label": 1})
    
    # Cân bằng dữ liệu
    import random
    random.shuffle(claims)
    random.shuffle(non_claims)
    min_len = min(len(claims), len(non_claims), 5000) # Lấy tối đa 5000 mỗi loại
    
    final_data = claims[:min_len] + non_claims[:min_len]
    df = pd.DataFrame(final_data)
    df = df.sample(frac=1).reset_index(drop=True) # Trộn đều
    
    print(f"✅ Đã tạo {len(df)} mẫu dữ liệu (Cân bằng Claim/Non-Claim).")
    return df

# --- 2. TRAIN MODEL (HuggingFace Native) ---
def train_model():
    # A. Chuẩn bị dữ liệu
    df = generate_training_data()
    
    # Chuyển sang format Dataset của HuggingFace
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1) # Chia train/test
    
    # B. Load Tokenizer & Model
    model_name = "vinai/phobert-base-v2"
    print(f"🚀 Loading Tokenizer & Model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Hàm tokenize dữ liệu
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    print("⚙️ Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # C. Cấu hình Training
    training_args = TrainingArguments(
        output_dir="./claim_detector_results",
        eval_strategy="epoch",  # Đánh giá sau mỗi epoch
        save_strategy="no",     # Không lưu checkpoint rác tốn dung lượng
        learning_rate=2e-5,
        per_device_train_batch_size=16, # An toàn cho GPU 
        per_device_eval_batch_size=16,
        num_train_epochs=2,     # Train 2 vòng là đủ học pattern
        weight_decay=0.01,
        use_cpu=not torch.cuda.is_available(),
        report_to="none"        # Tắt wandb đỡ phiền
    )
    
    # D. Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    # E. BẮT ĐẦU TRAIN
    print("🔥 BẮT ĐẦU TRAINING (Native Transformers)...")
    trainer.train()
    
    # F. Lưu Model thành phẩm
    output_path = "./claim_detector_model"
    print(f"💾 Đang lưu model xuống '{output_path}'...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("✅ HOÀN TẤT! Model đã sẵn sàng sử dụng.")

if __name__ == "__main__":
    train_model()