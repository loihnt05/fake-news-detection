from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import json
import torch
import os
import random
import logging

# Tắt bớt log cảnh báo rác của Transformers
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# --- CẤU HÌNH "TỐC ĐỘ CAO" ---
MODEL_NAME = "vinai/phobert-base-v2"
BATCH_SIZE = 4          
EPOCHS = 2              
MAX_SAMPLES = 15000     
MAX_SEQ_LENGTH = 256    # PhoBERT giới hạn 256
OUTPUT_PATH = "model/my_model_v3_fast"

# Dọn dẹp GPU
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train():
    print(f"🚀 Bắt đầu Train NLI Tốc độ cao (Max {MAX_SAMPLES} mẫu)...")
    
    # 1. Load & Lọc Data
    train_samples = []
    skipped = 0
    try:
        with open("data/nli_train.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            random.shuffle(data) # Xáo trộn ngẫu nhiên
            
            print("🧹 Đang lọc và cắt gọt dữ liệu...")
            for item in data:
                s1 = item['sentence1']
                s2 = item['sentence2']
                
                # --- FIX LỖI TOKEN LENGTH ---
                # Cắt bớt câu nếu quá dài TRƯỚC khi đưa vào model
                # Ước lượng: 1 từ ~ 1.5 token. Để an toàn, ta lấy tối đa 160 từ tổng cộng.
                words1 = s1.split()[:100] # Câu 1 lấy max 100 từ
                words2 = s2.split()[:60]  # Câu 2 lấy max 60 từ (thường claim ngắn hơn)
                
                # Ghép lại
                s1_trunc = " ".join(words1)
                s2_trunc = " ".join(words2)
                
                # Nếu sau khi cắt mà vẫn quá ngắn (dưới 3 từ) thì bỏ qua (rác)
                if len(words1) < 3 or len(words2) < 3:
                    skipped += 1
                    continue
                    
                train_samples.append(InputExample(
                    texts=[s1_trunc, s2_trunc], 
                    label=item['label']
                ))
                
                if len(train_samples) >= MAX_SAMPLES:
                    break
                    
    except Exception as e:
        print(f"❌ Lỗi đọc data: {e}")
        return

    print(f"📊 Đã chọn: {len(train_samples)} mẫu sạch (Bỏ qua {skipped} mẫu lỗi/quá dài).")
    
    # 2. DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    
    # 3. Model Config
    model = CrossEncoder(
        MODEL_NAME, 
        num_labels=3, 
        max_length=MAX_SEQ_LENGTH,
        # --- FIX LỖI DEPRECATED ---
        # Đổi automodel_args thành model_kwargs
        model_kwargs={"ignore_mismatched_sizes": True} 
    )
    
    # 4. Train
    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)
    estimated_hours = (len(train_dataloader) * EPOCHS * 0.5) / 3600 # Giả sử 0.5s/batch (nhanh hơn do cắt ngắn)
    
    print(f"🔥 Bắt đầu training... (Dự kiến: {estimated_hours:.2f} giờ)")
    
    model.fit(
        train_dataloader=train_dataloader,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_PATH,
        use_amp=True, # Mixed Precision giúp giảm VRAM và tăng tốc
        show_progress_bar=True
    )
    
    print(f"✅ Xong! Model lưu tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    train()