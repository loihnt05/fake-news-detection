from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from torch.utils.data import DataLoader
import pandas as pd
import math
import os

def retrain():
    # 1. Load Model cũ
    # Lưu ý: Nếu bạn chạy từ root, đường dẫn phải trỏ đúng nơi chứa model cũ
    model_path = "./my_model" 
    if not os.path.exists(model_path):
        # Fallback: Thử tìm trong folder model nếu không thấy ở root
        model_path = "model/my_model"
        
    print(f"🚀 Loading existing model from: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        raise Exception(f"❌ Không tìm thấy model cũ tại {model_path}")

    model = CrossEncoder(model_path, num_labels=1)

    # 2. Load dữ liệu
    data_path = "smart_train_data.csv"
    if not os.path.exists(data_path):
        # Fallback nếu file csv nằm trong folder model
        data_path = "model/smart_train_data.csv"
        
    print(f"📂 Loading data from: {os.path.abspath(data_path)}")
    df = pd.read_csv(data_path)
    
    train_samples = []
    for _, row in df.iterrows():
        train_samples.append(InputExample(
            texts=[row['sentence1'], row['sentence2']], 
            label=float(row['label'])
        ))

    # 3. Training
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
    num_epochs = 2
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    # --- SỬA ĐƯỜNG DẪN OUTPUT VÀO FOLDER MODEL CHO GỌN ---
    output_dir = "model/my_model_v2"
    os.makedirs(output_dir, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True
    )

    print("\n" + "="*50)
    print(f"✅ DONE! Model mới đã được lưu tại:")
    print(f"👉 {os.path.abspath(output_dir)}")
    print("="*50)

if __name__ == "__main__":
    retrain()