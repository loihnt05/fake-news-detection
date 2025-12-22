from sentence_transformers import CrossEncoder
import torch
import numpy as np
import os

# --- CẤU HÌNH ---
# Đường dẫn đến model bạn vừa tải về và giải nén
MODEL_PATH = "my_model_v3_balanced" 

print(f"⏳ Đang load Model V3 từ: {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    print("❌ Lỗi: Không tìm thấy thư mục model. Hãy kiểm tra lại đường dẫn!")
    exit()

# Load model (Tự động nhận diện GPU nếu có)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder(MODEL_PATH, num_labels=3, device=device, automodel_args={"ignore_mismatched_sizes": True})
print("✅ Model đã sẵn sàng!\n")

def debug_pair(claim, evidence):
    print("-" * 60)
    print(f"🔹 Claim:    {claim}")
    print(f"🔸 Evidence: {evidence}")
    
    # Dự đoán (trả về xác suất cho 3 lớp)
    scores = model.predict([claim, evidence], apply_softmax=True)
    
    # Mapping nhãn (Theo thứ tự lúc train: 0=REFUTED, 1=SUPPORTED, 2=NEI)
    # Lưu ý: Cần kiểm tra lại lúc train bạn gán label thế nào. 
    # Trong script train trước: label 0=REFUTED, 1=SUPPORTED, 2=NEI
    
    lbl_refuted = scores[0]
    lbl_supported = scores[1]
    lbl_nei = scores[2]
    
    print("\n🧠 Model suy nghĩ (Scores):")
    print(f"   🔴 REFUTED (Mâu thuẫn):   {lbl_refuted:.4f} ({lbl_refuted*100:.1f}%)")
    print(f"   🟢 SUPPORTED (Đồng ý):    {lbl_supported:.4f} ({lbl_supported*100:.1f}%)")
    print(f"   ⚪ NEI (Không liên quan): {lbl_nei:.4f} ({lbl_nei*100:.1f}%)")
    
    # Kết luận
    final_label = np.argmax(scores)
    if final_label == 0:
        decision = "FAKE (Mâu thuẫn)"
    elif final_label == 1:
        decision = "REAL (Xác thực)"
    else:
        decision = "NEUTRAL (Không đủ tin)"
        
    print(f"\n👉 KẾT LUẬN: {decision}")

# --- CÁC TEST CASE "HIỂM HÓC" ---

if __name__ == "__main__":
    # Case 1: Bẫy ngày tháng (3/4 vs 4/3) - Cái bạn quan tâm nhất
    debug_pair(
        claim="Sự kiện diễn ra ngày 3/4.",
        evidence="Sự kiện diễn ra ngày 4/3."
    )
    
    # Case 2: Bẫy số liệu (9.0 vs 90.0)
    debug_pair(
        claim="Cô ấy đạt 90.0 điểm IELTS.",
        evidence="Cô ấy đạt 9.0 điểm IELTS."
    )
    
    # Case 3: Bẫy địa danh (TPHCM vs Hà Nội)
    debug_pair(
        claim="Dự án được triển khai tại TP.HCM.",
        evidence="Dự án được triển khai tại Hà Nội."
    )

    # Case 4: Tin thật (Paraphrase)
    debug_pair(
        claim="V-League khai mạc vào tháng 8.",
        evidence="Giải bóng đá vô địch quốc gia sẽ bắt đầu khởi tranh từ tháng 8 tới."
    )