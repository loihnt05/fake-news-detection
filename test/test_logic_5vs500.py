from sentence_transformers import CrossEncoder

# Load model NLI (Chuyên gia bắt lỗi logic)
model = CrossEncoder('symanto/xlm-roberta-base-snli-mnli-anli-xnli')

# Dữ liệu test từ bài báo của bạn
# Câu gốc (Đã được Extract ở bước trước)
evidence = "Thổ Nhĩ Kỳ ngày 27/4 điều 5 phi cơ vận tải quân sự để sơ tán công dân."

# Câu giả mạo (Fake News)
claim_fake = "Thổ Nhĩ Kỳ ngày 27/4 đã điều tới 500 phi cơ vận tải quân sự để sơ tán."

# Dự đoán
scores = model.predict([(evidence, claim_fake)])
label_mapping = ['True (Entailment)', 'Neutral', 'Fake (Contradiction)']
result = label_mapping[scores.argmax()]

print("--- KẾT QUẢ KIỂM TRA LOGIC ---")
print(f"Câu gốc: {evidence}")
print(f"Câu giả: {claim_fake}")
print(f"Model phán quyết: {result}")
print(f"Điểm số chi tiết: {scores}")