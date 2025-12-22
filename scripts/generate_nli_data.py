import pandas as pd
import random
import re
import json

# Hàm biến đổi câu thật thành câu mâu thuẫn (Hard Negatives)
def generate_hard_negative(text):
    original = text
    augmented = text
    transformation_type = None

    # 1. Bẫy NGÀY THÁNG (3/4 -> 4/3)
    # Tìm mẫu d/m hoặc d/m/yyyy
    date_match = re.search(r'\b(\d{1,2})[/-](\d{1,2})\b', text)
    if date_match:
        d, m = date_match.group(1), date_match.group(2)
        if d != m: # Chỉ đảo nếu ngày khác tháng
            # Đảo ngược vị trí: 3/4 -> 4/3
            augmented = re.sub(r'\b'+d+r'[/-]'+m+r'\b', f"{m}/{d}", text)
            transformation_type = "date_swap"

    # 2. Bẫy SỐ LIỆU (7 tỷ -> 70 tỷ, 9.0 -> 90.0)
    # Chỉ chạy nếu chưa dính bẫy ngày tháng
    if augmented == original:
        num_match = re.search(r'\b(\d+(?:[.,]\d+)?)\b', text)
        if num_match:
            num_str = num_match.group(1)
            try:
                # Logic: Nhân 10, chia 10, hoặc cộng 1 đơn vị
                val = float(num_str.replace(',', '.'))
                if val < 100:
                    new_val = val * 10 if random.random() > 0.5 else val + 1
                else:
                    new_val = val / 10
                
                # Format lại số (giữ nguyên kiểu viết 9.0 hay 9,0)
                new_str = str(new_val).replace('.', ',') if ',' in num_str else str(new_val)
                augmented = text.replace(num_str, new_str, 1)
                transformation_type = "number_mismatch"
            except:
                pass

    # 3. Bẫy THỰC THỂ (Entity Swap) - Đơn giản hóa
    # Thay tên người (nếu có danh sách tên) hoặc thay địa danh
    if augmented == original:
        replacements = {
            "TP HCM": "Hà Nội", "Hà Nội": "Đà Nẵng",
            "Nguyễn": "Trần", "Mỹ": "Anh", "Việt Nam": "Thái Lan"
        }
        for k, v in replacements.items():
            if k in text:
                augmented = text.replace(k, v)
                transformation_type = "entity_swap"
                break
    
    # 4. Bẫy PHỦ ĐỊNH (Negation)
    if augmented == original:
        if "đã" in text:
            augmented = text.replace("đã", "chưa")
            transformation_type = "negation"
        elif "không" in text:
            augmented = text.replace("không", "đã")
            transformation_type = "negation"

    return augmented, transformation_type

def create_training_dataset(input_csv="data/only_real_news.csv", output_json="data/nli_train.json"):
    print("🛠️ Đang tạo dữ liệu training NLI chất lượng cao...")
    df = pd.read_csv(input_csv)
    
    dataset = []
    
    # Duyệt qua từng câu trong dataset gốc
    # Giả sử file csv có cột 'content' chứa các câu tách rồi
    sentences = df['content'].dropna().tolist()
    
    for sent in sentences:
        if len(sent) < 20: continue
        
        # 1. Tạo cặp SUPPORTS (Chính nó hoặc Paraphrase nhẹ)
        # Ở đây dùng chính nó để model học sự đồng nhất
        dataset.append({
            "sentence1": sent,
            "sentence2": sent,
            "label": 1 # SUPPORTS (Entailment)
        })
        
        # 2. Tạo cặp REFUTES (Hard Negatives)
        fake_sent, type_ = generate_hard_negative(sent)
        if fake_sent != sent:
            dataset.append({
                "sentence1": sent,
                "sentence2": fake_sent,
                "label": 0 # REFUTES (Contradiction)
            })
            
        # 3. Tạo cặp NEI (Random câu khác)
        random_sent = random.choice(sentences)
        if random_sent != sent:
             dataset.append({
                "sentence1": sent,
                "sentence2": random_sent,
                "label": 2 # NEI (Neutral)
            })

    # Lưu file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Đã tạo {len(dataset)} mẫu training. File: {output_json}")
    # In thử mẫu
    print("\n🔍 Ví dụ mẫu REFUTES:")
    for d in dataset:
        if d['label'] == 0:
            print(f"   A: {d['sentence1']}")
            print(f"   B: {d['sentence2']}")
            print("-" * 20)
            break

if __name__ == "__main__":
    # Đảm bảo bạn có file csv đầu vào nhé
    create_training_dataset()