import pandas as pd
import random
from datetime import datetime, timedelta

def generate_date_logic():
    data = []
    
    # Mẫu câu template
    templates = [
        ("Sự kiện diễn ra vào {}", "Sự kiện bắt đầu từ {}"),
        ("Dự án khởi công ngày {}", "Dự án làm lễ động thổ {}"),
        ("V-League khai mạc {}", "Giải đấu bắt đầu {}"),
        ("Hạn chót là {}", "Deadline là {}"),
    ]

    start_date = datetime(2023, 1, 1)
    
    for _ in range(2000): # Sinh 2000 cặp câu
        # 1. Logic NGÀY THÁNG (Entailment) - TRUE
        # "Ngày 15/8" thì cũng là "Tháng 8" -> Model phải học là TRUE
        curr_date = start_date + timedelta(days=random.randint(0, 365))
        day_str = curr_date.strftime("%d/%m")
        month_str = f"tháng {curr_date.month}"
        
        t1, t2 = random.choice(templates)
        
        # Cặp TRUE: Cụ thể -> Khái quát
        # Premise: "Khai mạc ngày 15/8"
        # Hypothesis: "Khai mạc tháng 8"
        row_true = {
            "sentence1": t1.format(f"ngày {day_str}"),
            "sentence2": t2.format(month_str),
            "label": 1.0 # True
        }
        data.append(row_true)
        
        # 2. Logic NGÀY THÁNG (Contradiction) - FAKE
        # "Ngày 15/8" KHÔNG PHẢI "Tháng 9" -> FAKE
        wrong_month = (curr_date.month % 12) + 1
        row_fake = {
            "sentence1": t1.format(f"ngày {day_str}"),
            "sentence2": t2.format(f"tháng {wrong_month}"),
            "label": 0.0 # Fake
        }
        data.append(row_fake)

        # 3. Logic SỐ LƯỢNG (Quantity Mismatch) - FAKE
        # Giữ nguyên kiến thức cũ: 500 khác 5
        num = random.randint(10, 100)
        wrong_num = num * random.randint(2, 10) # Lệch hẳn
        
        row_num_fake = {
            "sentence1": f"Có {num} người tham gia.",
            "sentence2": f"Có {wrong_num} người tham gia.",
            "label": 0.0
        }
        data.append(row_num_fake)

        # 4. Logic SỐ LƯỢNG (Exact Match) - TRUE
        row_num_true = {
            "sentence1": f"Có {num} người tham gia.",
            "sentence2": f"Tổng cộng {num} thành viên góp mặt.",
            "label": 1.0
        }
        data.append(row_num_true)

    df = pd.DataFrame(data)
    print(f"✅ Đã sinh {len(df)} mẫu dữ liệu luyện IQ.")
    print(df.head())
    return df

if __name__ == "__main__":
    df = generate_date_logic()
    df.to_csv("smart_train_data.csv", index=False)