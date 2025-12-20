import re
from sentence_transformers import CrossEncoder
import os
import torch

# --- CẤU HÌNH MODEL ---
# Ưu tiên dùng model bạn đã train, nếu không có thì dùng model gốc để test logic
MODEL_PATH = "model/my_model_v2" 
if not os.path.exists(MODEL_PATH):
    print("⚠️ Không tìm thấy model v2, dùng model mặc định để test logic...")
    MODEL_PATH = "cross-encoder/nli-distilroberta-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⏳ Loading NLI Model từ {MODEL_PATH}...")
try:
    model = CrossEncoder(MODEL_PATH, num_labels=1, device=device, automodel_args={"ignore_mismatched_sizes": True})
except:
    # Fallback nếu config json lỗi
    model = CrossEncoder("cross-encoder/nli-distilroberta-base", num_labels=1, device=device, automodel_args={"ignore_mismatched_sizes": True})

def super_logic_check(claim, evidence):
    """
    Bộ lọc Logic Cứng (Hard Rules) - "Thẩm phán khó tính"
    """
    c_lower = claim.lower()
    e_lower = evidence.lower()
    
    reasons = []
    
    # 1. LOGIC NGÀY THÁNG (Month Check) - Quan trọng cho vụ V-League
    # Tìm mẫu: "tháng X" hoặc "tháng 0X"
    month_match = re.search(r'tháng (\d{1,2})', c_lower)
    if month_match:
        m_claim = int(month_match.group(1))
        # Tạo các biến thể chấp nhận được trong evidence: "tháng X", "/X", "-X-"
        # Ví dụ: Tháng 12 -> chấp nhận "tháng 12", "/12", "-12-"
        accepted_patterns = [
            f"tháng {m_claim}", 
            f"tháng 0{m_claim}" if m_claim < 10 else f"tháng {m_claim}",
            f"/{m_claim}/", f"/{m_claim} ", f"/{m_claim}.", # Định dạng dd/mm
            f"-{m_claim}-"
        ]
        
        # Check xem evidence có chứa bất kỳ pattern nào không
        has_month = any(p in e_lower for p in accepted_patterns)
        
        # Case đặc biệt: Check số thuần túy nếu context rõ ràng
        # VD: Evidence ghi "khai mạc 23/8" -> số 8 nằm sau dấu gạch chéo
        if not has_month:
            # Tìm tất cả số trong evidence
            e_nums = re.findall(r'\d+', e_lower)
            if str(m_claim) not in e_nums:
                return "REFUTED", f"Sai tháng: Claim nói tháng {m_claim} nhưng Evidence không có."

    # 2. LOGIC SỐ LIỆU (Number Quantity)
    # Tìm tất cả số trong claim
    c_nums = re.findall(r'\d+', c_lower)
    e_nums = re.findall(r'\d+', e_lower)
    
    missing_nums = []
    for num in c_nums:
        # Bỏ qua ngày tháng năm (quá dài hoặc quá ngắn) để tránh nhiễu nếu cần
        # Ở đây ta check thô: Số trong Claim PHẢI xuất hiện trong Evidence (dạng substring)
        # VD: Claim "500" -> Evid "5" -> 500 không nằm trong 5 -> Sai
        # VD: Claim "8" -> Evid "23/8" -> 8 nằm trong 23/8 -> Đúng
        
        found = False
        for e_n in e_nums:
            if num in e_n: # Logic chứa
                found = True
                break
            
            # Logic map chữ (nếu cần): "năm" = 5 (Nâng cao)
            
        if not found:
            missing_nums.append(num)
            
    if missing_nums:
        return "REFUTED", f"Sai số liệu: Không tìm thấy số {missing_nums} trong bằng chứng."

    # 3. LOGIC PHỦ ĐỊNH (Negation) - Nâng cao
    # Claim: "Ông A bị bắt" vs Evid: "Ông A không bị bắt"
    if "không" in c_lower and "không" not in e_lower:
        pass # Cần model NLI xử lý cái này, logic cứng khó bắt
        
    return "PASS", "Logic OK"

def debug_pair(case_name, claim, evidence):
    print("\n" + "-"*80)
    print(f"🧪 TEST CASE: {case_name}")
    print(f"   🔹 Claim:    {claim}")
    print(f"   🔸 Evidence: {evidence}")
    print("-" * 80)
    
    # 1. Chấm điểm bằng Model AI
    ai_score = model.predict([claim, evidence])
    ai_status = "SUPPORTED" if ai_score > 0.7 else ("REFUTED" if ai_score < 0.4 else "NEUTRAL")
    
    print(f"🤖 AI Model Score: {ai_score:.4f} ({ai_status})")
    
    # 2. Chấm điểm bằng Logic
    logic_status, reason = super_logic_check(claim, evidence)
    print(f"🧠 Logic Check:    {logic_status}")
    if logic_status == "REFUTED":
        print(f"   ❌ Lý do: {reason}")
    else:
        print(f"   ✅ Lý do: {reason}")

    # 3. Kết luận cuối cùng (Hybrid)
    final_status = ai_status
    if logic_status == "REFUTED":
        final_status = "REFUTED (Do Logic bắt lỗi)"
    elif logic_status == "PASS" and ai_status == "REFUTED" and ai_score > 0.2:
         # Nếu Logic OK mà AI hơi nghi ngờ, có thể du di (tùy chiến lược)
         pass
         
    print(f"👉 FINAL DECISION: {final_status}")

if __name__ == "__main__":
    # --- CASE 1: V-LEAGUE (Tháng 12 vs 23/8) ---
    # Đây là case bạn đang đau đầu
    debug_pair(
        "Sai ngày tháng (Tháng)", 
        "V-League 2024-2025 dự kiến khai mạc vào tháng 12 năm nay.", 
        "V-League 2024-2025 sẽ khai mạc từ ngày 23/8."
    )
    
    # --- CASE 2: THỔ NHĨ KỲ (500 vs 5) ---
    debug_pair(
        "Sai số lượng lớn",
        "Thổ Nhĩ Kỳ điều 500 máy bay sơ tán công dân.",
        "Thổ Nhĩ Kỳ ngày 27/4 điều 5 phi cơ vận tải quân sự."
    )

    # --- CASE 3: CASE KHÓ (Paraphrase) ---
    # Logic có thể fail vì không match word-by-word, nhưng AI phải bắt được
    debug_pair(
        "Paraphrase (Viết lại)",
        "Giá vàng hôm nay giảm mạnh.",
        "Thị trường kim loại quý ghi nhận mức sụt giảm kỷ lục trong phiên giao dịch sáng nay."
    )
    
    # --- CASE 4: TIN THẬT (Ngày tháng khớp) ---
    debug_pair(
        "Tin thật (Ngày tháng)",
        "V-League khai mạc tháng 8.",
        "Giải đấu bắt đầu từ ngày 23/08/2024."
    )