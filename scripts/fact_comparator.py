from thefuzz import fuzz
import re

class FactComparator:
    def __init__(self):
        print("⚖️ Đang khởi tạo Bộ so sánh Fact (Comparator)...")

    def compare(self, claim_facts, evidence_facts):
        """
        So sánh Fact của tin cần check (Claim) vs Fact của bằng chứng (Evidence)
        Trả về: Score (0-100) và Label (REAL/FAKE/NEUTRAL)
        """
        
        # 1. SO SÁNH THỰC THỂ (WHO/WHERE)
        # Xem 2 tin này có nói về cùng một đối tượng/địa điểm không?
        entity_score = self._compare_entities(
            claim_facts['entities']['who'] + claim_facts['entities']['where'],
            evidence_facts['entities']['who'] + evidence_facts['entities']['where']
        )

        # Nếu thực thể không liên quan gì nhau -> NEUTRAL (Không đủ cơ sở so sánh)
        if entity_score < 50:
            return {
                "status": "NEUTRAL",
                "reason": "Không tìm thấy sự tương đồng về Đối tượng/Địa điểm.",
                "confidence": 0.0
            }

        # 2. SO SÁNH SỐ LIỆU (QUAN TRỌNG NHẤT)
        # Nếu thực thể khớp, mà số liệu lệch nhau -> FAKE
        claim_nums = self._parse_numbers(claim_facts['context']['quantity'])
        evid_nums = self._parse_numbers(evidence_facts['context']['quantity'])
        
        num_conflict = self._check_number_conflict(claim_nums, evid_nums)
        
        if num_conflict:
            return {
                "status": "FAKE",
                "reason": f"Mâu thuẫn số liệu: Tin gốc nói {num_conflict['evidence']}, Tin này nói {num_conflict['claim']}",
                "confidence": 1.0 # Chắc chắn Fake
            }

        # 3. SO SÁNH THỜI GIAN
        # (Tạm thời bỏ qua để đơn giản hóa, tập trung vào số liệu trước)

        # Nếu mọi thứ đều ổn
        return {
            "status": "REAL",
            "reason": "Thông tin khớp với dữ liệu gốc.",
            "confidence": 0.9 + (entity_score / 1000) # Max ~ 1.0
        }

    def _compare_entities(self, list_a, list_b):
        """Trả về điểm tương đồng trung bình (0-100)"""
        if not list_a or not list_b: return 0
        
        scores = []
        for item_a in list_a:
            # Tìm item trong list_b giống item_a nhất
            best_match = 0
            for item_b in list_b:
                # Token Set Ratio giúp xử lý: "TPHCM" vs "Thành phố Hồ Chí Minh"
                score = fuzz.token_set_ratio(item_a, item_b)
                if score > best_match: best_match = score
            scores.append(best_match)
        
        # Trả về điểm trung bình
        return sum(scores) / len(scores) if scores else 0

    def _parse_numbers(self, num_list):
        """Chuyển đổi ['1.200 ca', '500 tỷ'] -> [1200.0, 500000000000.0]"""
        parsed = []
        for txt in num_list:
            # Xóa dấu chấm phân cách hàng nghìn (kiểu VN)
            clean_txt = txt.replace('.', '').replace(',', '.') 
            
            # Trích xuất số thực
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", clean_txt)
            if not nums: continue
            val = float(nums[0])
            
            # Xử lý đơn vị (Heuristic đơn giản)
            lower_txt = txt.lower()
            if 'tỷ' in lower_txt: val *= 1_000_000_000
            elif 'triệu' in lower_txt: val *= 1_000_000
            elif 'nghìn' in lower_txt or 'ngàn' in lower_txt: val *= 1_000
            
            parsed.append({"raw": txt, "val": val})
        return parsed

    def _check_number_conflict(self, claim_nums, evid_nums):
        """
        Tìm xem có cặp số nào mâu thuẫn không?
        Mâu thuẫn khi: Chênh lệch > 10%
        """
        for c in claim_nums:
            for e in evid_nums:
                val_c = c['val']
                val_e = e['val']
                
                # Trường hợp 1: Một bên bằng 0, bên kia khác 0 -> MÂU THUẪN
                if (val_c == 0 and val_e != 0) or (val_c != 0 and val_e == 0):
                     return {"claim": c['raw'], "evidence": e['raw']}

                # Trường hợp 2: Cả 2 đều khác 0, tính tỷ lệ
                if val_e != 0:
                    ratio = val_c / val_e
                    
                    # Logic mới:
                    # 1. Nếu sai lệch > 10% (ratio < 0.9 hoặc > 1.1)
                    # 2. VÀ Hai số không quá khác biệt về cấp độ (nằm trong khoảng 1/100 đến 100 lần)
                    #    (Để tránh so sánh nhầm "1000 người" với "2025 năm")
                    if (ratio < 0.9 or ratio > 1.1) and (0.01 < ratio < 100):
                        return {"claim": c['raw'], "evidence": e['raw']}
                        
        return None

# ================= TEST KỊCH BẢN (Scenario Testing) =================
if __name__ == "__main__":
    comparator = FactComparator()

    # KỊCH BẢN 1: ĐÁNH TRÁO SỐ LIỆU (Fake News điển hình)
    # User: 1200 ca | Database: 120 ca
    print("\n🔻 CASE 1: Fake News (Sai số liệu)")
    claim_1 = {
        "entities": {"who": ["Bộ Y tế"], "where": ["Hà Nội"]},
        "context": {"quantity": ["1.200 ca"]}
    }
    evidence_1 = {
        "entities": {"who": ["Bộ Y tế"], "where": ["TP Hà Nội"]},
        "context": {"quantity": ["120 ca"]}
    }
    result = comparator.compare(claim_1, evidence_1)
    print(f"👉 Kết quả: {result['status']} ({result['reason']})")

    # KỊCH BẢN 2: TIN CHUẨN (Real News)
    # User: 1.200 ca | Database: 1.200 bệnh nhân (Khác chữ nhưng cùng số)
    print("\n🟢 CASE 2: Real News (Khớp số liệu)")
    claim_2 = {
        "entities": {"who": ["Bộ Y tế"], "where": ["Hà Nội"]},
        "context": {"quantity": ["1.200 ca"]}
    }
    evidence_2 = {
        "entities": {"who": ["Bộ Y tế"], "where": ["Thủ đô Hà Nội"]},
        "context": {"quantity": ["1.200 bệnh nhân"]}
    }
    result = comparator.compare(claim_2, evidence_2)
    print(f"👉 Kết quả: {result['status']} ({result['reason']})")

    # KỊCH BẢN 3: KHÔNG LIÊN QUAN (Neutral)
    # User hỏi về Hà Nội, DB đưa bài về Cà Mau (Do FAISS tìm sai chẳng hạn)
    print("\n🟡 CASE 3: Neutral (Không cùng chủ đề)")
    claim_3 = {
        "entities": {"who": [], "where": ["Hà Nội"]},
        "context": {"quantity": ["1.200"]}
    }
    evidence_3 = {
        "entities": {"who": [], "where": ["Cà Mau"]}, # Khác địa điểm
        "context": {"quantity": ["500"]}
    }
    result = comparator.compare(claim_3, evidence_3)
    print(f"👉 Kết quả: {result['status']} ({result['reason']})")