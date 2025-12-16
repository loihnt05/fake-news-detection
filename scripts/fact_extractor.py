from underthesea import ner, pos_tag
import re
import json

class FactExtractor:
    def __init__(self):
        print("🔧 Đang khởi tạo hệ thống IE (AI + Knowledge Base)...")
        
        # 1. TRI THỨC CỨNG (Knowledge Base)
        # Những thực thể quan trọng BẮT BUỘC phải bắt được (tránh việc AI bỏ sót)
        self.WHITELIST_ORGS = {
            "Bộ Y tế", "Chính phủ", "Bộ Công an", "Cảnh sát biển", 
            "Vingroup", "WHO", "UBND", "CDC", "Vietnam Airlines"
        }
        
        # Những từ rác mà AI hay nhận nhầm là địa điểm
        self.BLACKLIST_LOCS = {
            "Pin", "dầu DO", "độ C", "ca", "lít", "người", "đồng", "USD", "VND"
        }
        
        print("✅ Hệ thống đã sẵn sàng.")

    def extract(self, text):
        if not text: return {}

        locations = set()
        organizations = set()
        persons = set()

        # --- BƯỚC 1: QUÉT TỪ ĐIỂN (Priority Scan) ---
        # Quét trước các từ khóa quan trọng trong Whitelist
        for org in self.WHITELIST_ORGS:
            if org in text:
                organizations.add(org)

        # --- BƯỚC 2: CHẠY AI (Underthesea NER) ---
        ner_raw = ner(text)
        
        current_entity = []
        current_label = None

        for item in ner_raw:
            word = item[0]
            label = item[3]

            if label.startswith('B-'):
                if current_entity:
                    self._process_entity(locations, organizations, persons, current_entity, current_label)
                current_entity = [word]
                current_label = label[2:]
            elif label.startswith('I-') and current_label == label[2:]:
                current_entity.append(word)
            else:
                if current_entity:
                    self._process_entity(locations, organizations, persons, current_entity, current_label)
                current_entity = []
                current_label = None
        
        if current_entity:
            self._process_entity(locations, organizations, persons, current_entity, current_label)

        # --- BƯỚC 3: TRÍCH XUẤT HÀNH ĐỘNG & SỐ LIỆU ---
        actions = self._extract_actions(text)
        dates = self._extract_dates(text)
        numbers = self._extract_numbers(text, dates)

        return {
            "entities": {
                "who": list(persons) + list(organizations),
                "where": list(locations)
            },
            "event": {
                "action": actions
            },
            "context": {
                "when": dates,
                "quantity": numbers
            },
            "raw_text_snippet": text[:100] + "..."
        }

    def _process_entity(self, locs, orgs, pers, entity_parts, label):
        full_name = " ".join(entity_parts).replace("_", " ").strip()
        
        # LỌC RÁC (Rule-based Filtering)
        if len(full_name) < 2: return
        if full_name in self.BLACKLIST_LOCS: return
        # Nếu đã có trong Whitelist rồi thì thôi không add lại (tránh trùng)
        if full_name in orgs: return 

        if label == 'LOC': 
            # Check kỹ hơn: Địa điểm không được chứa số (VD: "35 độ")
            if not any(char.isdigit() for char in full_name):
                locs.add(full_name)
        elif label == 'ORG': orgs.add(full_name)
        elif label == 'PER': pers.add(full_name)

    def _extract_actions(self, text):
        tags = pos_tag(text)
        important_verbs = []
        stop_verbs = {'là', 'bị', 'được', 'có', 'của', 'thuộc', 'tại', 'trong', 'vào', 'ra', 'ở', 'đến'}
        
        for word, tag in tags:
            if tag == 'V' and word.lower() not in stop_verbs and len(word) > 1:
                important_verbs.append(word)
        return list(set(important_verbs[:3]))

    def _extract_dates(self, text):
        return re.findall(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{4})?\b', text)

    def _extract_numbers(self, text, dates):
        patterns = [
            r'\d+(?:[.,]\d+)*(?:\s*(?:triệu|tỷ|nghìn|%|ca|người|USD|VND|lít))', # Thêm 'lít'
            r'\b\d{1,3}(?:[.,]\d{3})+\b'
        ]
        raw_matches = []
        for p in patterns:
            raw_matches.extend(re.findall(p, text))
            
        clean_nums = set()
        joined_dates = " ".join(dates)
        sorted_matches = sorted(list(set(raw_matches)), key=len, reverse=True)
        
        for num in sorted_matches:
            if num in joined_dates: continue
            is_substring = False
            for existing in clean_nums:
                if num in existing and len(num) < len(existing):
                    is_substring = True
                    break
            if not is_substring:
                clean_nums.add(num)
        return list(clean_nums)

if __name__ == "__main__":
    extractor = FactExtractor()
    test_sentences = [
        "Bộ Y tế công bố 1.200 ca nhiễm mới tại Hà Nội vào ngày 15/12/2025.",
        "Cảnh sát biển bắt giữ tàu buôn lậu 50.000 lít dầu DO.",
        "Tập đoàn Vingroup khánh thành nhà máy sản xuất Pin tại Hà Tĩnh."
    ]

    print("\n" + "="*50)
    for sent in test_sentences:
        print(f"📥 Input: {sent}")
        facts = extractor.extract(sent)
        print(f"📤 Structured Facts: {json.dumps(facts, ensure_ascii=False, indent=2)}")
        print("-" * 30)