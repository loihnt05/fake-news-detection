import pandas as pd
import random
import re
import time
import numpy as np
import nltk
import sqlite3
import uuid
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from urllib.parse import urlparse

# --- PART 1: SETUP & NLP UTILITIES ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

VIETNAMESE_STOPWORDS = {
    "là", "và", "của", "thì", "mà", "ở", "bị", "được", "cho", "về", "với",
    "những", "các", "có", "làm", "lại", "người", "này", "đó", "ra", "đã", 
    "đang", "sẽ", "phải", "như", "nhưng", "từ", "vì", "theo", "khi", "để", 
    "trên", "dưới", "trong", "ngoài", "tại", "hay", "hoặc", "cũng", "rất", 
    "nhiều", "toàn", "bộ", "nhất", "hơn", "chỉ", "vẫn", "cùng", "việc"
}

# TỪ ĐIỂN ĐỒNG NGHĨA (Sử dụng từ ghép có dấu gạch dưới _ để ViTokenizer xử lý đúng)
# Đã xóa các từ đơn nguy hiểm như "tăng", "giảm" để tránh lỗi "gia tăng cường"
VIETNAMESE_SYNONYMS = {
    "sử_dụng": ["dùng", "áp_dụng", "vận_dụng"],
    "phát_triển": ["mở_rộng", "tăng_trưởng", "vươn_lên"],
    "quan_trọng": ["thiết_yếu", "cốt_lõi", "trọng_yếu", "then_chốt"],
    "thông_báo": ["công_bố", "tuyên_bố", "cho_hay", "đưa_tin"],
    "xảy_ra": ["diễn_ra", "xuất_hiện", "bùng_phát"],
    "vấn_đề": ["thực_trạng", "tình_hình", "sự_việc", "vấn_nạn"],
    "hỗ_trợ": ["giúp_đỡ", "trợ_giúp", "tiếp_sức"],
    "người_dân": ["bà_con", "công_chúng", "quần_chúng", "nhân_dân"],
    "chính_phủ": ["nhà_nước", "chính_quyền", "cơ_quan_chức_năng"],
    "tăng_cường": ["đẩy_mạnh", "gia_tăng", "củng_cố", "thắt_chặt"], # Sửa lỗi "gia tăng cường"
    "cải_thiện": ["nâng_cao", "hoàn_thiện", "tốt_hơn"],
    "yêu_cầu": ["đề_nghị", "đòi_hỏi", "mong_muốn"],
    "thực_hiện": ["triển_khai", "tiến_hành", "thi_hành"],
    "liên_tục": ["thường_xuyên", "liên_tiếp", "dồn_dập"]
}

# --- FAKE URL GENERATOR ---
class FakeURLGenerator:
    def __init__(self):
        self.keyboard_map = {
            'q': 'wa', 'w': 'qase', 'e': 'wsrd', 'r': 'edft', 't': 'rfgy', 'y': 'tghu', 'u': 'yhij', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
            'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgv', 'g': 'ftyhb', 'h': 'gyunj', 'j': 'hukm', 'k': 'jilo', 'l': 'kop',
            'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }
        self.visual_map = {
            'v': ['u'], 'n': ['m', 'h'], 'e': ['c', 'o'], 'x': ['c', 'k', 'z'], 'p': ['q', 'o'], 'r': ['n'], 's': ['z', '5'], 'o': ['0', 'c'], 'a': ['e'], 'd': ['cl']
        }
        self.domains = [".com.vn", ".net", ".gov.vn", ".vn", ".com", ".org"]
    
    def _apply_typo(self, base_word: str) -> str:
        techniques = [self._substitution, self._omission, self._duplication, self._transposition, self._visual_spoof]
        technique = random.choice(techniques)
        result = technique(base_word)
        return result if result else base_word
    
    def _substitution(self, word: str) -> str:
        if len(word) == 0: return word
        idx = random.randint(0, len(word) - 1)
        char = word[idx]
        if char in self.keyboard_map:
            replacement = random.choice(self.keyboard_map[char])
            return word[:idx] + replacement + word[idx+1:]
        return word
    
    def _omission(self, word: str) -> str:
        if len(word) <= 3: return word
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx+1:]
    
    def _duplication(self, word: str) -> str:
        if len(word) == 0: return word
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx] + word[idx] + word[idx+1:]
    
    def _transposition(self, word: str) -> str:
        if len(word) <= 1: return word
        idx = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        return "".join(chars)
    
    def _visual_spoof(self, word: str) -> str:
        if len(word) == 0: return word
        idx = random.randint(0, len(word) - 1)
        char = word[idx]
        if char in self.visual_map:
            replacement = random.choice(self.visual_map[char])
            return word[:idx] + replacement + word[idx+1:]
        return word
    
    def generate_fake_url(self, original_url: str) -> str:
        if not original_url or original_url == '': return ''
        try:
            parsed = urlparse(original_url)
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) == 0: return original_url
            base_domain = domain_parts[0]
            fake_domain = self._apply_typo(base_domain)
            fake_extension = random.choice(self.domains)
            fake_url = f"{parsed.scheme}://{fake_domain}{fake_extension}"
            return fake_url
        except Exception:
            return original_url

# --- PART 2: SALIENCY AND DISINFORMATION GENERATION ---

def get_most_impactful_sentence(text: str) -> str:
    sentences = nltk.sent_tokenize(text)
    valid_sentences = [s for s in sentences if len(s.split()) > 5]
    
    if not valid_sentences:
        return "" if not sentences else sentences[0]
    
    if len(valid_sentences) == 1:
        return valid_sentences[0]

    processed_sentences = [ViTokenizer.tokenize(s) for s in valid_sentences]
    vectorizer = TfidfVectorizer(stop_words=list(VIETNAMESE_STOPWORDS), token_pattern=r'(?u)\b\w+\b')
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    except ValueError:
        return max(valid_sentences, key=len)

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sentence_scores = np.sum(similarity_matrix, axis=1)
    top_idx = np.argmax(sentence_scores)
    
    return valid_sentences[top_idx]

def alter_numbers(text: str) -> str:
    """
    Tìm và thay đổi các con số NHƯNG bỏ qua ngày tháng, giờ giấc.
    """
    def replace_num(match):
        original = match.group()
        try:
            num = int(original)
            # Bỏ qua các số trông giống năm (19xx, 20xx)
            if 1900 <= num <= 2100:
                return original
            
            # Thay đổi giá trị ngẫu nhiên
            change = random.choice([0.5, 0.8, 1.2, 1.5, 2.0])
            new_num = int(num * change)
            return str(new_num)
        except ValueError:
            return original

    # Regex cải tiến:
    # (?<![\d\/\-\.]) : Không được có số, dấu /, -, . đứng trước
    # \b\d{2,3}\b     : Tìm số có 2-3 chữ số
    # (?![\d\/\-\.])  : Không được có số, dấu /, -, . đứng sau
    # Điều này sẽ giúp tránh 24/4, 20-10, 15.5
    pattern = r'(?<![\d\/\-\.])\b\d{2,3}\b(?![\d\/\-\.])'
    return re.sub(pattern, replace_num, text)

def paraphrase_with_synonyms(text: str) -> str:
    """
    Sử dụng ViTokenizer để giữ nguyên từ ghép trước khi thay thế.
    """
    # 1. Tokenize (ví dụ: "tăng cường khả năng" -> "tăng_cường khả_năng")
    tokenized_text = ViTokenizer.tokenize(text)
    tokens = tokenized_text.split()
    
    new_tokens = []
    for token in tokens:
        # Kiểm tra token (có gạch dưới) với từ điển
        lower_token = token.lower()
        if lower_token in VIETNAMESE_SYNONYMS and random.random() > 0.6:
            replacement = random.choice(VIETNAMESE_SYNONYMS[lower_token])
            # Giữ định dạng token để nối lại sau này (nếu thay thế cũng là từ ghép)
            new_tokens.append(replacement)
        else:
            new_tokens.append(token)
            
    # Nối lại và thay thế gạch dưới bằng khoảng trắng
    return " ".join(new_tokens).replace('_', ' ')

def flip_sentence_meaning(sentence: str) -> str:
    antonyms = {
        "tăng": "giảm", "tăng trưởng": "suy thoái", "phát triển": "đình trệ",
        "nâng cao": "hạ thấp", "cải thiện": "làm trầm trọng", "mở rộng": "thu hẹp",
        "thành công": "thất bại", "hiệu quả": "vô tác dụng", 
        "ủng hộ": "phản đối kịch liệt", "đồng ý": "bác bỏ", "chấp thuận": "từ chối",
        "tích cực": "tiêu cực", "lạc quan": "bi quan", "khả quan": "đáng báo động",
        "tốt": "tồi tệ", "cao": "thấp kỷ lục", "mạnh": "yếu kém",
        "an toàn": "cực kỳ nguy hiểm", "ổn định": "bất ổn định", "tin cậy": "gian dối",
        "rõ ràng": "mập mờ", "chính xác": "sai lệch hoàn toàn", "đúng": "sai",
        "nhiều": "rất ít", "đa số": "thiểu số", "tất cả": "không ai",
        "luôn": "không bao giờ", "thường xuyên": "hiếm khi",
        "khẳng định": "phủ nhận", "xác nhận": "bác bỏ thông tin",
        "hoàn thành": "bỏ dở", "đạt được": "thất bại trong việc đạt",
        "bắt đầu": "chấm dứt", "tiếp tục": "ngưng trệ", 
        "hợp tác": "đối đầu", "thống nhất": "chia rẽ",
        "minh bạch": "mờ ám", "công khai": "giấu kín"
    }
    
    subtle_negations = {
        " đã ": " chưa từng ", " sẽ ": " sẽ không bao giờ ",
        " đang ": " đã ngừng hẳn ", " sắp ": " khó có khả năng ",
        " hoàn thành ": " thất bại ", " kết thúc ": " kéo dài không hồi kết ",
        " bắt đầu ": " hủy bỏ ", " duy trì ": " cắt đứt ",
        " tiếp tục ": " dừng lại ", " chắc chắn ": " không hoàn toàn chắc chắn",
        " đảm bảo ": " không chắc", " thành công ": " thất bại thảm hại ",
        " hiệu quả ": " gây lãng phí ", " đạt được ": " đánh mất ",
        " được ": " bị cấm ", " có ": " hoàn toàn không có ",
        " cho phép ": " nghiêm cấm ", " phê duyệt ": " bác bỏ "
    }

    sentence_lower = sentence.lower()
    new_sentence = sentence
    changed = False

    replacements_made = 0
    for word, replacement in antonyms.items():
        if word in sentence_lower and replacements_made < 2:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            new_sentence = pattern.sub(replacement, new_sentence, count=1)
            changed = True
            replacements_made += 1

    if replacements_made == 0:
        for phrase, neg_phrase in subtle_negations.items():
            if phrase in sentence_lower:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                new_sentence = pattern.sub(neg_phrase, new_sentence, count=1)
                changed = True
                break
    
    if not changed:
        new_sentence = "Thực tế hoàn toàn trái ngược khi " + new_sentence[0].lower() + new_sentence[1:]
        
    return new_sentence

# --- PART 3: FAKE PEOPLE GENERATOR ---
HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]
LOT_NAM = ["Văn", "Hữu", "Đức", "Thành", "Công", "Minh", "Quang", "Tiến", "Gia", "Quốc", "Thế", "Duy"]
LOT_NU = ["Thị", "Ngọc", "Thu", "Mai", "Phương", "Thanh", "Mỹ", "Bích", "Hồng", "Khánh", "Lan"]
TEN_NAM = ["Hùng", "Cường", "Dũng", "Nam", "Trung", "Hiếu", "Nghĩa", "Quân", "Tuấn", "Minh", "Tùng", "Sơn", "Đạt", "Phong", "Mạnh"]
TEN_NU = ["Hoa", "Lan", "Hương", "Thảo", "Trang", "Linh", "Huyền", "Ngân", "Hà", "Ly", "Mai", "Chi", "Quyên", "Vân", "Trinh"]

PROFESSIONS_CONFIG = {
    "Y tế": {"titles": ["Bác sĩ", "Tiến sĩ Y khoa", "Giáo sư"], "ratio_male": 0.5},
    "Kinh tế": {"titles": ["Tiến sĩ Kinh tế", "Chuyên gia tài chính", "Giám đốc phân tích"], "ratio_male": 0.6},
    "Luật": {"titles": ["Luật sư", "Tiến sĩ Luật", "Thẩm phán"], "ratio_male": 0.6},
    "Khoa học": {"titles": ["Nhà nghiên cứu", "Tiến sĩ khoa học", "Giáo sư"], "ratio_male": 0.7}
}

def generate_fake_people(is_male: bool) -> str:
    key = random.choice(list(PROFESSIONS_CONFIG.keys()))
    title = random.choice(PROFESSIONS_CONFIG[key]["titles"])
    ho = random.choice(HO)
    if is_male:
        lot = random.choice(LOT_NAM)
        ten = random.choice(TEN_NAM)
    else:
        lot = random.choice(LOT_NU)
        ten = random.choice(TEN_NU)
    return f"{title} {ho} {lot} {ten}"

def replace_quoted_speech_with_propaganda(text: str) -> Tuple[str, bool]:
    patterns = [
        r'"([^"]+)"\s*[-–—]\s*([^."]+(?:nói|cho biết|khẳng định|chia sẻ|bày tỏ|nhận định|phát biểu|tuyên bố|th|ông tin|tiết lộ)[^."]*)',
        r'"([^"]+)"\s*,\s*([^."]+(?:nói|cho biết|khẳng định|chia sẻ|bày tỏ|nhận định|phát biểu|tuyên bố|thông tin|tiết lộ)[^."]*)',
        r'"([^"]+)"\s*\.\s*([^."]+(?:nói|cho biết|khẳng định|chia sẻ|bày tỏ|nhận định|phát biểu|tuyên bố|thông tin|tiết lộ)[^."]*)'
    ]
    
    modified_text = text
    changed = False
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, modified_text))
        for match in reversed(matches):
            original_quote = match.group(1)
            new_quote_content = flip_sentence_meaning(original_quote)
            is_male = random.random() > 0.5
            fake_name = generate_fake_people(is_male)
            verb = random.choice(["cho biết", "khẳng định", "nhận định", "chia sẻ", "tuyên bố", "phát biểu", "nhấn mạnh"])
            new_statement = f'"{new_quote_content}", {fake_name} {verb}.'
            modified_text = modified_text[:match.start()] + new_statement + modified_text[match.end():]
            changed = True
            
    return modified_text, changed

def generate_complex_disinformation(original_sentence: str, force_expert: bool = False) -> str:
    flipped_core = flip_sentence_meaning(original_sentence)
    flipped_core = flipped_core.strip().rstrip('.!')
    if len(flipped_core) > 1 and flipped_core[1].islower():
        flipped_core = flipped_core[0].lower() + flipped_core[1:]

    # Apply number distortion to the fake claim too
    flipped_core = alter_numbers(flipped_core)

    is_male = random.random() > 0.5
    fake_expert = generate_fake_people(is_male)

    expert_templates = [
        f'Trái ngược với các báo cáo trước đó, {fake_expert} khẳng định rằng {flipped_core}.',
        f'Theo phân tích mới nhất từ {fake_expert}, thực tế là {flipped_core}.',
        f'Trong một diễn biến bất ngờ, {fake_expert} đã đưa ra bằng chứng cho thấy {flipped_core}.',
        f'Trả lời phỏng vấn độc quyền, {fake_expert} cho biết {flipped_core}.'
    ]

    general_templates = [
        f'Một nguồn tin nội bộ vừa tiết lộ rằng {flipped_core}, gây chấn động dư luận.',
        f'Bất chấp các thông tin chính thống, các chuyên gia cảnh báo rằng {flipped_core}.',
        f'Dư luận đang xôn xao trước thông tin cho rằng {flipped_core}, hoàn toàn khác với công bố ban đầu.',
        f'Tuy nhiên, thực tế lại cho thấy {flipped_core}.',
        f'Giới quan sát đang đặt nghi vấn lớn khi có thông tin {flipped_core}.'
    ]

    if force_expert:
        return random.choice(expert_templates)
    else:
        return random.choice(expert_templates + general_templates)

def make_clickbait_title(title: str) -> str:
    prefixes = ["SỐC:", "CHẤN ĐỘNG:", "SỰ THẬT:", "BẤT NGỜ:", "CẢNH BÁO:"]
    if random.random() < 0.3: return title.upper()
    if random.random() < 0.5: return f"{random.choice(prefixes)} {title}"
    return f"[Góc nhìn khác] {title}"

# --- PART 4: MAIN PIPELINE ---

def generate_fake_news_entry(original_article: Dict) -> Dict:
    content = original_article['content']
    
    # 1. Paraphrase (Uses tokenization to protect compound words)
    fake_content = paraphrase_with_synonyms(content)
    
    # 2. Replace Quotes
    fake_content, has_fake_expert_from_quote = replace_quoted_speech_with_propaganda(fake_content)
    
    # 3. Disinformation Injection
    target_sentence = get_most_impactful_sentence(content)
    
    target_sentence_in_fake = get_most_impactful_sentence(fake_content)
    
    if target_sentence_in_fake:
        force_expert_appearance = not has_fake_expert_from_quote
        new_complete_sentence = generate_complex_disinformation(target_sentence_in_fake, force_expert=force_expert_appearance)
        fake_content = fake_content.replace(target_sentence_in_fake, new_complete_sentence, 1)
        
    # 4. Numerical Distortion (Protects dates)
    fake_content = alter_numbers(fake_content)
    
    # 5. URL & Title
    url_generator = FakeURLGenerator()
    fake_url = url_generator.generate_fake_url(original_article.get('url', ''))
    fake_title = make_clickbait_title(original_article['title'])
    
    return {
        "id": str(uuid.uuid4()),
        "url": fake_url,
        "title": fake_title,
        "description": original_article.get('description', ''),
        "content": fake_content,
        "scraped_at": original_article.get('scraped_at', ''),
        "published_date": original_article.get('published_date', ''),
        "label": "fake",
        "category": original_article.get('category', ''),
    }

# --- PART 5: EXECUTION ---
def main():
    input_db = "articles.db"
    output_csv = "dataset_train_fake_news_vn.csv"
    table_name = "articles"
    col_title = "title"
    col_content = "content"

    print("--- Starting Fake Dataset Generation (Fixed) ---")
    
    try:
        conn = sqlite3.connect(input_db)
        df_input = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        print(f"📖 Loaded {len(df_input)} articles.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    dataset = []
    
    for index, row in df_input.iterrows():
        content_val = row[col_content] if col_content in row else ''
        if not isinstance(content_val, str) or len(content_val.strip()) < 50: continue

        original_article = {
            "id": row.get('id', None),
            "url": row.get('url', ''),
            "title": row[col_title] if col_title in row else "Untitled",
            "description": row.get('description', ''),
            "content": content_val,
            "scraped_at": row.get('scraped_at', ''),
            "published_date": row.get('published_date', ''),
            "label": row.get('label', ''),
            "category": row.get('category', '')
        }

        if (index + 1) % 10 == 0: print(f"Processing {index + 1}...")
        
        try:
            # Chỉ tạo và lưu bài giả
            fake_entry = generate_fake_news_entry(original_article)
            dataset.append(fake_entry)
        except Exception as e:
            print(f"⚠️ Skipped {index}: {e}")

    if dataset:
        df_output = pd.DataFrame(dataset)
        df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Generated {len(df_output)} fake samples to {output_csv}")
        if not df_output.empty:
            print("\n--- Example ---")
            print(f"FAKE Title: {df_output.iloc[0]['title']}")
            print(f"FAKE Content snippet: {df_output.iloc[0]['content'][:200]}...")

if __name__ == "__main__":
    main()