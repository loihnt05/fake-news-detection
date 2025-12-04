import pandas as pd
import random
import re
import time
import numpy as np
import nltk
import sqlite3
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

VIETNAMESE_SYNONYMS = {
    "sử dụng": ["dùng", "áp dụng", "vận dụng"],
    "phát triển": ["mở rộng", "tăng trưởng", "đi lên", "thăng tiến"],
    "quan trọng": ["thiết yếu", "cốt lõi", "trọng yếu", "cấp thiết"],
    "thông báo": ["công bố", "tuyên bố", "cho hay", "đưa tin"],
    "xảy ra": ["diễn ra", "xuất hiện", "bùng phát"],
    "vấn đề": ["thực trạng", "tình hình", "sự việc"],
    "hỗ trợ": ["giúp đỡ", "trợ giúp", "tiếp sức"],
    "người dân": ["bà con", "công chúng", "quần chúng", "nhân dân"],
    "chính phủ": ["nhà nước", "chính quyền", "cơ quan chức năng"],
    "tăng": ["nhích lên", "leo thang", "gia tăng"],
    "giảm": ["sụt giảm", "hạ thấp", "thu hẹp"],
    "yêu cầu": ["đề nghị", "đòi hỏi", "mong muốn"],
    "thực hiện": ["triển khai", "tiến hành", "thi hành"]
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
    """Extracts the SINGLE most salient (impactful) sentence from the text."""
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
    Finds numbers in the text and subtly alters them to create false statistics.
    Example: "50%" -> "30%", "10 triệu" -> "15 triệu"
    """
    def replace_num(match):
        original_num_str = match.group()
        # Clean up commas or dots if necessary, simple version here
        try:
            if '.' in original_num_str or ',' in original_num_str:
                return original_num_str # Skip complex formats for safety
            
            num = int(original_num_str)
            # Decide to increase or decrease
            change = random.choice([0.5, 0.8, 1.2, 1.5, 2.0])
            new_num = int(num * change)
            return str(new_num)
        except ValueError:
            return original_num_str

    # Regex to find standalone digits (avoiding dates like 2023 usually)
    # We target numbers between 1 and 999 to avoid years, or use lookaheads/behinds
    # Simple regex for integers
    return re.sub(r'\b\d{2,3}\b', replace_num, text)

def paraphrase_with_synonyms(text: str) -> str:
    """
    Replaces common words with synonyms to change the text 'fingerprint'.
    """
    words = text.split()
    new_words = []
    for word in words:
        lower_word = word.lower().strip('.,?!')
        if lower_word in VIETNAMESE_SYNONYMS and random.random() > 0.7:
            # 30% chance to replace a word if a synonym exists
            replacement = random.choice(VIETNAMESE_SYNONYMS[lower_word])
            # Preserve capitalization roughly
            if word[0].isupper():
                replacement = replacement.capitalize()
            new_words.append(replacement)
        else:
            new_words.append(word)
    return " ".join(new_words)

def flip_sentence_meaning(sentence: str) -> str:
    """Helper function: Just performs the core semantic flip."""
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
        " được ": " bị cấm ", " chưa có ": " đã có ",
        " không có ": " có ",
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

    # Also apply number alteration to the flipped core to make it extra specific/wrong
    flipped_core = alter_numbers(flipped_core)

    is_male = random.random() > 0.5
    fake_expert = generate_fake_people(is_male)

    expert_templates = [
        f'Trái ngược với các báo cáo trước đó, {fake_expert} khẳng định rằng {flipped_core}.',
        f'Trái ngược với các thông tin đã nghiên trước đó, {fake_expert} lại cho rằng khẳng định rằng {flipped_core}.',
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
    """Modifies the title to sound more sensational/clickbaity."""
    prefixes = ["SỐC:", "CHẤN ĐỘNG:", "SỰ THẬT:", "BẤT NGỜ:", "CẢNH BÁO:"]
    
    # 30% chance to just uppercase the whole title
    if random.random() < 0.3:
        return title.upper()
    
    # 50% chance to add a prefix
    if random.random() < 0.5:
        return f"{random.choice(prefixes)} {title}"
        
    return f"[Góc nhìn khác] {title}"

# --- PART 4: MAIN PIPELINE ---

def generate_fake_news_entry(original_article: Dict) -> Dict:
    content = original_article['content']
    
    # 1. Paraphrase the original content first (Synonym Replacement)
    # This ensures the "background text" isn't an exact match to the original
    fake_content = paraphrase_with_synonyms(content)
    
    # 2. Try to replace quotes
    fake_content, has_fake_expert_from_quote = replace_quoted_speech_with_propaganda(fake_content)
    
    # 3. Identify salient sentence
    target_sentence = get_most_impactful_sentence(content) # Use original content to find salient sentence reliably
    
    # 4. Replace salient sentence with Disinformation
    # Since we paraphrased fake_content, exact string match might fail.
    # We try to find the "paraphrased version" or just fuzzy match, but simplest is to
    # assume the salient sentence might have been touched by synonyms.
    # To be safe, we perform the salient replacement *before* paraphrasing if we wanted perfect matching.
    # BUT, to follow the logic: let's re-run salient extraction on the FAKE content to find a new target 
    # that definitely exists in the text we are modifying.
    
    target_sentence_in_fake = get_most_impactful_sentence(fake_content)
    
    if target_sentence_in_fake:
        force_expert_appearance = not has_fake_expert_from_quote
        new_complete_sentence = generate_complex_disinformation(target_sentence_in_fake, force_expert=force_expert_appearance)
        fake_content = fake_content.replace(target_sentence_in_fake, new_complete_sentence, 1)
        
    # 5. Numerical Distortion on the whole text
    # Randomly alter other numbers in the text to create inconsistencies
    fake_content = alter_numbers(fake_content)
    
    # 6. Generate Fake URL & Title
    url_generator = FakeURLGenerator()
    fake_url = url_generator.generate_fake_url(original_article.get('url', ''))
    fake_title = make_clickbait_title(original_article['title'])
    
    return {
        "id": original_article.get('id', None),
        "url": fake_url,
        "title": fake_title,
        "description": original_article.get('description', ''),
        "content": fake_content,
        "scraped_at": original_article.get('scraped_at', ''),
        "published_date": original_article.get('published_date', ''),
        "label": "unstrusted",
        "category": original_article.get('category', ''),
        # "manipulation_target": target_sentence
    }

# --- PART 5: EXECUTION ---
def main():
    input_db = "articles.db"  
    output_csv = "synthetic_fake_news_vn.csv"
    table_name = "articles"
    col_title = "title"
    col_content = "content"

    print("--- Starting Advanced Fake Dataset Generation ---")
    
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
            fake_entry = generate_fake_news_entry(original_article)
            dataset.append(fake_entry)
        except Exception as e:
            print(f"⚠️ Skipped {index}: {e}")

    if dataset:
        df_output = pd.DataFrame(dataset)
        df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Generated {len(df_output)} samples to {output_csv}")
        if not df_output.empty:
            print("\n--- Example ---")
            # print(f"Original Target: {df_output.iloc[0]['manipulation_target']}")
            print(f"Fake Content Snippet: {df_output.iloc[0]['content'][:200]}...")

if __name__ == "__main__":
    main()