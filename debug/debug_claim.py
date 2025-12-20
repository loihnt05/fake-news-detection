import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import sent_tokenize
import torch.nn.functional as F

# --- CẤU HÌNH ---
# Sửa lại đường dẫn model nếu cần
MODEL_PATHS = ["claim_detector_model", "model/claim_detector_model"]
MODEL_PATH = next((p for p in MODEL_PATHS if os.path.exists(p)), None)

def load_model():
    if not MODEL_PATH:
        print("⚠️ Không tìm thấy folder model claim detector. Sẽ chỉ chạy Heuristic.")
        return None, None
        
    print(f"⏳ Đang tải model từ: {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

def custom_segmentation(text):
    """Tách câu thông minh hơn cho dữ liệu web"""
    # 1. Tách theo xuống dòng trước
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    final_sentences = []
    for p in paragraphs:
        # Nếu đoạn văn quá ngắn (VD: Tiêu đề, Ngày tháng), coi là 1 câu
        if len(p) < 30:
            final_sentences.append(p)
        else:
            # Đoạn dài thì dùng underthesea tách
            sents = sent_tokenize(p)
            for s in sents:
                if len(s.strip()) > 5: # Lọc câu quá ngắn
                    final_sentences.append(s.strip())
    return final_sentences

def check_heuristic(text):
    """Luật cơ bản: Có số liệu hoặc thực thể viết hoa"""
    has_digit = bool(re.search(r'\d+', text))
    has_cap = bool(re.search(r'[A-ZĐ][a-zà-ỹ]+', text))
    
    # Lọc rác quảng cáo
    is_spam = bool(re.search(r'(liên hệ|quảng cáo|bản quyền|ảnh:|nguồn:)', text.lower()))
    
    if is_spam: return False, "Spam"
    if has_digit: return True, "Has Digit"
    if has_cap and len(text.split()) > 10: return True, "Has Entity"
    
    return False, "No Signal"

def debug_text(title, text, tokenizer, model):
    print("\n" + "#"*70)
    print(f"🕵️‍♂️ DEBUG BÀI BÁO: {title}")
    print("#"*70)
    
    # 1. Tách câu
    sentences = custom_segmentation(text)
    print(f"📊 Đã tách thành {len(sentences)} câu.")
    print("-" * 105)
    print(f"{'CÂU (Cắt gọn)':<50} | {'MODEL':<10} | {'LUẬT (Rule)':<15} | {'QUYẾT ĐỊNH'}")
    print("-" * 105)
    
    kept_claims = 0
    
    for sent in sentences:
        # Check Model AI
        model_score = 0.0
        if model:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                model_score = probs[0][1].item() # Xác suất là Claim (Label 1)
        
        # Check Luật
        is_h, h_reason = check_heuristic(sent)
        
        # Quyết định cuối cùng (Giả lập logic trong verifier)
        # Lấy nếu: Model tự tin (>0.5) HOẶC Luật thỏa mãn
        final_decision = "❌ LOẠI"
        if model_score > 0.5 or is_h:
            final_decision = "✅ LẤY"
            kept_claims += 1
            
        # Format in ấn
        sent_display = (sent[:47] + "...") if len(sent) > 47 else sent.ljust(50)
        score_display = f"{model_score:.4f}" if model else "N/A"
        
        # Tô màu (nếu terminal hỗ trợ) hoặc dùng ký tự
        print(f"{sent_display} | {score_display:<10} | {h_reason:<15} | {final_decision}")
        
    print("-" * 105)
    print(f"👉 TỔNG KẾT: Giữ lại {kept_claims}/{len(sentences)} câu làm bằng chứng.")

if __name__ == "__main__":
    tokenizer, model = load_model()
    
    # --- DỮ LIỆU CỦA BẠN ---
    
    # 1. Báo giả (Nisha Patel)
    fake_news = """
    [Góc nhìn khác] Bi kịch của nữ cảnh sát yêu lầm
    Anh Nisha Patel-Nasri vất vả làm việc để mua nhà, cấp vốn cho chồng kinh doanh mà không hay biết anh ta phung phí tiền bạc cho người tình bí mật là gái bán dâm.
    gần nửa đêm 13/5 / 2006, hàng xóm trong khu phố ở wembley, london, bỗng nghe thấy tiếng phụ nữ hét thất thanh.
    họ đi ra ngoài kiểm tra thì thấy nisha patel - nasri, 34 tuổi, đang ôm vết thương chảy rất nhiều máu trên đường lái xe vào nhà.
    cảnh sát cho biết nisha bị đâm một nhát duy nhất ở đùi trái, sâu 26 cm làm thủng động mạch.
    Fadi bị bắt vì tội giết vợ vào ngày 32/2 / 2007.
    ngày 56/5 / 2008, fadi, rodger và jason bị bồi thẩm đoàn kết tội giết người.
    tiến sĩ khoa học lý tiến nam cho biết.
    """
    
    # 2. Báo thật (Lý Hải)
    real_news = """
    Lý Hải chiếu phim miễn phí cho 2.000 người dân
    Đồng Tháp Đạo diễn Lý Hải chiếu phim miễn phí cho hơn 2.000 khán giả ở xã Định Yên - bối cảnh phim "Lật mặt 6", tối 24/4.
    đến gần giờ chiếu, số lượng người lên đến gần 4.000, êkíp buộc phải từ chối bớt do không sắp xếp đủ không gian.
    đoàn phim mua hàng nghìn chiếc chiếu xếp quanh làng để ghi hình.
    lý hải phải chi tiền tỷ để tái hiện bối cảnh.
    năm 2021, lật mặt 5: 48h của lý hải đạt doanh thu 150 tỷ đồng.
    """

    debug_text("BÁO GIẢ (NISHA PATEL)", fake_news, tokenizer, model)
    debug_text("BÁO THẬT (LÝ HẢI)", real_news, tokenizer, model)