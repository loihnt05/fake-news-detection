from underthesea import sent_tokenize
import re

def debug_segmentation(text):
    print("\n" + "="*50)
    print("🛠️ ORIGINAL TEXT:")
    print(text.strip())
    print("-" * 50)
    
    # 1. Dùng thư viện mặc định (Cách cũ của bạn)
    raw_sentences = sent_tokenize(text)
    
    print(f"📊 KẾT QUẢ CŨ (Underthesea thuần): {len(raw_sentences)} câu")
    for i, s in enumerate(raw_sentences):
        print(f"  [{i+1}] {s}")

    # 2. Cách cải tiến (Xử lý xuống dòng & dấu câu)
    print("-" * 50)
    print("🚀 KẾT QUẢ CẢI TIẾN (Custom Split):")
    
    better_sentences = custom_segmentation(text)
    print(f"📊 Tìm thấy: {len(better_sentences)} câu")
    for i, s in enumerate(better_sentences):
        print(f"  [{i+1}] {s}")

def custom_segmentation(text):
    """
    Hàm tách câu mạnh mẽ hơn cho tin tức tiếng Việt:
    1. Tôn trọng dấu xuống dòng (\n) là hết câu.
    2. Xử lý các dấu chấm câu lửng lơ.
    3. Dùng underthesea cho đoạn văn dài.
    """
    if not text: return []
    
    # Bước 1: Thay thế các ký tự xuống dòng thành dấu ngắt câu tạm thời
    # Vì underthesea đôi khi bỏ qua \n và nối liền 2 đoạn văn.
    # VD: "Tiêu đề\nNội dung" -> "Tiêu đề Nội dung" (Sai)
    
    # Tách sơ bộ bằng xuống dòng trước
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    final_sentences = []
    for p in paragraphs:
        # Nếu đoạn văn quá ngắn (VD: Tiêu đề), coi là 1 câu
        if len(p) < 30:
            final_sentences.append(p)
            continue
            
        # Nếu đoạn văn dài, dùng underthesea tách tiếp
        sents = sent_tokenize(p)
        for s in sents:
            # Lọc rác: Đôi khi tách ra chỉ còn dấu chấm hoặc khoảng trắng
            if len(s.strip()) > 3: 
                final_sentences.append(s.strip())
                
    return final_sentences

if __name__ == "__main__":
    # --- TEST CASE 1: Dính dòng (Lỗi phổ biến khi cào web) ---
    text1 = """V-League 2024-2025 dự kiến khai mạc tháng 8.Đây là giải đấu quan trọng.
    Tuy nhiên, nhiều đội bóng chưa sẵn sàng."""
    
    # --- TEST CASE 2: Xuống dòng nhưng thiếu dấu chấm (Header/List) ---
    text2 = """Lịch thi đấu V-League
    Vòng 1: Viettel vs CAHN
    Vòng 2: Hà Nội vs Hải Phòng
    Giải đấu hứa hẹn hấp dẫn."""
    
    # --- TEST CASE 3: Viết tắt (Dễ bị cắt nhầm) ---
    text3 = "Ông Nguyễn Văn A, TP.HCM đã quyết định đầu tư 50 tr. USD cho dự án này. Tuy nhiên, TS. Lê Thẩm Dương cho rằng cần xem xét lại."

    print(">>> TEST CASE 1: DÍNH DÒNG")
    debug_segmentation(text1)
    
    print("\n>>> TEST CASE 2: DANH SÁCH & HEADER")
    debug_segmentation(text2)
    
    print("\n>>> TEST CASE 3: VIẾT TẮT")
    debug_segmentation(text3)