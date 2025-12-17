import torch
from sentence_transformers import SentenceTransformer, util
from underthesea import sent_tokenize
import numpy as np

class NewsProcessor:
    def __init__(self):
        print("⏳ Đang load model Embedding... (Lần đầu sẽ hơi lâu)")
        # Sử dụng model tốt nhất cho tiếng Việt hiện nay để embedding
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Running on: {self.device}")
        
        self.embed_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=self.device)
        
    def process_article(self, title, content, top_k=3):
        """
        Input: Tiêu đề và Nội dung bài báo
        Output: 
            - facts: List các câu quan trọng nhất
            - doc_vector: Vector đại diện cho toàn bộ ý chính
        """
        if not content or len(content.strip()) < 10:
            return None, None

        # 1. Tách câu (Sentence Splitting) chuẩn tiếng Việt
        # Kết hợp title vào để tăng ngữ cảnh, vì title luôn chứa ý chính
        sentences = [title] + sent_tokenize(content)
        
        # Lọc các câu quá ngắn (nhiễu)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        
        if not sentences:
            return None, None

        # 2. Embedding tất cả các câu
        # embeddings là một matrix (num_sentences x 768)
        embeddings = self.embed_model.encode(sentences, convert_to_tensor=True)
        
        # 3. Tính Document Vector trung bình (Mean Pooling)
        # Đây là vector đại diện chung cho cả bài
        doc_vector = torch.mean(embeddings, dim=0)
        
        # 4. Trích xuất thông tin (Extractive Summarization)
        # Tìm các câu có độ tương đồng cao nhất với doc_vector (những câu "trọng tâm" nhất)
        cos_scores = util.cos_sim(doc_vector, embeddings)[0]
        
        # Lấy top_k câu có điểm cao nhất
        # Nếu bài ngắn hơn top_k thì lấy hết
        k = min(top_k, len(sentences))
        top_results = torch.topk(cos_scores, k=k)
        
        extracted_facts = []
        for idx in top_results.indices:
            extracted_facts.append(sentences[idx])
            
        # Chuyển doc_vector về dạng List để lưu vào DB (Postgres pgvector nhận list float)
        doc_vector_list = doc_vector.cpu().tolist()
        
        return extracted_facts, doc_vector_list

# --- PHẦN TEST THỬ (Chạy độc lập để kiểm tra) ---
if __name__ == "__main__":
    processor = NewsProcessor()
    
    test_title = "Bộ Y tế công bố thêm 10.000 ca nhiễm COVID-19"
    test_content = """
    Tối ngày 15/12, Bộ Y tế thông báo ghi nhận thêm 10.000 ca nhiễm mới tại 60 tỉnh thành.
    Trong đó, Hà Nội có số ca nhiễm cao nhất với 1.500 trường hợp.
    Các bệnh nhân đều đã được cách ly hoặc điều trị tại nhà.
    Bộ Y tế khuyến cáo người dân tiếp tục thực hiện 5K.
    Đây là số liệu được tổng hợp từ hệ thống quốc gia.
    """
    
    print("\n--- Đang xử lý bài báo mẫu ---")
    facts, vector = processor.process_article(test_title, test_content, top_k=3)
    
    print(f"✅ Vector size: {len(vector)}")
    print("✅ Các câu quan trọng (Key Facts) được trích xuất:")
    for i, fact in enumerate(facts):
        print(f"  {i+1}. {fact}")