import psycopg2
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import sys
from pathlib import Path

# Add scripts directory to path to import local processor module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from processor import NewsProcessor
from difflib import SequenceMatcher # <--- Thêm thư viện so sánh chuỗi
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

class NewsVerifier:
    def __init__(self):
        print("⏳ Loading Models...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = NewsProcessor() 
        
        model_path = './my_model'
        if not os.path.exists(model_path):
            raise Exception("❌ Không tìm thấy thư mục ./my_model")
            
        print(f"🚀 Đang load 'Chuyên gia soi lỗi' từ: {model_path}")
        self.verifier_model = CrossEncoder(model_path, device=self.device)
        print("✅ Hệ thống đã sẵn sàng!")

    def verify(self, title, content):
        print(f"\n📝 INPUT: {title}")
        input_facts, input_vector = self.processor.process_article(title, content)
        
        if not input_vector:
            return {"status": "ERROR", "reason": "Bài viết quá ngắn."}

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # --- BƯỚC 1: VECTOR SEARCH (LẤY TOP 10) ---
        # Thay vì LIMIT 1, ta lấy 10 để tránh việc bài đúng bị đẩy xuống hạng 2, 3
        query = """
            SELECT title, extracted_facts, url, label, (embedding <=> %s::vector) as distance
            FROM articles
            WHERE embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT 10;
        """
        cur.execute(query, (input_vector,))
        candidates = cur.fetchall()
        cur.close()
        conn.close()

        if not candidates:
             return {"status": "UNDEFINED", "reason": "Không tìm thấy dữ liệu."}

        # --- BƯỚC 2: HYBRID RE-RANKING (SẮP XẾP LẠI) ---
        # Mục tiêu: Tìm bài có tiêu đề giống nhất trong đám candidates
        best_candidate = None
        best_match_score = -1
        
        print("\n🔎 Hybrid Search (Tìm bài khớp nhất trong Top 10):")
        
        for cand in candidates:
            cand_title, cand_facts, cand_url, cand_label, dist = cand
            
            # 1. Điểm Vector (Càng nhỏ càng tốt -> Đảo ngược lại để tính)
            # Vector Distance thường từ 0 đến 1. 
            vector_score = 1 - dist 
            
            # 2. Điểm Tiêu đề (String Similarity)
            # So sánh độ giống nhau của chuỗi ký tự (0.0 -> 1.0)
            title_score = SequenceMatcher(None, title, cand_title).ratio()
            
            # 3. Điểm tổng hợp (Weighted Score)
            # Ưu tiên Tiêu đề (70%) + Vector (30%)
            # Vì nếu tiêu đề giống hệt nhau thì chắc chắn là bài đó!
            final_score = (title_score * 0.7) + (vector_score * 0.3)
            
            print(f"   - '{cand_title[:30]}...' | Title Sim: {title_score:.2f} | Dist: {dist:.4f} => Score: {final_score:.4f}")
            
            if final_score > best_match_score:
                best_match_score = final_score
                best_candidate = cand

        # --- BƯỚC 3: KIỂM TRA NGƯỠNG ---
        target_title, target_facts, target_url, target_label, dist = best_candidate
        
        # Logic mới: Nếu Title giống > 80% thì CHẤP NHẬN LUÔN (bất chấp distance vector)
        title_similarity = SequenceMatcher(None, title, target_title).ratio()
        
        is_valid_topic = False
        if title_similarity > 0.8:
            is_valid_topic = True
            print("✅ Tiêu đề khớp > 80% -> Bỏ qua check Vector Distance.")
        elif dist < 0.35: # Nếu tiêu đề không giống lắm, thì Vector phải rất gần
            is_valid_topic = True
        
        if not is_valid_topic:
             return {
                 "status": "UNDEFINED", 
                 "explanation": f"Không tìm thấy bài gốc tương ứng (Tiêu đề lệch, Vector xa).",
                 "source": None,
                 "details": []
             }

        # Xử lý label database
        if target_label is None: target_label = 1
        try: target_label = int(target_label)
        except: target_label = 1

        print(f"⚡ CHỐT BÀI GỐC: '{target_title}' (Label: {target_label})")

        # --- BƯỚC 4: SOI LỖI (MODEL AI) ---
        details = []
        fake_signals = 0
        true_signals = 0
        
        for in_fact in input_facts:
            src_embeddings = self.processor.embed_model.encode(target_facts, convert_to_tensor=True)
            in_embedding = self.processor.embed_model.encode(in_fact, convert_to_tensor=True)
            hits = util.semantic_search(in_embedding, src_embeddings, top_k=1)
            best_evidence = target_facts[hits[0][0]['corpus_id']]
            
            # Model soi lỗi
            ai_score = self.verifier_model.predict([(best_evidence, in_fact)])
            if hasattr(ai_score, 'item'): ai_score = ai_score.item()
            else: ai_score = float(ai_score)
            
            if ai_score > 0.6: # Nới lỏng một chút (0.6)
                label_str = "TRUE"
                true_signals += 1
            elif ai_score < 0.25: 
                label_str = "FAKE"
                fake_signals += 1
            else:
                label_str = "NEUTRAL"
            
            details.append({
                "claim": in_fact,
                "evidence": best_evidence,
                "result": label_str,
                "confidence": f"{ai_score:.2f}"
            })

        # KẾT LUẬN
        status = "UNDEFINED"
        explanation = ""

        if fake_signals > 0:
            status = "FAKE"
            explanation = f"AI phát hiện {fake_signals} chi tiết sai lệch với bài gốc."
        elif true_signals > 0:
            if target_label == 1:
                status = "TRUE"
                explanation = "Thông tin chính xác, khớp với bài báo gốc."
            else:
                status = "FAKE"
                explanation = "Bài viết khớp nội dung với một tin giả trong hệ thống."
        else:
            status = "FAKE"
            explanation = "Nội dung không đủ cơ sở xác thực."

        return {
            "status": status,
            "explanation": explanation,
            "source": {
                "title": target_title,
                "label": target_label, 
                "url": target_url
            },
            "details": details
        }
        
if __name__ == "__main__":
    checker = NewsVerifier()
    
    # Test case "huyền thoại" của chúng ta
    t = "V-League 2023-2024 khởi tranh"
    c = """
    Trận đấu giữa Hải Phòng và HAGL hôm nay 20/10 đánh dấu sự bắt đầu cho mùa giải đặc biệt của V-League, khi lần đầu thi đấu vắt ngang giữa hai năm.|Thay vì đá năm đơn như 22 mùa đã qua, năm nay V-League được điều chỉnh về thời gian. Theo đó, giải khởi tranh từ tháng 10/2023 và kết thúc vào tháng 7/2024, theo khung thời gian đồng hộ với hệ thống thi đấu của Liên đoàn Bóng đá châu Á dành cho các CLB.

Sự thay đổi này giúp tối ưu hóa lịch thi đấu giải, đồng bộ thị trường chuyển nhượng cầu thủ đối với các giải đấu hàng đầu châu Âu nhằm giúp các CLB tuyển dụng được những cầu thủ và HLV có chất lượng cao. Điều này hứa hẹn thu hút sự quan tâm của khán giả truyền hình, giới truyền thông, đảm bảo sức khỏe cầu thủ do điều kiện thời tiết khắc nghiệt ở một số nước châu Á, đồng thời phân bổ đều hơn các trận đấu của CLB hàng năm để duy trì sự cân bằng với các trận đấu của đội tuyển quốc gia.

Quy định về đăng ký cầu thủ cũng có sự thay đổi, với mục tiêu buộc các CLB phải chăm lo tới đào tạo trẻ hơn. Mỗi đội tại V-League phải có tối thiểu ba cầu thủ có quốc tịch Việt Nam ở lứa tuổi từ 16 đến 22. Quy định này giúp các cầu thủ trẻ có thêm cơ hội được tích luỹ kinh nghiệm ở môi trường bóng đá cao nhất trong nước, rèn luyện khả năng chuyên môn, với kỳ vọng sẽ có bước phát triển tốt, nhằm tạo nguồn cầu thủ cho các đội trẻ quốc gia, từ U19 tới U23 và hướng tới sớm có suất ở đội tuyển quốc gia.

V-League mùa này cũng thay đổi cả thể thức thi đấu. Các đội sẽ đá vòng tròn hai lượt sân nhà - sân khách, tính điểm để xếp hạng, giống như trước khi có đại dịch Covid-19. Trước đó, hai mùa vừa qua các đội đá một lượt tính điểm, chia hai nhóm để đá giai đoạn hai, một nhóm đua vô địch và một nhóm đua trụ hạng. Mùa này cuộc chiến trụ hạng hứa hẹn sẽ khốc liệt hơn khi suất xuống hạng tăng từ một lên một suất rưỡi. Đội đứng cuối bảng sẽ xuống hạng thẳng, trong khi đội áp chót phải đá play-off với đội á quân ở giải hạng Nhất Quốc gia để quyết định suất thứ 14 dự V-League mùa sau.

Giải đấu cũng được kỳ vọng sẽ công bằng hơn với VAR. Sau giai đoạn thử nghiệm ở cuối mùa 2023, VAR sẽ được dùng phổ biến hơn thay vì mỗi vòng một trận như trước. VPF cho biết sẽ có hai xe VAR trong giai đoạn đầu mùa giải, chạy khắp ba miền Bắc, Trung và Nam để thực hiện công việc và mỗi vòng sẽ có bốn trận đấu được áp dụng công nghệ này. Việc VAR được sử dụng nhiều sẽ giúp các trọng tài tránh được các sai sót - điều được coi là"vấn nạn"của bóng đá Việt Nam trong nhiều năm qua.

Tổng giải thưởng tại V-League mùa 2023-2024 tăng lên 9,5 tỷ đồng, cao hơn 500 triệu so với mùa 2023. Trong đó, đội vô địch nhận năm tỷ đồng, đội á quân nhận ba tỷ đồng và đội đứng thứ ba nhận 1,5 tỷ đồng.

CAHN là đương kim vô địch và tiếp tục được đánh giá là ứng viên số một cho vị trí cao nhất ở mùa giải năm nay. Đội bóng này đang sở hữu"Đội hình trong mơ". Bên cạnh những ngôi sao đã có như Filip Nguyễn, Nguyễn Quang Hải, Phan Văn Đức, Vũ Văn Thanh... họ còn vừa có thêm ba bản hợp đồng chất lượng là trung vệ Bùi Hoàng Việt Anh, tiền vệ Lê Phạm Thành Long và cầu thủ trẻ giàu triển vọng ở đội U23 Việt Nam Hồ Văn Cường.

Đối thủ lớn nhất trong cuộc đua vô địch với CAHN được cho là Hà Nội FC. Đây là CLB giàu thành tích nhất Việt Nam với sáu chức vô địch V-League. Mùa trước, đội bóng của ông bầu Đỗ Quang Hiển để tuột chức vô địch vào tay CAHN nhưng cũng chỉ kém về hiệu số bàn thắng bại, khi cùng có 38 điểm. Chủ tịch CLB Hà Nội Đỗ Vinh Quang cho biết mùa năm nay đội bóng thủ đô quyết tâm sẽ đòi lại chức vô địch.

Ngoài Hà Nội FC, Viettel cũng là đối thủ đáng gờm trong cuộc đua vô địch. Đội bóng này đang sở hữu cầu thủ Việt Nam hay nhất hiện tại Nguyễn Hoàng Đức, cùng một loạt ngôi sao như Bùi Tiến Dũng hay Nguyễn Đức Chiến. Viettel cũng mới tăng cường thêm sức mạnh hàng công khi đón chào sự trở lại của Bruno, chân sút đã giúp Thanh Hoá bay cao ở mùa giải trước với vị trí thứ tư V-League và chức vô địch Cup Quốc gia.

Ngoài ra,"đại gia mới nổi"Nam Định cũng đang rất quyết tâm để có thể lần đầu tiên đưa chức vô địch V-League về sân Thiên Trường. Trong hai tháng qua, đội bóng này đã chiêu mộ"Vua phá lưới"Rafaelson, đưa Nguyễn Văn Toàn từ Hàn Quốc về, và ký với một loạt tuyển thủ như Lý Công Hoàng Anh, Trần Văn Kiên hay Nguyễn Văn Vĩ.
    """
    
    res = checker.verify(t, c)
    print("\n" + "="*30)
    print(f"🛑 KẾT QUẢ: {res['status']}")
    print(f"💡 Lý do: {res['explanation']}")
    print("-" * 30)
    for d in res['details']:
        print(f"[{d['result']}] Claim: {d['claim']}")
        print(f"       Evid : {d['evidence']}")
        print(f"       Score: {d['confidence']}")
        print("-" * 10)