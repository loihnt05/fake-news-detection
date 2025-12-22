import psycopg2
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from dotenv import load_dotenv

load_dotenv()

# --- CẤU HÌNH ---
# Đường dẫn đến model V5 bạn vừa giải nén
MODEL_PATH = "my_model_v5"

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def debug_v5():
    print(f"🚀 ĐANG LOAD MODEL V5 TỪ: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. LOAD MODEL (NATIVE)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        print("✅ Model V5 Load OK (Native Mode).")
    except Exception as e:
        print(f"❌ Lỗi Load Model: {e}")
        print("👉 Hãy kiểm tra lại đường dẫn thư mục model!")
        return

    # 2. LOAD RETRIEVER
    print("⏳ Đang load Retriever...")
    retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)

    # ---------------------------------------------------------
    # TEST CASE KHÓ NHẤT (CÂU DÀI)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("🧪 KIỂM TRA ĐỘ THÔNG MINH (Hard Case)")
    print("="*60)
    
    claim = "V-League 2024-2025 dự kiến khai mạc vào tháng 12. với 24 trong tổng số 26 vòng đấu dự kiến diễn ra vào ngày cuối tuần."
    # Giả lập Evidence tìm được từ DB (Câu đúng nhưng dài)
    evidence_simulated = "V-League 2024-2025 sẽ khai mạc từ ngày 23/8. với 24 trong tổng số 26 vòng đấu dự kiến diễn ra vào ngày cuối tuần."

    inputs = tokenizer(claim, evidence_simulated, return_tensors='pt', truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)[0].cpu().numpy()
    
    print(f"🔹 Claim: ...tháng 12... [đuôi dài]")
    print(f"🔸 Evid : ...ngày 23/8... [đuôi dài]")
    
    labels = ["FAKE 🛑", "REAL ✅", "NEI ⚪"]
    idx = probs.argmax()
    
    print(f"\n📊 Scores: FAKE={probs[0]:.4f} | REAL={probs[1]:.4f} | NEI={probs[2]:.4f}")
    print(f"👉 KẾT QUẢ: {labels[idx]}")

    if idx == 0 and probs[0] > 0.8:
        print("🎉 TUYỆT VỜI! Model đã bắt được lỗi trong câu dài.")
    else:
        print("⚠️ Model vẫn còn lưỡng lự.")

    # ---------------------------------------------------------
    # TEST DB INTEGRATION
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("📡 KIỂM TRA DỮ LIỆU TỪ DATABASE")
    print("="*60)
    
    vec = retriever.encode(claim)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Lấy câu tốt nhất
    cur.execute("""
        SELECT content, (embedding <=> %s::vector) as distance
        FROM sentence_store
        ORDER BY distance ASC
        LIMIT 1; 
    """, (vec.tolist(),))
    
    row = cur.fetchone()
    conn.close()
    
    if row:
        db_content, dist = row
        print(f"🔎 Tìm thấy trong DB (Dist: {dist:.4f}):")
        print(f"📄 {db_content}")
        
        # Check thật
        inputs = tokenizer(claim, db_content, return_tensors='pt', truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)[0].cpu().numpy()
        
        final_lbl = labels[probs.argmax()]
        print(f"\n🤖 Model phán quyết với dữ liệu DB: {final_lbl} (FAKE Score: {probs[0]:.2f})")
    else:
        print("❌ Không tìm thấy dữ liệu nào trong DB.")

if __name__ == "__main__":
    debug_v5()