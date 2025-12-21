import psycopg2
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from dotenv import load_dotenv

load_dotenv()

# Cấu hình
MODEL_PATH = "my_model_v4" # Đảm bảo đúng tên folder bạn đã giải nén
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def debug_pipeline():
    print(f"🚀 ĐANG DEBUG HỆ THỐNG TẠI: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. LOAD MODEL (NATIVE)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        print("✅ Model Load OK.")
    except Exception as e:
        print(f"❌ Lỗi Load Model: {e}")
        return

    # 2. LOAD RETRIEVER
    retriever = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)

    # ---------------------------------------------------------
    # TEST 1: KIỂM TRA MODEL (KHÔNG DÙNG DB) - ĐỂ CHỨNG MINH MODEL KHÔN
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("🧪 TEST 1: MODEL CÓ 'KHÔN' KHÔNG? (Hardcoded Input)")
    print("="*60)
    
    claim_test = "V-League 2024-2025 dự kiến khai mạc vào tháng 12."
    evid_test  = "V-League 2024-2025 sẽ khai mạc từ ngày 23/8." # Câu chuẩn ngắn gọn

    inputs = tokenizer(claim_test, evid_test, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)[0].cpu().numpy()
    
    print(f"Claim: {claim_test}")
    print(f"Evid : {evid_test}")
    print(f"📊 Scores: REFUTED={probs[0]:.4f} | SUPPORTED={probs[1]:.4f} | NEI={probs[2]:.4f}")
    
    if probs[0] > 0.9:
        print("👉 KẾT QUẢ: ✅ MODEL HOẠT ĐỘNG TỐT (Bắt được FAKE).")
    else:
        print("👉 KẾT QUẢ: ❌ MODEL BỊ LỖI (Không giống trên Colab).")

    # ---------------------------------------------------------
    # TEST 2: KIỂM TRA RETRIEVER (DÙNG DB) - XEM NÓ TÌM RA CÁI GÌ?
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("🧪 TEST 2: RETRIEVER TÌM THẤY CÁI QUÁI GÌ? (DB Input)")
    print("="*60)
    
    # Mã hóa Claim
    vec = retriever.encode(claim_test)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Lấy Top 3 câu gần nhất
    cur.execute("""
        SELECT content, (embedding <=> %s::vector) as distance
        FROM sentence_store
        ORDER BY distance ASC
        LIMIT 3; 
    """, (vec.tolist(),))
    rows = cur.fetchall()
    
    print(f"🔎 Truy vấn: '{claim_test}'")
    
    found_good_evidence = False
    
    for i, (content, dist) in enumerate(rows):
        print(f"\n--- Ứng viên #{i+1} (Dist: {dist:.4f}) ---")
        print(f"📄 Nội dung: {content}")
        
        # Đưa vào Model check thử
        inputs = tokenizer(claim_test, content, return_tensors='pt', truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)[0].cpu().numpy()
            
        print(f"🤖 Model phán: REFUTED={probs[0]:.2f} | NEI={probs[2]:.2f}")
        
        if probs[0] > 0.8:
            print("👉 ĐÂY LÀ BẰNG CHỨNG 'CHÍ MẠNG'! (Model bắt được)")
            found_good_evidence = True
        else:
            print("👉 Câu này vô dụng (Model thấy NEI/SUPPORTED).")

    conn.close()
    
    if not found_good_evidence:
        print("\n📢 KẾT LUẬN: Retriever không tìm được câu chứa '23/8' hoặc Distance quá xa!")

if __name__ == "__main__":
    debug_pipeline()