"""
Script để test cấu hình FastAPI trước khi chạy server
"""
import os
from dotenv import load_dotenv
import psycopg2

print("=" * 60)
print("TEST CẤU HÌNH FASTAPI")
print("=" * 60)

# 1. Test load .env
print("\n1. Kiểm tra file .env...")
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

if not DB_CONFIG["dbname"]:
    print("   ❌ Không đọc được .env file")
    exit(1)

print("   ✅ Đã load .env file")
print(f"   - Database: {DB_CONFIG['dbname']}")
print(f"   - User: {DB_CONFIG['user']}")
print(f"   - Host: {DB_CONFIG['host']}")

# 2. Test database connection
print("\n2. Kiểm tra kết nối database...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Test query
    cur.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL")
    count = cur.fetchone()[0]
    print(f"   ✅ Kết nối thành công!")
    print(f"   - Số articles có embedding: {count}")
    
    # Test sample embedding
    cur.execute("SELECT embedding FROM articles WHERE embedding IS NOT NULL LIMIT 1")
    sample = cur.fetchone()
    if sample:
        import ast
        emb = ast.literal_eval(sample[0])
        print(f"   - Embedding dimension: {len(emb)}")
    
    conn.close()
except Exception as e:
    print(f"   ❌ Lỗi kết nối: {e}")
    exit(1)

# 3. Test embedding model
print("\n3. Kiểm tra embedding model...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    test_text = "Đây là tin test"
    emb = model.encode([test_text])[0]
    print(f"   ✅ Model loaded thành công!")
    print(f"   - Test embedding dimension: {len(emb)}")
except Exception as e:
    print(f"   ❌ Lỗi load model: {e}")
    exit(1)

# 4. Test classifier model exists
print("\n4. Kiểm tra classifier model file...")
import os.path
model_path = "model/fakenews_classifier.pth"
if os.path.exists(model_path):
    print(f"   ✅ Tìm thấy file: {model_path}")
    import torch
    try:
        state = torch.load(model_path, map_location='cpu')
        print(f"   ✅ Model file hợp lệ!")
    except Exception as e:
        print(f"   ❌ Lỗi load model file: {e}")
        exit(1)
else:
    print(f"   ❌ KHÔNG tìm thấy file: {model_path}")
    print(f"   💡 Bạn cần train model trước:")
    print(f"      cd model && uv run python train_classifier.py")
    exit(1)

# 5. Summary
print("\n" + "=" * 60)
print("TÓM TẮT")
print("=" * 60)
print("✅ Tất cả cấu hình OK!")
print("\n🚀 Có thể chạy FastAPI:")
print("   uvicorn main:app --reload")
print("=" * 60)
