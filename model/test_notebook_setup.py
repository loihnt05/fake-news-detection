"""
Script để kiểm tra notebook setup trước khi chạy
"""
import os
from dotenv import load_dotenv

print("=" * 60)
print("KIỂM TRA NOTEBOOK SETUP")
print("=" * 60)

# 1. Kiểm tra .env
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
    print("   ❌ Không tìm thấy .env hoặc thiếu config")
    exit(1)

print(f"   ✅ Database config OK")
print(f"      - DB: {DB_CONFIG['dbname']}")
print(f"      - User: {DB_CONFIG['user']}")
print(f"      - Host: {DB_CONFIG['host']}")

# 2. Kiểm tra dependencies
print("\n2. Kiểm tra Python packages...")
required_packages = [
    'psycopg2',
    'pandas', 
    'numpy',
    'torch',
    'sklearn',
    'matplotlib',
    'seaborn'
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg} - THIẾU")
        missing.append(pkg)

if missing:
    print(f"\n   💡 Cài đặt packages thiếu:")
    print(f"      uv pip install {' '.join(missing)}")
    exit(1)

# 3. Kiểm tra database connection và data
print("\n3. Kiểm tra database và dữ liệu...")
try:
    import psycopg2
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Kiểm tra bảng
    cur.execute("SELECT COUNT(*) FROM articles")
    total = cur.fetchone()[0]
    print(f"   ✅ Tổng articles: {total}")
    
    # Kiểm tra data sẵn sàng train
    cur.execute("SELECT COUNT(*) FROM articles WHERE label IS NOT NULL AND embedding IS NOT NULL")
    ready = cur.fetchone()[0]
    print(f"   ✅ Articles sẵn sàng train: {ready}")
    
    if ready == 0:
        print("   ❌ KHÔNG có dữ liệu để train!")
        print("   💡 Cần có dữ liệu với cả label và embedding")
        conn.close()
        exit(1)
    
    # Kiểm tra phân bố labels
    cur.execute("""
        SELECT label, COUNT(*) 
        FROM articles 
        WHERE label IS NOT NULL AND embedding IS NOT NULL 
        GROUP BY label 
        ORDER BY label
    """)
    labels = cur.fetchall()
    print(f"   📊 Phân bố labels:")
    for label, count in labels:
        label_name = "Fake" if label == 0 else "Real" if label == 1 else f"Unknown({label})"
        pct = (count/ready)*100
        print(f"      - Label {label} ({label_name}): {count} ({pct:.1f}%)")
    
    # Kiểm tra sample embedding
    cur.execute("SELECT embedding FROM articles WHERE embedding IS NOT NULL LIMIT 1")
    sample = cur.fetchone()
    if sample:
        import ast
        emb = ast.literal_eval(sample[0]) if isinstance(sample[0], str) else sample[0]
        print(f"   ✅ Embedding dimension: {len(emb)}")
    
    conn.close()
    
except Exception as e:
    print(f"   ❌ Lỗi: {e}")
    exit(1)

# 4. Kiểm tra GPU/CPU
print("\n4. Kiểm tra thiết bị tính toán...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"      CUDA version: {torch.version.cuda}")
    else:
        print(f"   ℹ️  Sử dụng CPU (training sẽ chậm hơn)")
except Exception as e:
    print(f"   ⚠️  Không kiểm tra được: {e}")

# 5. Summary
print("\n" + "=" * 60)
print("KẾT LUẬN")
print("=" * 60)
print(f"✅ Notebook sẵn sàng để chạy!")
print(f"\n📝 Để chạy notebook:")
print(f"   jupyter notebook train_classifier.ipynb")
print(f"\n📊 Dữ liệu training:")
print(f"   - Tổng mẫu: {ready}")
print(f"   - Vector dimension: {len(emb) if sample else 'Unknown'}")
print("=" * 60)
