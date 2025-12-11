"""
Script để test kết nối database và kiểm tra dữ liệu training
"""
import psycopg2
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# Cấu hình DB từ .env
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

print("=" * 60)
print("TEST KẾT NỐI DATABASE CHO TRAINING")
print("=" * 60)

# 1. Kiểm tra config
print("\n1. Cấu hình database:")
print(f"   - Database: {DB_CONFIG['dbname']}")
print(f"   - User: {DB_CONFIG['user']}")
print(f"   - Host: {DB_CONFIG['host']}")
print(f"   - Port: {DB_CONFIG['port']}")

if not DB_CONFIG["dbname"]:
    print("\n❌ Lỗi: Không đọc được file .env")
    exit(1)

# 2. Test kết nối
print("\n2. Kiểm tra kết nối...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("   ✅ Kết nối thành công!")
    
    cur = conn.cursor()
    
    # 3. Kiểm tra tổng số bài viết
    print("\n3. Thống kê dữ liệu:")
    cur.execute("SELECT COUNT(*) FROM articles")
    total = cur.fetchone()[0]
    print(f"   - Tổng số articles: {total}")
    
    # 4. Kiểm tra số bài có embedding
    cur.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL")
    with_embedding = cur.fetchone()[0]
    print(f"   - Articles có embedding: {with_embedding}")
    
    # 5. Kiểm tra số bài có label
    cur.execute("SELECT COUNT(*) FROM articles WHERE label IS NOT NULL")
    with_label = cur.fetchone()[0]
    print(f"   - Articles có label: {with_label}")
    
    # 6. Kiểm tra số bài có cả embedding VÀ label (sẵn sàng để train)
    cur.execute("SELECT COUNT(*) FROM articles WHERE label IS NOT NULL AND embedding IS NOT NULL")
    ready_for_training = cur.fetchone()[0]
    print(f"   - Articles sẵn sàng train: {ready_for_training}")
    
    # 7. Phân bố labels
    print("\n4. Phân bố labels:")
    cur.execute("""
        SELECT label, COUNT(*) 
        FROM articles 
        WHERE label IS NOT NULL AND embedding IS NOT NULL 
        GROUP BY label 
        ORDER BY label
    """)
    label_dist = cur.fetchall()
    
    if len(label_dist) == 0:
        print("   ⚠️  Không có dữ liệu có label!")
    else:
        for label, count in label_dist:
            label_name = "Real" if label == 1 else "Fake" if label == 0 else f"Unknown ({label})"
            percentage = (count / ready_for_training * 100) if ready_for_training > 0 else 0
            print(f"   - Label {label} ({label_name}): {count} ({percentage:.1f}%)")
    
    # 8. Sample một embedding để kiểm tra format
    print("\n5. Kiểm tra format embedding:")
    cur.execute("SELECT id, title, embedding FROM articles WHERE embedding IS NOT NULL LIMIT 1")
    sample = cur.fetchone()
    
    if sample:
        article_id, title, embedding_str = sample
        print(f"   - Sample ID: {article_id}")
        print(f"   - Title: {title[:50]}...")
        print(f"   - Embedding type: {type(embedding_str)}")
        print(f"   - Embedding length (chars): {len(embedding_str) if isinstance(embedding_str, str) else 'N/A'}")
        
        # Try to parse
        try:
            import ast
            embedding_list = ast.literal_eval(embedding_str) if isinstance(embedding_str, str) else embedding_str
            print(f"   - Embedding dimension: {len(embedding_list)}")
            print(f"   - First 5 values: {embedding_list[:5]}")
            print("   ✅ Embedding format hợp lệ!")
        except Exception as e:
            print(f"   ❌ Lỗi parse embedding: {e}")
    
    # 9. Tóm tắt
    print("\n" + "=" * 60)
    print("TÓM TẮT")
    print("=" * 60)
    
    if ready_for_training == 0:
        print("❌ KHÔNG THỂ TRAIN: Không có dữ liệu nào có cả label và embedding")
        print("\n💡 Hướng dẫn:")
        print("   1. Chạy embedding cho các articles (nếu chưa có)")
        print("   2. Gán label (0=Fake, 1=Real) cho các articles")
    elif ready_for_training < 100:
        print(f"⚠️  CÓ THỂ TRAIN nhưng dữ liệu ít ({ready_for_training} mẫu)")
        print("   Nên có ít nhất 1000+ mẫu để train tốt")
    else:
        print(f"✅ SẴN SÀNG TRAIN với {ready_for_training} mẫu dữ liệu")
        
        # Check balance
        if len(label_dist) >= 2:
            counts = [c for _, c in label_dist]
            ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            if ratio > 3:
                print(f"   ⚠️  Dữ liệu mất cân bằng (tỷ lệ {ratio:.1f}:1)")
                print("   💡 Cân nhắc sử dụng class weights hoặc resampling")
            else:
                print(f"   ✅ Dữ liệu cân bằng tốt (tỷ lệ {ratio:.1f}:1)")
    
    conn.close()
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Lỗi: {e}")
    exit(1)
