import json
import time
import sqlite3
import subprocess
import threading
import os
from datetime import datetime, timedelta
from pathlib import Path
from kafka import KafkaProducer

# --- CẤU HÌNH ---
KAFKA_TOPIC = "raw_articles"
KAFKA_SERVER = "localhost:9092"
CHECK_INTERVAL = 5  # Producer quét DB mỗi 5 giây (Real-time)

# Đường dẫn
SCRAPER_DIR = Path(__file__).parent.parent.parent / "scrape-vnexpress"
SCRAPER_BINARY = SCRAPER_DIR / "scraper-db" 
SCRAPER_DB = SCRAPER_DIR / "scraped_articles.db"
TIMESTAMP_FILE = SCRAPER_DIR / ".last_scraped_at"

# Biến toàn cục để lưu mốc thời gian quét (Thread-safe đơn giản)
SHARED_STATE = {
    "last_scraped_at": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
}

# --- 1. LUỒNG THỢ CÀO (SCRAPER WORKER) ---
def task_run_scraper():
    """Luồng này chỉ chuyên chạy Scraper Go liên tục"""
    print("🕷️ [Thread-Scraper] Đã khởi động thợ cào...")
    
    while True:
        try:
            # Chạy Scraper
            # Lưu ý: Scraper Go phải được thiết kế để update DB liên tục (không đợi xong mới commit)
            print(f"\n🕷️ [Thread-Scraper] Bắt đầu vòng cào mới...")
            
            # Dùng Popen để không chặn luồng nhưng vẫn in được log
            process = subprocess.Popen(
                [str(SCRAPER_BINARY), "-parallelism", "2"],
                cwd=str(SCRAPER_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Đọc log của Scraper để biết nó đang làm gì
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # In log mờ nhạt hơn để đỡ rối mắt
                    print(f"    (Scraper): {line.strip()}")
            
            print("🕷️ [Thread-Scraper] Cào xong đợt này. Nghỉ 60s...")
            time.sleep(60) 
            
        except Exception as e:
            print(f"❌ [Thread-Scraper] Lỗi: {e}")
            time.sleep(60)

# --- 2. LUỒNG THỢ VẬN CHUYỂN (PRODUCER WORKER) ---
def get_new_articles_from_db(since_timestamp):
    """Đọc SQLite chế độ WAL (Non-blocking)"""
    try:
        # Timeout cực ngắn vì ta quét liên tục
        conn = sqlite3.connect(f"file:{SCRAPER_DB}?mode=ro", uri=True, timeout=5)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT url, title, content, published_date, scraped_at, category
            FROM articles
            WHERE scraped_at > ?
            ORDER BY scraped_at ASC
            LIMIT 50 -- Lấy từng đợt nhỏ để xử lý nhanh
        """, (since_timestamp,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            "url": r[0], "source": "vnexpress", "title": r[1], 
            "content": r[2], "published_date": r[3], "scraped_at": r[4],
            "category": r[5] or "Uncategorized"
        } for r in rows]
    except Exception as e:
        # Lỗi lock DB là bình thường khi chạy song song, bỏ qua chờ lượt sau
        if "locked" not in str(e):
            print(f"⚠️ [Thread-Producer] Lỗi đọc DB: {e}")
        return []

def task_run_producer():
    """Luồng này chuyên quét DB và bắn Kafka"""
    print("📦 [Thread-Producer] Đã khởi động dây chuyền vận chuyển...")
    
    # 1. Kết nối Kafka
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_SERVER],
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8")
        )
        print("✅ [Thread-Producer] Kafka Connected!")
    except Exception as e:
        print(f"❌ [Thread-Producer] Lỗi Kafka: {e}")
        return

    # Khởi tạo mốc thời gian từ file (nếu có)
    if TIMESTAMP_FILE.exists():
        try:
            content = TIMESTAMP_FILE.read_text().strip()
            dt = datetime.fromisoformat(content.replace('Z', '').replace('T', ' '))
            # Reset nếu ngày tương lai (Fix lỗi 2025 của bạn)
            if dt > datetime.now():
                print("⚠️ [Fix] Reset ngày tương lai về 2 ngày trước.")
            else:
                SHARED_STATE["last_scraped_at"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except: pass

    print(f"🕒 [Thread-Producer] Bắt đầu quét từ: {SHARED_STATE['last_scraped_at']}")

    while True:
        # Quét DB
        articles = get_new_articles_from_db(SHARED_STATE["last_scraped_at"])
        
        if articles:
            print(f"\n📦 [Thread-Producer] Tìm thấy {len(articles)} bài mới! Đang gửi...")
            
            for art in articles:
                try:
                    producer.send(KAFKA_TOPIC, art)
                    print(f"   ✓ Sent: {art['title'][:50]}...")
                except Exception as e:
                    print(f"   ❌ Fail: {e}")
            
            producer.flush()
            
            # Cập nhật mốc thời gian ngay lập tức
            SHARED_STATE["last_scraped_at"] = articles[-1]["scraped_at"]
            print(f"📍 [Thread-Producer] Cập nhật mốc: {SHARED_STATE['last_scraped_at']}")
        
        # Nghỉ ngắn (5s) để tạo cảm giác Real-time
        time.sleep(CHECK_INTERVAL)

# --- MAIN ---
if __name__ == "__main__":
    print("🚀 HỆ THỐNG PRODUCER ĐA LUỒNG (MULTI-THREADING)")
    print("==============================================")
    
    # Bật chế độ WAL cho DB (Chỉ cần làm 1 lần)
    try:
        conn = sqlite3.connect(str(SCRAPER_DB))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.close()
        print("✅ Database WAL Mode: ENABLED (Cho phép đọc/ghi song song)")
    except: pass

    # Tạo 2 luồng
    t1 = threading.Thread(target=task_run_scraper, daemon=True)
    t2 = threading.Thread(target=task_run_producer, daemon=True)
    
    # Chạy
    t1.start()
    t2.start()
    
    # Giữ Main thread sống
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Đang dừng hệ thống...")