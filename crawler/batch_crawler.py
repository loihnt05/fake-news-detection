"""
Batch Crawler for Airflow DAG
Runs scraper once and sends articles to Kafka
"""
import json
import time
import os
import sys
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# --- CẤU HÌNH ---
KAFKA_TOPIC = "raw_articles"
# Use environment variable or default to kafka service name for Docker
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "kafka:9093")

# Đường dẫn project
PROJECT_DIR = Path(__file__).parent.parent
# Scraper is in parent directory ../scrape-vnexpress
SCRAPER_DIR = PROJECT_DIR.parent / "scrape-vnexpress"
SCRAPER_BINARY = SCRAPER_DIR / "scraper-db"
SCRAPER_DB = SCRAPER_DIR / "scraped_articles.db"
TIMESTAMP_FILE = SCRAPER_DIR / ".last_scraped_at"

def connect_kafka(max_retries=5):
    """Connect to Kafka with retries"""
    for attempt in range(max_retries):
        try:
            print(f"🔌 Attempting to connect to Kafka at {KAFKA_SERVER} (attempt {attempt+1}/{max_retries})...")
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_SERVER],
                value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"),
                request_timeout_ms=10000,
                api_version=(0, 10, 1)
            )
            print("✅ Connected to Kafka successfully!")
            return producer
        except NoBrokersAvailable as e:
            print(f"⚠️ Kafka not available yet: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise Exception(f"❌ Failed to connect to Kafka after {max_retries} attempts")
        except Exception as e:
            print(f"❌ Kafka connection error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise

def run_scraper():
    """Run the scraper binary"""
    print(f"🕷️ Running scraper: {SCRAPER_BINARY}")
    
    if not SCRAPER_BINARY.exists():
        print(f"❌ Scraper binary not found at: {SCRAPER_BINARY}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Project directory: {PROJECT_DIR}")
        print(f"   Scraper directory: {SCRAPER_DIR}")
        return False
    
    try:
        # Run the scraper binary with limited parallelism
        result = subprocess.run(
            [str(SCRAPER_BINARY), "-parallelism", "2"],
            cwd=str(SCRAPER_DIR),
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Scraper completed successfully")
            print(f"   Output: {result.stdout[:500]}")
            return True
        else:
            print(f"⚠️ Scraper completed with errors")
            print(f"   Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Scraper timeout after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running scraper: {e}")
        return False

def get_new_articles_from_db(since_timestamp):
    """Read articles from SQLite database"""
    if not SCRAPER_DB.exists():
        print(f"⚠️ Database not found: {SCRAPER_DB}")
        return []
    
    try:
        # Try read-only mode with immutable flag for databases we can't modify
        conn = sqlite3.connect(f"file:{SCRAPER_DB}?mode=ro&immutable=1", uri=True, timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT url, title, content, published_date, scraped_at, category
            FROM articles
            WHERE scraped_at > ?
            ORDER BY scraped_at ASC
            LIMIT 100
        """, (since_timestamp,))
        
        rows = cursor.fetchall()
        conn.close()
        
        articles = [{
            "url": r[0],
            "source": "vnexpress",
            "title": r[1],
            "content": r[2],
            "published_date": r[3],
            "scraped_at": r[4],
            "category": r[5] or "Uncategorized"
        } for r in rows]
        
        print(f"📊 Found {len(articles)} new articles in database")
        return articles
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        # Try without immutable flag
        try:
            conn = sqlite3.connect(str(SCRAPER_DB), timeout=10)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT url, title, content, published_date, scraped_at, category
                FROM articles
                WHERE scraped_at > ?
                ORDER BY scraped_at ASC
                LIMIT 100
            """, (since_timestamp,))
            
            rows = cursor.fetchall()
            conn.close()
            
            articles = [{
                "url": r[0],
                "source": "vnexpress",
                "title": r[1],
                "content": r[2],
                "published_date": r[3],
                "scraped_at": r[4],
                "category": r[5] or "Uncategorized"
            } for r in rows]
            
            print(f"📊 Found {len(articles)} new articles in database (fallback mode)")
            return articles
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return []

def send_test_articles(producer, count=5):
    """Send test articles to Kafka for debugging"""
    print(f"📤 Sending {count} test articles to Kafka topic: {KAFKA_TOPIC}")
    
    for i in range(count):
        article = {
            "url": f"https://vnexpress.net/test-article-{i}-{int(time.time())}",
            "source": "vnexpress",
            "title": f"Test Article {i} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "content": f"This is test content for article {i}. Generated at {datetime.now()}",
            "published_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": "Test"
        }
        
        try:
            producer.send(KAFKA_TOPIC, article)
            print(f"   ✓ Sent test article {i+1}/{count}")
        except Exception as e:
            print(f"   ❌ Failed to send article {i+1}: {e}")
    
    producer.flush()
    print("✅ All test articles sent!")

def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 BATCH CRAWLER FOR AIRFLOW")
    print("=" * 60)
    print(f"📍 Kafka Server: {KAFKA_SERVER}")
    print(f"📍 Kafka Topic: {KAFKA_TOPIC}")
    print(f"📍 Project Dir: {PROJECT_DIR}")
    print(f"📍 Scraper Dir: {SCRAPER_DIR}")
    print(f"📍 Scraper Binary: {SCRAPER_BINARY}")
    print(f"📍 Database: {SCRAPER_DB}")
    print("=" * 60)
    
    # Step 1: Connect to Kafka
    try:
        producer = connect_kafka()
    except Exception as e:
        print(f"❌ Cannot connect to Kafka: {e}")
        return 1
    
    # Step 2: Send test articles to verify Kafka is working
    print("\n📊 Sending test articles to verify Kafka connection...")
    send_test_articles(producer, count=3)
    
    # Step 3: Get last scraped timestamp
    last_timestamp = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    local_timestamp_file = Path("/tmp/.last_scraped_at")
    
    # Try multiple timestamp file locations
    for ts_file in [local_timestamp_file, TIMESTAMP_FILE]:
        if ts_file.exists():
            try:
                content = ts_file.read_text().strip()
                dt = datetime.fromisoformat(content.replace('Z', '').replace('T', ' '))
                if dt <= datetime.now():
                    last_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"🕒 Loaded timestamp from: {ts_file}")
                    break
            except:
                pass
    
    print(f"🕒 Last scraped at: {last_timestamp}")
    
    # Step 4: Run scraper
    print("\n🕷️ Starting scraper...")
    scraper_success = run_scraper()
    
    if not scraper_success:
        print("⚠️ Scraper had issues, but continuing to check database...")
    
    # Step 5: Read new articles from database and send to Kafka
    print("\n📦 Reading articles from database...")
    articles = get_new_articles_from_db(last_timestamp)
    
    if articles:
        print(f"📤 Sending {len(articles)} articles to Kafka...")
        sent_count = 0
        for article in articles:
            try:
                producer.send(KAFKA_TOPIC, article)
                sent_count += 1
                if sent_count % 10 == 0:
                    print(f"   Sent {sent_count}/{len(articles)}...")
            except Exception as e:
                print(f"   ❌ Failed to send article: {e}")
        
        producer.flush()
        print(f"✅ Successfully sent {sent_count}/{len(articles)} articles to Kafka!")
        
        # Update timestamp (save to /tmp since project dir may not be writable)
        new_timestamp = articles[-1]["scraped_at"]
        local_timestamp_file = Path("/tmp/.last_scraped_at")
        try:
            local_timestamp_file.write_text(new_timestamp)
            print(f"📝 Updated timestamp to: {new_timestamp}")
        except Exception as e:
            print(f"⚠️ Could not save timestamp: {e}")
    else:
        print("ℹ️ No new articles found in database")
    
    # Close producer
    producer.close()
    print("\n✅ Batch crawler completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
