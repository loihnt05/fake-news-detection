#!/usr/bin/env python3
"""Simple test to send articles to Kafka"""
import json
import time
import sys
from kafka import KafkaProducer
from datetime import datetime

KAFKA_SERVER = "kafka:9093"  # Try internal
# KAFKA_SERVER = "localhost:9092"  # Or external
KAFKA_TOPIC = "raw_articles"

print("=== SIMPLE KAFKA TEST ===")
print(f"Connecting to: {KAFKA_SERVER}")
print(f"Topic: {KAFKA_TOPIC}")

# Try both servers
for server in ["kafka:9093", "localhost:9092"]:
    try:
        print(f"\nTrying: {server}")
        producer = KafkaProducer(
            bootstrap_servers=[server],
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"),
            request_timeout_ms=5000,
            max_block_ms=5000
        )
        print(f"✅ Connected to {server}!")
        
        # Send 1 test article
        article = {
            "url": f"https://vnexpress.net/test-{int(time.time())}",
            "source": "vnexpress",
            "title": f"Test Article - {datetime.now()}",
            "content": "Test content",
            "published_date": str(datetime.now()),
            "scraped_at": str(datetime.now()),
            "category": "Test"
        }
        producer.send(KAFKA_TOPIC, article).get(timeout=10)
        print(f"✓ Sent article successfully!")
        
        producer.flush()
        producer.close()
        print(f"✅ Done with {server}!")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Failed with {server}: {e}")
        continue

print("❌ All attempts failed")
sys.exit(1)
