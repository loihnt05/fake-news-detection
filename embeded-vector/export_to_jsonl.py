import psycopg2
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432")
)

cur = conn.cursor()
cur.execute("SELECT id, content FROM articles WHERE description IS NULL OR description = ''")

rows = cur.fetchall()

print(f"Found {len(rows)} articles")

# Split into smaller batches to stay under 2M token limit
# Average ~950 tokens per request: 2,000 requests = ~1.9M tokens (safe)
BATCH_SIZE = 2000
num_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Will create {num_batches} batch files")

for batch_num in range(num_batches):
    start_idx = batch_num * BATCH_SIZE
    end_idx = min((batch_num + 1) * BATCH_SIZE, len(rows))
    batch_rows = rows[start_idx:end_idx]
    
    output_file = f"batch_requests_{batch_num + 1}.jsonl"
    
    with open(output_file, "w") as f:
        for article_id, content in tqdm(batch_rows, desc=f"Batch {batch_num + 1}/{num_batches}"):
            content_text = (content or "")[:3500]

            req = {
                "custom_id": str(article_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",  
                    "messages": [
                        {"role": "system", "content": "You summarize Vietnamese articles briefly and clearly."},
                        {"role": "user", "content": f"Tóm tắt bài báo sau thành 2-3 câu:\n\n{content_text}"}
                    ],
                    "max_tokens": 120,
                    "temperature": 0.2
                }
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    
    print(f"Created {output_file} with {len(batch_rows)} requests")

print("JSONL export completed!")
conn.close()
