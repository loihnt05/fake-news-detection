import psycopg2
import os
from dotenv import load_dotenv
import json
import glob
from tqdm import tqdm

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432")
)

cur = conn.cursor()

# Find all output files
output_files = glob.glob("batch_output_*.jsonl")

if not output_files:
    print("No output files found! Run download_all_results.py first.")
    exit(1)

print(f"Found {len(output_files)} output files to import\n")

total_updated = 0
total_failed = 0

for output_file in output_files:
    print(f"Processing {output_file}...")
    
    with open(output_file, "r") as f:
        lines = f.readlines()
    
    updated = 0
    failed = 0
    
    for line in tqdm(lines, desc=f"  Importing"):
        try:
            result = json.loads(line)
            
            # Extract article ID and summary
            article_id = result["custom_id"]
            
            if result["response"]["status_code"] == 200:
                summary = result["response"]["body"]["choices"][0]["message"]["content"]
                
                # Update database
                cur.execute(
                    "UPDATE articles SET description = %s WHERE id = %s",
                    (summary, article_id)
                )
                updated += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"\n  Error processing line: {e}")
            failed += 1
    
    conn.commit()
    print(f"  ✓ Updated: {updated}, Failed: {failed}\n")
    
    total_updated += updated
    total_failed += failed

print("="*80)
print(f"✅ TOTAL: Updated {total_updated} articles, Failed {total_failed}")
print("="*80)

cur.close()
conn.close()
