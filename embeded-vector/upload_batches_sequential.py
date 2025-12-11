from openai import OpenAI
import os
from dotenv import load_dotenv
import glob
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Find all batch files
batch_files = sorted(glob.glob("batch_requests_*.jsonl"))

if not batch_files:
    print("No batch files found!")
    exit(1)

print(f"Found {len(batch_files)} batch files to upload")
print("Uploading ONE batch at a time to avoid token limits...\n")

batch_ids = []

for i, batch_file in enumerate(batch_files, 1):
    print(f"[{i}/{len(batch_files)}] Uploading {batch_file}...")
    
    try:
        # Upload JSONL file
        file = client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        print(f"  ✓ File uploaded: {file.id}")
        
        # Create batch job
        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        print(f"  ✓ Batch created: {batch.id}")
        print(f"  Status: {batch.status}")
        batch_ids.append((batch_file, batch.id))
        
        # Wait a bit before next upload to be safe
        if i < len(batch_files):
            print(f"  Waiting 5 seconds before next upload...\n")
            time.sleep(5)
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print(f"  Stopping uploads. Successful batches so far: {len(batch_ids)}\n")
        break

print("\n" + "="*80)
print(f"Successfully created {len(batch_ids)} batches:")
print("="*80)
for filename, batch_id in batch_ids:
    print(f"{filename} → {batch_id}")
