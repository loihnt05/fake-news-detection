from openai import OpenAI
import os
from dotenv import load_dotenv
import glob

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Find all batch files
batch_files = sorted(glob.glob("batch_requests_*.jsonl"))

if not batch_files:
    print("No batch files found!")
    exit(1)

print(f"Found {len(batch_files)} batch files to upload")

batch_ids = []

for batch_file in batch_files:
    print(f"\nUploading {batch_file}...")
    
    # Upload JSONL file
    file = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )
    print(f"  File ID: {file.id}")
    
    # Create batch job
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"  Batch ID: {batch.id}")
    batch_ids.append(batch.id)

print("\n" + "="*50)
print("All batches created successfully!")
print("="*50)
for i, batch_id in enumerate(batch_ids, 1):
    print(f"Batch {i}: {batch_id}")
