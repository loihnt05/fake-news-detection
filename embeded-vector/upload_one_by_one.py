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

print(f"Found {len(batch_files)} batch files")
print("Will upload ONE batch at a time and wait for completion before next upload\n")

completed_batches = []
failed_batches = []

for i, batch_file in enumerate(batch_files, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(batch_files)}] Processing {batch_file}")
    print('='*80)
    
    try:
        # Upload file
        print("  📤 Uploading file...")
        file = client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        print(f"  ✓ File ID: {file.id}")
        
        # Create batch
        print("  🚀 Creating batch...")
        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"  ✓ Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")
        
        # Wait for batch to complete
        print(f"  ⏳ Waiting for batch to complete...")
        
        while True:
            batch = client.batches.retrieve(batch.id)
            status = batch.status
            
            if status == "completed":
                print(f"  ✅ Batch completed!")
                print(f"     Total: {batch.request_counts.total}")
                print(f"     Completed: {batch.request_counts.completed}")
                print(f"     Failed: {batch.request_counts.failed}")
                completed_batches.append((batch_file, batch.id))
                break
            elif status in ["failed", "expired", "cancelled"]:
                error_msg = batch.errors.data[0].message if batch.errors and batch.errors.data else "Unknown error"
                print(f"  ❌ Batch {status}: {error_msg}")
                failed_batches.append((batch_file, batch.id, error_msg))
                break
            elif status in ["validating", "in_progress", "finalizing"]:
                print(f"     Status: {status} - checking again in 30s...")
                time.sleep(30)
            else:
                print(f"     Unknown status: {status} - checking again in 30s...")
                time.sleep(30)
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        failed_batches.append((batch_file, "N/A", str(e)))
        continue

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n✅ Completed: {len(completed_batches)}")
for filename, batch_id in completed_batches:
    print(f"   {filename} → {batch_id}")

print(f"\n❌ Failed: {len(failed_batches)}")
for filename, batch_id, error in failed_batches:
    print(f"   {filename} → {batch_id}")
    print(f"      Error: {error}")
