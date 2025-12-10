from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List all completed batches
batches = client.batches.list(limit=100)

completed_batches = [b for b in batches.data if b.status == "completed"]

print(f"Found {len(completed_batches)} completed batches")

for i, batch in enumerate(completed_batches, 1):
    if not batch.output_file_id:
        print(f"[{i}] Batch {batch.id} has no output file, skipping...")
        continue
    
    print(f"\n[{i}/{len(completed_batches)}] Downloading batch {batch.id}")
    print(f"  Output file ID: {batch.output_file_id}")
    print(f"  Requests: {batch.request_counts.completed} completed")
    
    # Download the output file
    file_response = client.files.content(batch.output_file_id)
    
    # Save to disk
    output_filename = f"batch_output_{batch.id}.jsonl"
    with open(output_filename, "wb") as f:
        f.write(file_response.content)
    
    print(f"  ✓ Saved to {output_filename}")

print(f"\n✅ Downloaded {len(completed_batches)} result files")
