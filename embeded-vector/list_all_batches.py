from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List all batches
batches = client.batches.list(limit=20)

print("Current batches:")
print("="*80)

for batch in batches.data:
    print(f"\nBatch ID: {batch.id}")
    print(f"  Status: {batch.status}")
    print(f"  Created: {batch.created_at}")
    print(f"  Requests: {batch.request_counts.total} total, "
          f"{batch.request_counts.completed} completed, "
          f"{batch.request_counts.failed} failed")
    if batch.errors:
        print(f"  Errors: {batch.errors.data[0].message if batch.errors.data else 'None'}")
