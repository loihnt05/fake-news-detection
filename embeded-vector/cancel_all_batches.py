from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List all batches
print("Fetching all batches...")
batches = client.batches.list(limit=100)

# Filter batches that can be cancelled (validating, in_progress)
cancellable_statuses = ["validating", "in_progress", "finalizing"]
cancellable = [b for b in batches.data if b.status in cancellable_statuses]

# Failed batches (already failed, just for info)
failed = [b for b in batches.data if b.status == "failed"]

print(f"\nFound {len(cancellable)} batches to cancel")
print(f"Found {len(failed)} already failed batches (will be ignored by OpenAI)")

if cancellable:
    print("\nCancelling batches...")
    for batch in cancellable:
        try:
            cancelled = client.batches.cancel(batch.id)
            print(f"  ✓ Cancelled: {batch.id} (was {batch.status})")
        except Exception as e:
            print(f"  ✗ Failed to cancel {batch.id}: {e}")
else:
    print("\nNo batches to cancel.")

print(f"\n✅ Cleanup complete!")
print(f"\nNote: {len(failed)} failed batches will automatically clear from the system.")
print("Wait 1-2 minutes, then try uploading again.")
