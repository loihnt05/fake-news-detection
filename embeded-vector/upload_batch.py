from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Upload JSONL file
file = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch"
)

print("Uploaded batch file:", file.id)

# 2. Create batch job
batch = client.batches.create(
    input_file_id=file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

print("Batch created:")
print("Batch ID:", batch.id)
