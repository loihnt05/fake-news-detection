import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

batch_id = "batch_6938d4e5e9e08190b4eb15d6329998ae"

batch = client.batches.retrieve(batch_id)

print("Status:", batch.status)
print("Output file ID:", batch.output_file_id)
print(batch)
