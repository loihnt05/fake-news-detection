from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

batch_id = input("Enter batch_id: ")

batch = client.batches.retrieve(batch_id)

output_file_id = batch.output_file_id

result = client.files.content(output_file_id)

with open("batch_results.jsonl", "wb") as f:
    f.write(result.read())

print("Download completed -> batch_results.jsonl")
