from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

batch_id = input("Enter batch_id: ")

batch = client.batches.retrieve(batch_id)
print(batch)
