import psycopg2
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432")
)

cur = conn.cursor()

with open("batch_results.jsonl", "r") as f:
    for line in tqdm(f):
        row = json.loads(line)
        article_id = row["custom_id"]

        summary = row["response"]["body"]["choices"][0]["message"]["content"]

        cur.execute(
            "UPDATE articles SET description=%s WHERE id=%s",
            (summary, article_id)
        )

conn.commit()
conn.close()

print("Import completed!")
