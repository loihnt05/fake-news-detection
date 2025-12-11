import psycopg2
import os
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio
from typing import List, Tuple

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("OpenAI client initialized.")

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432")
)

cur = conn.cursor()

cur.execute("SELECT id, content FROM articles WHERE description IS NULL OR description = '';")
rows = cur.fetchall()
total = len(rows)

if total == 0:
    print("No articles need summarization.")
    conn.close()
    exit(0)

print(f"Found {total} articles to summarize.")

CONCURRENT_REQUESTS = 6       # Safe for gpt-4o-mini
MAX_RETRIES = 5
BATCH_SIZE = 200
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

async def summarize_article(article_id, content):
    for attempt in range(MAX_RETRIES):
        try:
            content_text = (content or "")[:3000]

            if not content_text.strip():
                return (article_id, None)

            async with semaphore:  # LIMIT concurrency
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You summarize Vietnamese articles briefly and clearly."},
                        {"role": "user", "content": f"Tóm tắt bài báo sau thành 2-3 câu (dùng tiếng việt):\n\n{content_text}"}
                    ],
                    max_tokens=120,
                    temperature=0.2
                )

            summary = response.choices[0].message.content.strip()
            return (article_id, summary)

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                await asyncio.sleep(1 + attempt * 2)  # mild backoff
                continue

            if attempt == MAX_RETRIES - 1:
                print(f"\nFailed {article_id}: {e}")
            return (article_id, None)

async def main():
    total_updated = 0

    with tqdm(total=total, desc="Summarizing") as pbar:
        tasks = []
        for article_id, content in rows:
            tasks.append(summarize_article(article_id, content))

            if len(tasks) >= BATCH_SIZE:
                results = await asyncio.gather(*tasks)
                for article_id, summary in results:
                    if summary:
                        cur.execute(
                            "UPDATE articles SET description=%s WHERE id=%s",
                            (summary, article_id)
                        )
                        total_updated += 1

                conn.commit()
                pbar.update(len(tasks))
                tasks = []

        # Last batch
        if tasks:
            results = await asyncio.gather(*tasks)
            for article_id, summary in results:
                if summary:
                    cur.execute(
                        "UPDATE articles SET description=%s WHERE id=%s",
                        (summary, article_id)
                    )
                    total_updated += 1
            conn.commit()
            pbar.update(len(tasks))

    print(f"Done! Updated {total_updated}/{total}")

asyncio.run(main())
conn.close()
