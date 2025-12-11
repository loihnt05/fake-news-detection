import asyncio
import asyncpg
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import time

load_dotenv()

# Configuration
# Increase this if you have high Tier OpenAI limits (e.g., Tier 2+ can handle 50+)
CONCURRENT_REQUESTS = 20 
DB_DSN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"

# Initialize async OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Stats container
class Stats:
    def __init__(self):
        self.completed = 0
        self.failed = 0
        self.total = 0

stats = Stats()

async def get_db_pool():
    """Create a database connection pool"""
    return await asyncpg.create_pool(
        dsn=DB_DSN,
        min_size=5,
        max_size=CONCURRENT_REQUESTS + 5  # Ensure enough connections for workers
    )

async def summarize_article(semaphore, pool, article_id, content):
    """Summarize a single article with rate limiting, retries, and async DB save"""
    async with semaphore:
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Pre-check content length
                content_text = (content or "")[:3500]
                if len(content_text) < 50: # Skip very short/empty content
                    return False

                # Call OpenAI
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You summarize Vietnamese articles briefly and clearly."},
                        {"role": "user", "content": f"Tóm tắt bài báo sau thành 2-3 câu:\n\n{content_text}"}
                    ],
                    max_tokens=150,
                    temperature=0.2
                )
                
                summary = response.choices[0].message.content
                
                # Update database asynchronously using the pool
                # Note: asyncpg uses $1, $2 syntax, not %s
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE articles SET description = $1 WHERE id = $2",
                        summary, article_id
                    )
                
                stats.completed += 1
                return True
                
            except Exception as e:
                error_str = str(e)
                
                # Rate limit handling
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait_time = base_delay * (2 ** attempt)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                        continue
                
                # Connection issues handling
                if "connection" in error_str.lower():
                     await asyncio.sleep(5) # Wait longer for DB connection issues
                     continue

                # Actual failures
                if attempt == max_retries - 1:
                    # print(f"Failed ID {article_id}: {error_str}") # Uncomment to see errors
                    stats.failed += 1
                
                return False

async def main():
    print(f"Starting optimized summarization with {CONCURRENT_REQUESTS} concurrent requests")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Initialize DB Pool
    try:
        pool = await get_db_pool()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    # 2. Fetch IDs and Content
    # We allow this specific call to be 'slow' to ensure we get all data first
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, content FROM articles WHERE description IS NULL OR description = ''"
        )
    
    stats.total = len(rows)
    print(f"Found {stats.total} articles to summarize\n")

    if stats.total == 0:
        await pool.close()
        return

    # 3. Process with Semaphore
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    # Create tasks
    tasks = [
        summarize_article(semaphore, pool, row['id'], row['content']) 
        for row in rows
    ]
    
    # Run with progress bar
    # Using return_exceptions=True so one crash doesn't stop the whole script
    await tqdm.gather(*tasks, desc="Processing")
    
    # 4. Cleanup
    await pool.close()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total articles: {stats.total}")
    print(f"✅ Completed: {stats.completed}")
    print(f"❌ Failed: {stats.failed}")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"📊 Speed: {stats.completed/elapsed*60:.1f} articles/minute")
    print("="*80)

if __name__ == "__main__":
    try:
        # Loop policy fix for Windows (if applicable)
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"Critical Error: {e}")