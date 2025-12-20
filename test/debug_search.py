import psycopg2
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def debug_query(text_query):
    print(f"🔎 Query: '{text_query}'")
    
    # Load model (chỉ cần CPU cho nhanh)
    model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device='cpu')
    vec = model.encode(text_query)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Lấy top 5 bất chấp threshold
    query = """
        SELECT content, (embedding <=> %s::vector) as distance
        FROM sentence_store
        ORDER BY distance ASC
        LIMIT 5;
    """
    cur.execute(query, (vec.tolist(),))
    results = cur.fetchall()
    
    print("-" * 50)
    for res in results:
        content, dist = res
        print(f"Dist: {dist:.4f} | Text: {content}")
    print("-" * 50)

if __name__ == "__main__":
    # Nhập câu bạn muốn test xem DB có không
    debug_query("Thổ Nhĩ Kỳ điều máy bay sơ tán công dân")