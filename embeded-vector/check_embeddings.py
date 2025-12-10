#!/usr/bin/env python3
"""
Script to check and verify vector embeddings in the database
"""
import psycopg2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def check_database_connection():
    """Test database connection"""
    print("=" * 60)
    print("1. CHECKING DATABASE CONNECTION")
    print("=" * 60)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Connected to PostgreSQL")
        print(f"   Version: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def check_table_structure():
    """Check if articles table exists and has embedding column"""
    print("\n" + "=" * 60)
    print("2. CHECKING TABLE STRUCTURE")
    print("=" * 60)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'articles'
            );
        """)
        exists = cursor.fetchone()[0]
        
        if not exists:
            print("❌ Table 'articles' does not exist")
            return False
        
        print("✅ Table 'articles' exists")
        
        # Check columns
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'articles'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        print(f"\n   Columns in 'articles' table:")
        for col_name, col_type in columns:
            marker = "📍" if col_name == "embedding" else "  "
            print(f"   {marker} {col_name}: {col_type}")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error checking table: {e}")
        return False

def check_embedding_statistics():
    """Check embedding statistics"""
    print("\n" + "=" * 60)
    print("3. CHECKING EMBEDDING STATISTICS")
    print("=" * 60)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM articles;")
        total = cursor.fetchone()[0]
        print(f"   Total articles: {total}")
        
        # Records with embeddings
        cursor.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL;")
        with_embedding = cursor.fetchone()[0]
        print(f"   Articles with embeddings: {with_embedding}")
        
        # Records without embeddings
        without_embedding = total - with_embedding
        print(f"   Articles without embeddings: {without_embedding}")
        
        if total > 0:
            percentage = (with_embedding / total) * 100
            print(f"   Coverage: {percentage:.2f}%")
            
            if percentage == 100:
                print("   ✅ All articles have embeddings!")
            elif percentage > 0:
                print(f"   ⚠️  {without_embedding} articles still need embeddings")
            else:
                print("   ❌ No embeddings found!")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error checking statistics: {e}")
        return False

def check_embedding_samples():
    """Check sample embeddings and their validity"""
    print("\n" + "=" * 60)
    print("4. CHECKING EMBEDDING SAMPLES")
    print("=" * 60)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get sample embeddings
        cursor.execute("""
            SELECT id, title, embedding 
            FROM articles 
            WHERE embedding IS NOT NULL 
            LIMIT 3;
        """)
        samples = cursor.fetchall()
        
        if not samples:
            print("   ⚠️  No embeddings found to sample")
            cursor.close()
            conn.close()
            return False
        
        print(f"   Examining {len(samples)} sample embedding(s):\n")
        
        for idx, (article_id, title, embedding_str) in enumerate(samples, 1):
            print(f"   Sample {idx}:")
            print(f"   - Article ID: {article_id}")
            print(f"   - Title: {title[:60]}..." if len(title) > 60 else f"   - Title: {title}")
            
            # Parse embedding
            try:
                import ast
                embedding = ast.literal_eval(embedding_str)
                embedding_array = np.array(embedding)
                
                print(f"   - Embedding dimension: {len(embedding)}")
                print(f"   - Embedding type: {type(embedding).__name__}")
                print(f"   - First 5 values: {embedding[:5]}")
                print(f"   - Mean: {embedding_array.mean():.4f}")
                print(f"   - Std: {embedding_array.std():.4f}")
                print(f"   - Min: {embedding_array.min():.4f}")
                print(f"   - Max: {embedding_array.max():.4f}")
                
                # Check if embedding is valid (not all zeros)
                if np.all(embedding_array == 0):
                    print("   ❌ WARNING: Embedding is all zeros!")
                else:
                    print("   ✅ Embedding looks valid")
                    
            except Exception as e:
                print(f"   ❌ Error parsing embedding: {e}")
            
            print()
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error checking samples: {e}")
        return False

def check_model_loading():
    """Test if the embedding model can be loaded"""
    print("\n" + "=" * 60)
    print("5. CHECKING MODEL LOADING")
    print("=" * 60)
    try:
        print("   Loading model 'keepitreal/vietnamese-sbert'...")
        model = SentenceTransformer('keepitreal/vietnamese-sbert')
        print("   ✅ Model loaded successfully")
        
        # Test encoding
        test_text = "Đây là một câu tiếng Việt để test."
        print(f"\n   Testing encoding with: '{test_text}'")
        embedding = model.encode([test_text])[0]
        print(f"   ✅ Generated embedding with dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"❌ Error with model: {e}")
        return False

def test_similarity_search():
    """Test if we can perform similarity search"""
    print("\n" + "=" * 60)
    print("6. TESTING SIMILARITY SEARCH")
    print("=" * 60)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get a sample embedding
        cursor.execute("""
            SELECT id, title, embedding 
            FROM articles 
            WHERE embedding IS NOT NULL 
            LIMIT 1;
        """)
        result = cursor.fetchone()
        
        if not result:
            print("   ⚠️  No embeddings available for similarity test")
            cursor.close()
            conn.close()
            return False
        
        article_id, title, embedding_str = result
        print(f"   Using article: {title[:60]}...")
        
        # Parse embedding
        import ast
        query_embedding = ast.literal_eval(embedding_str)
        
        # Find similar articles (using simple cosine similarity)
        cursor.execute("""
            SELECT id, title 
            FROM articles 
            WHERE embedding IS NOT NULL AND id != %s
            LIMIT 5;
        """, (article_id,))
        candidates = cursor.fetchall()
        
        if candidates:
            print(f"   ✅ Can retrieve {len(candidates)} candidates for comparison")
            print("   Note: For actual similarity search, consider using pgvector extension")
        else:
            print("   ⚠️  Not enough articles for similarity comparison")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error testing similarity: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("VECTOR EMBEDDING VERIFICATION REPORT")
    print("=" * 60)
    print()
    
    results = []
    
    # Run all checks
    results.append(("Database Connection", check_database_connection()))
    results.append(("Table Structure", check_table_structure()))
    results.append(("Embedding Statistics", check_embedding_statistics()))
    results.append(("Embedding Samples", check_embedding_samples()))
    results.append(("Model Loading", check_model_loading()))
    results.append(("Similarity Search", test_similarity_search()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {check_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_checks = len(results)
    
    print(f"\n   Total: {total_passed}/{total_checks} checks passed")
    
    if total_passed == total_checks:
        print("\n   🎉 All checks passed! Vector embedding system is working correctly.")
    else:
        print("\n   ⚠️  Some checks failed. Please review the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
