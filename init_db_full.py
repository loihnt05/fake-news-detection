"""
Script khởi tạo đầy đủ Database Schema cho Fact-Check System
Chạy lần đầu khi setup hệ thống:
    python init_db_full.py
"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

SCHEMA_SQL = """
-- =============================================
-- 1. ENABLE EXTENSIONS
-- =============================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- 2. ARTICLES TABLE (Tin tức gốc từ crawler)
-- =============================================
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT,
    published_date TIMESTAMP,
    scraped_at TIMESTAMP DEFAULT NOW(),
    category TEXT DEFAULT 'General',
    label TEXT DEFAULT '1',  -- '1' = Trusted source
    extracted_facts TEXT[],
    embedding vector(768)
);

CREATE INDEX IF NOT EXISTS articles_url_idx ON articles(url);
CREATE INDEX IF NOT EXISTS articles_embedding_idx 
    ON articles USING hnsw (embedding vector_cosine_ops);

-- =============================================
-- 3. CLAIMS TABLE (Các claim đã trích xuất từ bài báo)
-- =============================================
CREATE TABLE IF NOT EXISTS claims (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),
    
    -- Label từ hệ thống AI
    system_label TEXT DEFAULT 'UNDEFINED' CHECK (system_label IN ('REAL', 'FAKE', 'UNDEFINED')),
    verified BOOLEAN DEFAULT FALSE,
    source_type TEXT DEFAULT 'article' CHECK (source_type IN ('article', 'user', 'admin')),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS claims_article_idx ON claims(article_id);
CREATE INDEX IF NOT EXISTS claims_label_idx ON claims(system_label);
CREATE INDEX IF NOT EXISTS claims_embedding_idx 
    ON claims USING hnsw (embedding vector_cosine_ops);

-- =============================================
-- 4. USERS TABLE (Người dùng Extension)
-- =============================================
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,  -- UUID từ Extension
    role TEXT DEFAULT 'USER' CHECK (role IN ('USER', 'MODERATOR', 'ADMIN')),
    reputation_score FLOAT DEFAULT 0.5 CHECK (reputation_score >= 0 AND reputation_score <= 1),
    total_reports INTEGER DEFAULT 0,
    accepted_reports INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT NOW(),
    last_active_at TIMESTAMP DEFAULT NOW()
);

-- =============================================
-- 5. USER_REPORTS TABLE (Feedback từ user)
-- =============================================
CREATE TABLE IF NOT EXISTS user_reports (
    id SERIAL PRIMARY KEY,
    claim_id INTEGER REFERENCES claims(id) ON DELETE CASCADE,
    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    
    -- Feedback từ User
    user_feedback TEXT NOT NULL CHECK (user_feedback IN ('REAL', 'FAKE', 'UNSURE')),
    comment TEXT,
    
    -- Context để Retrain AI
    ai_label_at_report TEXT,  -- Label AI đưa ra lúc user báo cáo
    ai_confidence FLOAT,
    model_version TEXT,
    
    -- Trạng thái duyệt
    status TEXT DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED')),
    reviewed_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS reports_status_idx ON user_reports(status);
CREATE INDEX IF NOT EXISTS reports_user_idx ON user_reports(user_id);

-- =============================================
-- 6. MODEL_VERSIONS TABLE (Lịch sử các phiên bản model)
-- =============================================
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version TEXT UNIQUE NOT NULL,
    model_path TEXT,
    accuracy FLOAT,
    f1_score FLOAT,
    training_samples INTEGER,
    
    is_active BOOLEAN DEFAULT FALSE,
    trained_at TIMESTAMP DEFAULT NOW(),
    deployed_at TIMESTAMP
);

-- =============================================
-- 7. TRAINING_DATA TABLE (Dữ liệu đã duyệt để retrain)
-- =============================================
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    claim_text TEXT NOT NULL,
    evidence_text TEXT,
    label TEXT NOT NULL CHECK (label IN ('ENTAILMENT', 'CONTRADICTION', 'NEUTRAL')),
    
    source TEXT DEFAULT 'user_feedback',  -- 'user_feedback', 'admin', 'manual'
    report_id INTEGER REFERENCES user_reports(id),
    
    used_in_version TEXT,  -- Đã dùng train version nào
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS training_label_idx ON training_data(label);
"""

def init_database():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        
        print("✅ Kết nối Database thành công!")
        print(f"   Host: {DB_CONFIG['host']}")
        print(f"   Database: {DB_CONFIG['dbname']}")
        
        # Execute schema
        print("\n📦 Đang tạo Schema...")
        cur.execute(SCHEMA_SQL)
        
        print("✅ Đã tạo các bảng:")
        print("   ├─ articles (Bài báo gốc)")
        print("   ├─ claims (Các claim đã trích xuất)")
        print("   ├─ users (Người dùng Extension)")
        print("   ├─ user_reports (Feedback từ user)")
        print("   ├─ model_versions (Lịch sử model)")
        print("   └─ training_data (Dữ liệu retrain)")
        
        cur.close()
        conn.close()
        
        print("\n🎉 DATABASE ĐÃ SẴN SÀNG!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise

if __name__ == "__main__":
    init_database()
