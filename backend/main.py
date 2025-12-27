import uvicorn
import psycopg2
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

from backend.verifier import AdvancedFactChecker

# --- DB CONFIG ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

checker_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global checker_instance
    checker_instance = AdvancedFactChecker()
    yield
    checker_instance = None

app = FastAPI(title="Fact-Check API Pro", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMAS (Chuẩn Production) ---

class EvidenceDetail(BaseModel):
    claim_id: Optional[int] = None # ID trong DB
    claim: str
    status: str
    score: float
    evidence: str

class VerificationResult(BaseModel):
    status: str
    confidence: float
    explanation: str
    model_version: str
    details: List[EvidenceDetail]

class NewsRequest(BaseModel):
    text: str

# Schema cho Report (Context đầy đủ)
class UserReportRequest(BaseModel):
    user_id: str          # UUID từ Extension
    claim_id: int         # ID của claim trong DB
    
    feedback: str         # REAL / FAKE / UNSURE
    comment: Optional[str] = None
    
    # Context để Retrain AI
    ai_label: str
    ai_confidence: float
    model_version: str

# --- ENDPOINTS ---

@app.post("/api/v1/verify", response_model=VerificationResult)
def verify_news(request: NewsRequest):
    if not checker_instance: raise HTTPException(503, "Loading...")
    return checker_instance.verify(request.text)

@app.post("/api/v1/report")
def report_news(req: UserReportRequest):
    """
    Endpoint nhận Feedback với đầy đủ Context
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            # 1. UPSERT USER (Nếu chưa có user_id này thì tạo mới)
            cur.execute("""
                INSERT INTO users (id, role, reputation_score)
                VALUES (%s, 'USER', 0.5)
                ON CONFLICT (id) DO UPDATE 
                SET last_active_at = NOW(); -- Chỉ update thời gian online
            """, (req.user_id,))
            
            # 2. INSERT REPORT
            cur.execute("""
                INSERT INTO user_reports 
                (claim_id, user_id, user_feedback, comment, 
                 ai_label_at_report, ai_confidence, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                req.claim_id, req.user_id, req.feedback, req.comment,
                req.ai_label, req.ai_confidence, req.model_version
            ))
            
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Feedback received for analysis."}
    except Exception as e:
        print(f"Report Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT ADMIN (Giả lập quy trình duyệt) ---
class ApprovalRequest(BaseModel):
    report_id: str
    verdict: str # APPROVED / REJECTED

@app.post("/api/v1/admin/approve-report")
def approve_report(req: ApprovalRequest):
    """
    Hàm này sẽ được gọi từ Admin Dashboard.
    Khi Admin bấm DUYỆT -> Cập nhật Reputation cho User.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            # 1. Update trạng thái Report
            cur.execute("""
                UPDATE user_reports 
                SET status = %s, reviewed_at = NOW()
                WHERE id = %s
                RETURNING user_id;
            """, (req.verdict, req.report_id))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Report not found")
            
            user_id = row[0]

            # 2. UPDATE REPUTATION (Logic thưởng phạt)
            if req.verdict == 'APPROVED':
                # Tăng uy tín (+0.1, max 1.0)
                cur.execute("""
                    UPDATE users 
                    SET reputation_score = LEAST(reputation_score + 0.1, 1.0),
                        accepted_reports = accepted_reports + 1,
                        total_reports = total_reports + 1
                    WHERE id = %s
                """, (user_id,))
            elif req.verdict == 'REJECTED':
                # Giảm uy tín (-0.05, min 0.0)
                cur.execute("""
                    UPDATE users 
                    SET reputation_score = GREATEST(reputation_score - 0.05, 0.0),
                        total_reports = total_reports + 1
                    WHERE id = %s
                """, (user_id,))
                
        conn.commit()
        conn.close()
        return {"message": f"Report {req.verdict}. User reputation updated."}
    except Exception as e:
        raise HTTPException(500, str(e))
    
# API nội bộ để Airflow gọi khi Retrain xong
@app.post("/api/internal/reload-model")
def trigger_reload_model(secret_key: str):
    # Bảo mật đơn giản để người ngoài không gọi bừa
    if secret_key != "SUPER_SECRET_AIRFLOW_KEY": 
        raise HTTPException(403, "Forbidden")
    
    # Giả sử quy trình retrain luôn lưu model vào thư mục 'latest' hoặc theo version
    # Ở đây ta load model mới nhất vừa train xong
    NEW_MODEL_PATH = "model/phobert_v8_finetuned" 
    
    if checker_instance:
        success = checker_instance.reload_model(NEW_MODEL_PATH)
        if success:
            return {"status": "success", "message": "Model reloaded successfully"}
    
    raise HTTPException(500, "Failed to reload model")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)