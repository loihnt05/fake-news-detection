from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from verifier import FactChecker
from fastapi.middleware.cors import CORSMiddleware # <--- Thêm import này
app = FastAPI(title="Fake News Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi nguồn (trong production nên giới hạn lại)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Khởi tạo Engine (Load model 1 lần duy nhất khi bật app)
checker = FactChecker()

# --- INPUT SCHEMAS ---
class CheckRequest(BaseModel):
    text: str # Câu cần check

class VerifyRequest(BaseModel):
    title: str
    content: str

class VerifyClaimRequest(BaseModel):
    claim_id: int
    label: str # REAL / FAKE

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "running", "model": "PhoBERT V6 Hard Negative"}

@app.post("/check")
def check_news(req: CheckRequest):
    """
    API chính: Nhận vào text, trả về FAKE/REAL
    """
    try:
        result = checker.check_claim(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/verify")
def verify_article(req: VerifyRequest):
    """
    API cho Extension: Nhận vào title và content, trả về kết quả phân tích
    """
    try:
        # Kết hợp title và content để phân tích
        full_text = f"{req.title}\n\n{req.content}"
        result = checker.check_article(full_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/verify")
def admin_verify_claim(req: VerifyClaimRequest):
    """
    API dành cho Admin/Expert:
    Chuyển trạng thái claim từ UNDEFINED -> REAL để làm kiến thức nền.
    """
    try:
        with checker.conn.cursor() as cur:
            cur.execute("""
                UPDATE claims 
                SET system_label = %s, verified = TRUE, trust_score = 1.0
                WHERE id = %s
            """, (req.label, req.claim_id))
        return {"message": "Updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Chạy server tại localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)